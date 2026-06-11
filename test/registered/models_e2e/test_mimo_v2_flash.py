import time
import unittest

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.spec_decoding_kit import SpecDecodingMixin
from sglang.test.kl_multiturn_utils import (
    _extract_output_logprobs,
    _flush_cache,
    _generate,
    _get_input_logprobs,
    compare_kl_divergence,
)
from sglang.test.server_fixtures.default_fixture import DefaultServerBase
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=900, stage="base-c", runner_config="8-gpu-h200")

MIMO_V2_FLASH_MODEL = "XiaomiMiMo/MiMo-V2-Flash"
MIMO_V2_FLASH_LAUNCH_TIMEOUT = 3600
MIMO_V2_FLASH_MODEL_LOADER_CONFIG = (
    '{"enable_multithread_load": true,"num_threads": 64}'
)


def _mimo_v2_flash_args(
    *,
    max_running_requests: int,
    cuda_graph_max_bs: int,
    page_size: int = 64,
    extra_args: list[str] | None = None,
) -> list[str]:
    args = [
        "--tp",
        "4",
        "--dp",
        "2",
        "--enable-dp-attention",
        "--trust-remote-code",
        "--attention-backend",
        "fa3",
        "--max-running-requests",
        str(max_running_requests),
        "--cuda-graph-max-bs-decode",
        str(cuda_graph_max_bs),
        "--page-size",
        str(page_size),
        "--mem-fraction-static",
        "0.75",
        "--model-loader-extra-config",
        MIMO_V2_FLASH_MODEL_LOADER_CONFIG,
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--enable-multi-layer-eagle",
        "--enable-hierarchical-cache",
        "--hicache-mem-layout",
        "page_first",
        "--hicache-io-backend",
        "kernel",
    ]
    if extra_args:
        args.extend(extra_args)
    return args


def _make_cache_pressure_input_ids(
    num_samples: int,
    prompt_len: int,
) -> list[list[int]]:
    # Use deterministic local token ids instead of downloading LongBench data.
    # The test only needs valid token sequences long enough to exercise radix
    # cache eviction and HiCache L2 load-back.
    return [
        [1000 + sample_id * prompt_len + token_id for token_id in range(prompt_len)]
        for sample_id in range(num_samples)
    ]


class TestMiMoV2Flash(GSM8KMixin, SpecDecodingMixin, DefaultServerBase):
    gsm8k_accuracy_thres = 0.8
    gsm8k_num_questions = 1319
    gsm8k_num_threads = 1319
    model = MIMO_V2_FLASH_MODEL

    other_args = _mimo_v2_flash_args(
        max_running_requests=128,
        cuda_graph_max_bs=64,
        extra_args=[
            "--hicache-ratio",
            "1.5",
        ],
    )

    bs_1_speed_thres = 170
    accept_length_thres = 3.2

    @classmethod
    def setUpClass(cls):
        with envs.SGLANG_ENABLE_UNIFIED_RADIX_TREE.override(True):
            super().setUpClass()


class TestMiMoV2FlashHiCacheL2Accuracy(CustomTestCase):
    model = MIMO_V2_FLASH_MODEL
    base_url = DEFAULT_URL_FOR_TEST
    page_size = 64
    kl_threshold = 0.005

    @classmethod
    def setUpClass(cls):
        with envs.SGLANG_ENABLE_UNIFIED_RADIX_TREE.override(True):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=MIMO_V2_FLASH_LAUNCH_TIMEOUT,
                other_args=_mimo_v2_flash_args(
                    max_running_requests=4,
                    cuda_graph_max_bs=4,
                    page_size=cls.page_size,
                    extra_args=[
                        "--max-total-tokens",
                        "20000",
                        "--hicache-write-policy",
                        "write_through",
                        "--enable-cache-report",
                        "--hicache-ratio",
                        "8",
                    ],
                ),
            )
        cls.input_ids = _make_cache_pressure_input_ids(
            num_samples=12,
            prompt_len=4096,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_l2_hit_logprobs_match(self):
        target_input_ids = self.input_ids[:4]
        pressure_input_ids = self.input_ids[4:]

        # First pass fills radix cache and write-through backs target KV to L2.
        _flush_cache(self.base_url)
        first_results = _generate(
            self.base_url,
            target_input_ids,
            max_new_tokens=8,
            return_logprob=True,
            temperature=0,
        )
        self.assertEqual(len(first_results), len(target_input_ids))
        time.sleep(5)

        # Pressure prompts exceed the small device KV budget and evict target KV
        # from L1/device while its write-through copy remains in L2/host.
        pressure_results = _generate(
            self.base_url,
            pressure_input_ids,
            max_new_tokens=8,
            temperature=0,
        )
        self.assertEqual(len(pressure_results), len(pressure_input_ids))

        # Replaying target prompts should now load matching pages back from L2.
        l2_hit_results = _generate(
            self.base_url,
            target_input_ids,
            max_new_tokens=8,
            return_logprob=True,
            temperature=0,
        )
        self.assertEqual(len(l2_hit_results), len(target_input_ids))

        # Only samples that actually report host hits participate in KL replay.
        host_hit_replay_input_ids = []
        host_hit_output_logprobs = []
        host_cached_tokens = 0
        cached_details_by_result = []
        for input_ids, result in zip(target_input_ids, l2_hit_results):
            cached_details = result["meta_info"].get("cached_tokens_details") or {}
            cached_details_by_result.append(cached_details)
            host_tokens = int(cached_details.get("host", 0))
            host_cached_tokens += host_tokens
            if host_tokens >= self.page_size:
                host_hit_replay_input_ids.append(input_ids + result["output_ids"])
                host_hit_output_logprobs.append(_extract_output_logprobs(result))
        self.assertGreaterEqual(
            host_cached_tokens,
            self.page_size,
            "Expected the second target pass to load at least one page from L2 "
            f"HiCache, got {host_cached_tokens=}, {cached_details_by_result=}",
        )
        self.assertGreater(
            len(host_hit_replay_input_ids),
            0,
            f"Expected at least one per-sample L2 hit, got {cached_details_by_result=}",
        )
        input_logprobs = _get_input_logprobs(
            self.base_url,
            host_hit_replay_input_ids,
            host_hit_output_logprobs,
            temperature=0,
        )
        # Compare L2-hit generation logprobs with a fresh prefill replay.
        compare_kl_divergence(
            input_logprobs,
            host_hit_output_logprobs,
            {self.model: {"kl_div": self.kl_threshold}},
            self.model,
            self.test_l2_hit_logprobs_match.__name__,
        )


if __name__ == "__main__":
    unittest.main()
