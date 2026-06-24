from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Union

import torch
from einops import rearrange

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

from sglang.srt.layers.attention.triton_ops.lightning_attn import (
    lightning_attention,
    linear_decode_forward_triton,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInfo


@dataclass
class LightningAttentionMetadata:
    """
    Attention metadata for lightning attention backend.
    Used for both prefill and decode batched together.
    """

    num_prefills: int
    num_prefill_tokens: int
    num_decode_tokens: int
    query_start_loc: Optional[torch.Tensor] = None
    context_lens_tensor: Optional[torch.Tensor] = None
    rotary_emb: Optional[Any] = None

    # Lightning-specific metadata
    kv_caches: Optional[torch.Tensor] = None
    state_indices_tensor: Optional[torch.Tensor] = None
    slope_rates: Optional[torch.Tensor] = None

    @property
    def prefill_metadata(self) -> Optional["LightningAttentionMetadata"]:
        return self

    @property
    def decode_metadata(self) -> Optional["LightningAttentionMetadata"]:
        return self

    def asdict_zerocopy(self, skip_fields: Optional[Set[str]] = None) -> Dict[str, Any]:
        """Similar to dataclasses.asdict, but avoids deepcopying."""
        if skip_fields is None:
            skip_fields = set()
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name not in skip_fields
        }


class LightningAttentionBackend(AttentionBackend):
    """
        Lightning attention backend for efficient linear attention computation.
        """

    def __init__(
        self,
        model_runner: ModelRunner,
        num_heads: int,
        head_dim: int,
        max_context_len: int,
        block_size: int = 256,
    ):
        super().__init__()

        self.model_runner = model_runner
        self.device = model_runner.device
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_context_len = max_context_len
        self.block_size = block_size

        self.slope_rates = None

        # Forward metadata
        self.forward_metadata: Optional[LightningAttentionMetadata] = None

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Initialize metadata from ForwardBatch."""
        # Ensure slope_rates is initialized for non-CUDA graph scenarios
        if self.slope_rates is None:
            self.slope_rates = build_slope_tensor(self.num_heads)

        metadata = self._create_metadata_from_forward_batch(forward_batch)
        self.forward_metadata = metadata

    def _create_metadata_from_forward_batch(
        self, forward_batch: ForwardBatch
    ) -> LightningAttentionMetadata:
        """Create LightningAttentionMetadata from ForwardBatch."""

        batch_size = forward_batch.batch_size
        seq_lens = forward_batch.seq_lens
        device = seq_lens.device

        # Check if we have extend information
        extend_seq_lens = getattr(forward_batch, "extend_seq_lens", None)

        # Determine if this is a prefill batch
        is_prefill = False

        # Handle forward_mode safely - it might be None in some cases
        if (
            hasattr(forward_batch, "forward_mode")
            and forward_batch.forward_mode is not None
        ):
            is_prefill = forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed()
        elif extend_seq_lens is not None:
            is_prefill = (extend_seq_lens > 1).any().item()
        # If forward_mode is None and extend_seq_lens is None, default to decode mode (is_prefill=False)

        if is_prefill or (
            extend_seq_lens is not None and extend_seq_lens.sum() > batch_size
        ):
            # Prefill/Extend mode
            if extend_seq_lens is not None:
                actual_extend_lens = extend_seq_lens
            else:
                actual_extend_lens = torch.ones(
                    batch_size, dtype=torch.int32, device=device
                )

            num_prefills = batch_size
            num_prefill_tokens = actual_extend_lens.sum().item()
            num_decode_tokens = 0

            query_start_loc = torch.nn.functional.pad(
                torch.cumsum(actual_extend_lens, dim=0, dtype=torch.int32), (1, 0)
            )
        else:
            # Pure decode mode
            num_prefills = 0
            num_prefill_tokens = 0
            num_decode_tokens = batch_size

            query_start_loc = torch.arange(
                0, batch_size + 1, dtype=torch.int32, device=device
            )

        # Set context lengths
        context_lens_tensor = seq_lens.to(torch.int32)

        # Get kv caches and state indices if available
        kv_caches = None
        state_indices_tensor = None
        if hasattr(forward_batch, "kv_caches"):
            kv_caches = forward_batch.kv_caches
        if hasattr(forward_batch, "state_indices_tensor"):
            state_indices_tensor = forward_batch.state_indices_tensor

        metadata = LightningAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            query_start_loc=query_start_loc,
            context_lens_tensor=context_lens_tensor,
            rotary_emb=None,
            kv_caches=kv_caches,
            state_indices_tensor=state_indices_tensor,
            slope_rates=self.slope_rates,  # Use pre-allocated slope_rates
        )

        return metadata

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        """Initialize CUDA graph state and pre-allocate slope rates."""
        # Pre-allocate slope rates to avoid device operations during CUDA graph capture
        if self.slope_rates is None:
            self.slope_rates = build_slope_tensor(self.num_heads)

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInfo],
    ):
        """Initialize metadata for CUDA graph capture."""

        class DummyForwardBatch:
            def __init__(self):
                self.batch_size = bs
                self.seq_lens = seq_lens
                self.forward_mode = forward_mode
                self.extend_seq_lens = None

        dummy_batch = DummyForwardBatch()
        self.forward_metadata = self._create_metadata_from_forward_batch(dummy_batch)

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInfo],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        """Initialize metadata for CUDA graph replay."""

        # Similar to capture, but might need to update some fields
        class DummyForwardBatch:
            def __init__(self):
                self.batch_size = bs
                self.seq_lens = seq_lens
                self.forward_mode = forward_mode  # Use the provided forward_mode
                self.extend_seq_lens = None

        dummy_batch = DummyForwardBatch()
        self.forward_metadata = self._create_metadata_from_forward_batch(dummy_batch)

    def get_cuda_graph_seq_len_fill_value(self):
        """Get fill value for padded sequence lengths."""
        return 1

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        """Forward pass for extend/prefill mode."""
        if self.forward_metadata is None:
            self.init_forward_metadata(forward_batch)

        # Save KV cache if requested
        if save_kv_cache and k is not None and v is not None:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        metadata = self.forward_metadata

        # Use the prefill and mixed inference from original code
        return self._prefill_and_mix_infer(
            q, k, v, metadata.kv_caches, metadata.state_indices_tensor, metadata
        )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        """Forward pass for decode mode."""
        if self.forward_metadata is None:
            self.init_forward_metadata(forward_batch)

        # Save KV cache if requested
        if save_kv_cache and k is not None and v is not None:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        metadata = self.forward_metadata

        # Use the decode inference from original code
        return self._decode_infer(
            q, k, v, metadata.kv_caches, metadata.state_indices_tensor, metadata
        )

    def support_triton(self) -> bool:
        """Check if the current backend supports triton."""
        return True

    def _prefill_and_mix_infer(
        self, q, k, v, kv_cache, state_indices_tensor, attn_metadata
    ):
        """Handle prefill and mixed mode inference."""
        hidden = []

        # Handle prefill tokens
        num_prefills = getattr(attn_metadata, "num_prefills", 0)
        query_start_loc = getattr(attn_metadata, "query_start_loc", None)

        for _prefill_idx in range(num_prefills):
            # Safety checks for indexing
            if (
                query_start_loc is None
                or _prefill_idx >= len(query_start_loc) - 1
                or state_indices_tensor is None
                or _prefill_idx >= len(state_indices_tensor)
            ):
                break

            _start = query_start_loc[_prefill_idx]
            _end = query_start_loc[_prefill_idx + 1]

            # Ensure valid slice bounds
            if _start >= _end or _end > q.shape[0]:
                continue

            slot_id = state_indices_tensor[_prefill_idx]

            qs = q[_start:_end].transpose(0, 1).contiguous()
            ks = k[_start:_end].transpose(0, 1).contiguous()
            vs = v[_start:_end].transpose(0, 1).contiguous()

            # Safety check for cache indexing
            if slot_id < 0 or slot_id >= kv_cache.shape[0]:
                continue

            slice_layer_cache = kv_cache[slot_id, ...]

            out_slice = self._lightning_forward_prefix(
                qs,
                ks,
                vs,
                slice_layer_cache,
                attn_metadata.slope_rates,
                self.block_size,
            )
            hidden.append(out_slice.contiguous())

        # Handle decode tokens if any
        if getattr(attn_metadata, "num_decode_tokens", 0) > 0:
            hidden.append(
                self._decode_infer(
                    q, k, v, kv_cache, state_indices_tensor, attn_metadata
                )
            )

        if not hidden:
            return torch.empty((0, q.shape[-1]), device=q.device, dtype=q.dtype)

        return torch.concat(hidden, dim=0).contiguous()

    def _decode_infer(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: torch.Tensor,
        state_indices_tensor: torch.Tensor,
        attn_metadata,
    ) -> torch.Tensor:
        """Handle decode mode inference."""
        num_prefill_tokens = getattr(attn_metadata, "num_prefill_tokens", 0)
        num_prefills = getattr(attn_metadata, "num_prefills", 0)

        if num_prefill_tokens > 0:
            q = q[num_prefill_tokens:].unsqueeze(2).contiguous()
            k = k[num_prefill_tokens:].unsqueeze(2).contiguous()
            v = v[num_prefill_tokens:].unsqueeze(2).contiguous()
        else:
            q = q.unsqueeze(2).contiguous()
            k = k.unsqueeze(2).contiguous()
            v = v.unsqueeze(2).contiguous()

        # Ensure safe indexing for state_indices_tensor
        if (
            state_indices_tensor is not None
            and len(state_indices_tensor) > num_prefills
        ):
            slot_id = state_indices_tensor[num_prefills:]
        else:
            # Fallback: create dummy slot indices
            decode_batch_size = q.shape[0]
            slot_id = torch.arange(decode_batch_size, dtype=torch.long, device=q.device)

        # Assert that slot_id length matches the expected decode batch size
        assert len(slot_id) == q.shape[0], (
            f"slot_id length {len(slot_id)} does not match decode batch size {q.shape[0]}. "
            "This indicates a bug in the upstream logic that should be investigated."
        )

        hidden = linear_decode_forward_triton(
            q, k, v, kv_cache, attn_metadata.slope_rates, slot_id, 32
        )
        return hidden

    def _lightning_forward_prefix(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_caches: torch.Tensor,
        slope_rate: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        """Lightning attention forward for prefix tokens."""
        slope_rate = slope_rate.to(torch.float32)
        should_pad_dim = q.dim() == 3
        if should_pad_dim:
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)

        b, h, n, d = q.shape
        e = v.shape[-1]
        kv_history = kv_caches.reshape(1, h, d, e).contiguous()

        output, kv_history = lightning_attention(
            q, k, v, slope_rate, block_size=block_size, kv_history=kv_history
        )

        kv_caches.copy_(kv_history[:, :, -1, :, :].reshape(h, d, e))
        assert output.shape[0] == 1, "batch size must be 1"
        return rearrange(output.squeeze(0), "h n d -> n (h d)")


def build_slope_tensor(n_attention_heads: int) -> torch.Tensor:
    """
    Build slope tensor for lightning attention decay rates.

    This function generates slope values for linear attention mechanisms,
    following the approach described in the Lightning Attention paper.

    Args:
        n_attention_heads: Number of attention heads

    Returns:
        torch.Tensor: Slope tensor with shape (n_attention_heads, 1, 1)
    """
    import math

    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

    slopes = torch.tensor(get_slopes(n_attention_heads), dtype=torch.float32).reshape(
        n_attention_heads, 1, 1
    )
    return slopes
