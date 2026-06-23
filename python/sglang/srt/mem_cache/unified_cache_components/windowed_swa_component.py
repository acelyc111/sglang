from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    InsertResult,
)
from sglang.srt.mem_cache.hicache_storage import (
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    BASE_COMPONENT_TYPE,
    CacheTransferPhase,
    ComponentType,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.unified_cache_components.swa_component import SWAComponent
    from sglang.srt.mem_cache.unified_radix_cache import (
        UnifiedRadixCache,
        UnifiedTreeNode,
    )


class WindowedSWAComponent(SWAComponent):
    """Windowed sliding window attention component."""

    def __init__(self, cache: UnifiedRadixCache, params: CacheInitParams):
        super().__init__(cache, params)

    component_type = ComponentType.WINDOWED_SWA

    # ---- HiCache Hooks ----

    def commit_hicache_transfer(
        self,
        node: UnifiedTreeNode,
        phase: CacheTransferPhase,
        transfers: list[PoolTransfer] = (),
        *,
        insert_result: Optional[InsertResult] = None,
        pool_storage_result: Optional[PoolTransferResult] = None,
    ) -> None:
        ct = self.component_type

        if phase == CacheTransferPhase.BACKUP_HOST:
            if transfers and transfers[0].host_indices is not None:
                cd = node.component_data[ct]
                if cd.host_value is None:
                    cd.host_value = transfers[0].host_indices.clone()
            return

        if phase == CacheTransferPhase.LOAD_BACK:
            assert transfers and transfers[0].device_indices is not None
            xfer = transfers[0]
            device_indices = xfer.device_indices
            allocator = self.cache.token_to_kv_pool_allocator

            offset = 0
            for n in xfer.nodes_to_load or []:
                cd_n = n.component_data[ct]
                cd_full_n = n.component_data[BASE_COMPONENT_TYPE]
                n_tokens = len(cd_n.host_value)
                swa_chunk = device_indices[offset : offset + n_tokens].clone()
                self._restore_device_value(n, swa_chunk)
                assert cd_full_n.value is not None and len(cd_full_n.value) == n_tokens
                # rebuild the mapping for the loaded SWA chunk
                allocator.set_full_to_swa_mapping(cd_full_n.value, swa_chunk)
                offset += n_tokens
            assert offset == len(xfer.host_indices)
            return

        if phase == CacheTransferPhase.PREFETCH:
            self._commit_prefetch(
                node,
                transfers,
                insert_result=insert_result,
                pool_storage_result=pool_storage_result,
            )
            return

    def _release_swa_host(self, host_indices: torch.Tensor) -> None:
        if host_indices is not None and host_indices.numel() > 0:
            self.cache.cache_controller.append_host_mem_release(
                extra_pools=[PoolTransfer(name=PoolName.SWA, host_indices=host_indices)]
            )

    def _attach_swa_host_value(
        self, node: UnifiedTreeNode, host_indices: torch.Tensor
    ) -> None:
        """Write host_indices into node's SWA host_value and refresh tree state."""
        ct = self.component_type
        cd = node.component_data[ct]
        cd.host_value = host_indices.clone()
        host_lru = self.cache.host_lru_lists[ct]
        if cd.value is None and not host_lru.in_list(node):
            host_lru.insert_mru(node)
        self.cache._update_evictable_leaf_sets(node)
        if node.parent:
            self.cache._update_evictable_leaf_sets(node.parent)

    def _commit_prefetch(
        self,
        anchor,
        transfers: list[PoolTransfer],
        *,
        insert_result: Optional[InsertResult] = None,
        pool_storage_result: Optional[PoolTransferResult] = None,
    ) -> None:
        """Fill the prefetched SWA window onto the leaf→anchor path.

        All-or-nothing over one full window: ``loaded_pages`` is the cross-rank
        MIN, so ``loaded_pages < window_pages`` drops the whole window (keeps the
        tree identical across TP ranks). Otherwise map the buffer to token range
        ``[loaded_start, total_len)`` and walk leaf→anchor, filling SWA
        tombstones and releasing slices that already have host_value.
        """
        if not transfers:
            return
        ct = self.component_type
        page_size = self.cache.page_size
        host_indices = transfers[0].host_indices
        window_require_pages = (
            host_indices.numel() // page_size if host_indices is not None else 0
        )
        loaded_pages = (
            pool_storage_result.extra_pool_hit_pages.get(PoolName.SWA, 0)
            if pool_storage_result
            else 0
        )
        target = insert_result.inserted_host_node if insert_result else None
        if (
            target is None
            or window_require_pages == 0
            or loaded_pages < window_require_pages
        ):
            self._release_swa_host(host_indices)
            return

        # Buffer covers token range [loaded_start, total_len).
        loaded_start = insert_result.total_len - window_require_pages * page_size

        # Walk leaf → anchor; ``pos`` is the right edge of ``cur`` in tokens.
        pos, cur = insert_result.total_len, target
        while cur is not anchor and pos > loaded_start:
            node_start = pos - len(cur.key)
            # Intersection of cur's range and the buffer.
            fill_start = max(node_start, loaded_start)
            fill_len = pos - fill_start
            buf_off = fill_start - loaded_start
            slice_ = host_indices[buf_off : buf_off + fill_len]

            cd = cur.component_data[ct]
            if cd.host_value is None and fill_len > 0:
                # Tombstone: split off the in-buffer tail if needed, then fill.
                if fill_start > node_start:
                    self.cache._split_node(cur.key, cur, fill_start - node_start)
                self._attach_swa_host_value(cur, slice_)
            else:
                # Already has SWA (or empty overlap): drop this slice.
                self._release_swa_host(slice_)

            pos = node_start
            cur = cur.parent

        # Buffer prefix that fell outside the anchor→leaf path.
        if pos > loaded_start:
            self._release_swa_host(host_indices[: pos - loaded_start])
