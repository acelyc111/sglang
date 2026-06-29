from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    ComponentType,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.unified_cache_components.swa_component import SWAComponent
    from sglang.srt.mem_cache.unified_radix_cache import (
        UnifiedRadixCache,
    )


class WindowedSWAComponent(SWAComponent):
    """Windowed sliding window attention component."""

    def __init__(self, cache: UnifiedRadixCache, params: CacheInitParams):
        super().__init__(cache, params)

    component_type = ComponentType.WINDOWED_SWA

    # ---- HiCache Hooks ----
