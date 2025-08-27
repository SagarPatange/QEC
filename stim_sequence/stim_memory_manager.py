
# =====================================================================
# stim_memory_manager.py
# ---------------------------------------------------------------------
# Lean manager for Stim keys (StimMemory) that mirrors MemoryManager.
# Adds get_info_by_link(...) for quick lookup from a link id.
# =====================================================================

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, List, Optional, Tuple, Union

from stim_memory import (
    StimMemory, StimMemoryArray,
    LinkLedger, EntanglementLink
)


class MemState(str, Enum):
    RAW = "RAW"
    OCCUPIED = "OCCUPIED"
    ENTANGLED = "ENTANGLED"


@dataclass
class StimMemoryInfo:
    memory: StimMemory
    index: int
    state: MemState = MemState.RAW
    link_id: Optional[str] = None
    remote_node: Optional[str] = None
    remote_key: Optional[Union[int, str]] = None
    entangle_time_ps: int = -1

    def is_raw(self) -> bool: return self.state == MemState.RAW
    def is_occupied(self) -> bool: return self.state == MemState.OCCUPIED
    def is_entangled(self) -> bool: return self.state == MemState.ENTANGLED


class StimMemoryManager:
    def __init__(self, memory_array: StimMemoryArray, link_ledger: LinkLedger):
        self.array = memory_array
        self.links = link_ledger
        self._map: List[StimMemoryInfo] = [
            StimMemoryInfo(memory=m, index=i) for i, m in enumerate(self.array.memories)
        ]
        self.resource_manager = None
        # optional fast reverse-lookup for link_id -> index
        self._link_to_index: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._map)

    def __getitem__(self, i: int) -> StimMemoryInfo:
        return self._map[i]

    def get_info_by_memory(self, memory: StimMemory) -> StimMemoryInfo:
        return self._map[memory.index]

    def get_memory_by_name(self, name: str) -> StimMemory:
        return self.array.get_memory_by_name(name)

    def get_info_by_link(self, link_id: str) -> StimMemoryInfo:
        """Find the local StimMemoryInfo that mirrors this link id."""
        idx = self._link_to_index.get(link_id)
        if idx is not None:
            return self._map[idx]
        # Fallback scan for robustness
        for info in self._map:
            if info.link_id == link_id:
                return info
        raise KeyError(f"Link {link_id} not mirrored on any local key")

    def iter_by_state(self, state: MemState) -> Iterable[StimMemoryInfo]:
        for info in self._map:
            if info.state == state:
                yield info

    def update(self, memory: StimMemory, state: Union[str, MemState]) -> None:
        info = self.get_info_by_memory(memory)
        state = MemState(state)
        if state == MemState.RAW:
            self.to_raw(info)
        elif state == MemState.OCCUPIED:
            self.to_occupied(info)
        elif state == MemState.ENTANGLED:
            self.to_entangled(info, link=None)
        else:
            raise ValueError(f"Unknown state: {state}")

    def to_raw(self, info: StimMemoryInfo) -> None:
        info.state = MemState.RAW
        if info.link_id:
            self._link_to_index.pop(info.link_id, None)
        info.link_id = None
        info.remote_node = None
        info.remote_key = None
        info.entangle_time_ps = -1
        # optional: info.memory.reset()

    def to_occupied(self, info: StimMemoryInfo) -> None:
        info.state = MemState.OCCUPIED

    def to_entangled(self, info: StimMemoryInfo, link: Optional[EntanglementLink]) -> None:
        info.state = MemState.ENTANGLED
        if link is not None:
            info.link_id = link.link_id
            info.remote_node = link.remote.node
            info.remote_key = link.remote.key
            info.entangle_time_ps = link.born_ps
            self._link_to_index[link.link_id] = info.index

    def claim_one(
        self,
        predicate: Optional[Callable[[StimMemory], bool]] = None
    ) -> Optional[StimMemoryInfo]:
        for info in self.iter_by_state(MemState.RAW):
            if predicate is None or predicate(info.memory):
                self.to_occupied(info)
                return info
        return None

    def release(self, item: Union[StimMemoryInfo, StimMemory]) -> None:
        info = item if isinstance(item, StimMemoryInfo) else self.get_info_by_memory(item)
        self.to_raw(info)

    def bind_link(self, memory: StimMemory, link_id: str) -> StimMemoryInfo:
        info = self.get_info_by_memory(memory)
        link = self.links.get(link_id)
        if link.local.index != info.index or link.local.array_name != memory.array.name:
            raise ValueError("Link does not belong to this memory")
        self.to_entangled(info, link=link)
        return info

    def clear_link(self, item: Union[StimMemoryInfo, StimMemory]) -> None:
        info = item if isinstance(item, StimMemoryInfo) else self.get_info_by_memory(item)
        if info.link_id:
            self._link_to_index.pop(info.link_id, None)
        info.link_id = None
        info.remote_node = None
        info.remote_key = None
        info.entangle_time_ps = -1
        if info.state != MemState.OCCUPIED:
            self.to_raw(info)
