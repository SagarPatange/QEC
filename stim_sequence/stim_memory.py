
# ============================================================================
# stim_memory.py
# ----------------------------------------------------------------------------
# Lean, Stim-centric memory (keys) and link bookkeeping for SeQUeNCe-like nodes.
# Adds `StimMemoryArray.update_memory_params(...)` for Chapter-5 compatibility.
# ============================================================================

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from collections import defaultdict

# SeQUeNCe kernel primitives
from sequence.kernel.entity import Entity
from sequence.kernel.event import Event
from sequence.kernel.process import Process


# =============================================================================
# StimMemory: a local "key" (slot) with static physical parameters
# =============================================================================

class StimMemory(Entity):
    """
    A *slot handle* for a local qubit placement in a Stim circuit.
    Stores identity + static physical params. No quantum state/fidelity.
    """

    def __init__(
        self,
        name: str,
        timeline,
        array: "StimMemoryArray",
        index: int,
        *,
        frequency: float = 0.0,
        efficiency: float = 1.0,
        coherence_time: Optional[float] = None,
        wavelength: Optional[float] = None,
        decoherence_errors: Optional[Dict[str, float]] = None,
        cutoff_ratio: Optional[float] = None,
    ):
        super().__init__(name, timeline)
        self.array = array
        self.index = index

        # Static physical parameters
        self.frequency: float = float(frequency)
        self.efficiency: float = float(efficiency)
        self.coherence_time: Optional[float] = coherence_time
        self.wavelength: Optional[float] = wavelength
        self.decoherence_errors: Dict[str, float] = dict(decoherence_errors or {})
        self.cutoff_ratio: Optional[float] = cutoff_ratio

        # Stim binding
        self.key: Optional[Union[int, str]] = None

        # Owner node (wired by StimMemoryArray.init)
        self.owner = None
    
    def init(self) -> None:
        """Initialize the memory (Entity abstract method implementation)."""
        pass

    def set_owner(self, owner) -> None:
        self.owner = owner

    def set_key(self, key: Optional[Union[int, str]]) -> None:
        self.key = key

    def get_key(self) -> Optional[Union[int, str]]:
        return self.key

    def reset(self) -> None:
        self.key = None

    def expire(self) -> None:
        """Signal owner to free this key due to timeout/failure."""
        if self.array is not None:
            self.array.memory_expire(self)

    # Optional: derive px,py,pz from a dwell time using this key's model
    def pauli_channel_from_dwell(self, dwell_ps: float) -> Optional[Dict[str, float]]:
        if not self.decoherence_errors:
            return None

        def clamp01(x: float) -> float:
            return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

        errs = self.decoherence_errors
        if {"x_rate", "y_rate", "z_rate"} <= errs.keys():
            import math
            t = max(0.0, float(dwell_ps))
            return {
                "px": clamp01(1.0 - math.exp(-errs["x_rate"] * t)),
                "py": clamp01(1.0 - math.exp(-errs["y_rate"] * t)),
                "pz": clamp01(1.0 - math.exp(-errs["z_rate"] * t)),
            }
        if {"px", "py", "pz"} <= errs.keys():
            k = max(0.0, float(dwell_ps))
            return {
                "px": clamp01(1.0 - (1.0 - errs["px"]) ** k),
                "py": clamp01(1.0 - (1.0 - errs["py"]) ** k),
                "pz": clamp01(1.0 - (1.0 - errs["pz"]) ** k),
            }
        return None

    def __repr__(self) -> str:
        return (f"StimMemory(name={self.name!r}, idx={self.index}, "
                f"key={self.key!r}, freq={self.frequency}, eff={self.efficiency})")


# =============================================================================
# StimMemoryArray: indexable registry of StimMemory slots ("keys")
# =============================================================================

class StimMemoryArray(Entity):
    """
    Minimal registry of StimMemory slots.
    Exposes update_memory_params(field,value) for tutorial compatibility.
    """

    COMPONENT_TYPE = "MemoryArray"  # alias to match tutorial helpers

    def __init__(self, name: str, timeline, num_memories: int = 0, **defaults: Any):
        super().__init__(name, timeline)
        self.owner = None
        self.memories: List[StimMemory] = []
        self._name_to_idx: Dict[str, int] = {}

        if num_memories > 0:
            self.grow(num_memories, **defaults)
    
    def init(self, owner=None) -> None:
        """Initialize the memory array (Entity abstract method implementation)."""
        if owner is not None:
            self.owner = owner
            for m in self.memories:
                m.set_owner(owner)

    def __len__(self) -> int:
        return len(self.memories)

    def __getitem__(self, i: int) -> StimMemory:
        return self.memories[i]

    def _slot_name(self, idx: int) -> str:
        return f"{self.name}[{idx}]"

    def grow(self, n: int, **defaults: Any) -> List[StimMemory]:
        start = len(self.memories)
        out: List[StimMemory] = []
        for k in range(n):
            idx = start + k
            m = StimMemory(
                name=self._slot_name(idx),
                timeline=self.timeline,
                array=self,
                index=idx,
                **defaults,
            )
            self.memories.append(m)
            self._name_to_idx[m.name] = idx
            if self.owner is not None:
                m.set_owner(self.owner)
            out.append(m)
        return out

    def get_memory_by_name(self, name: str) -> StimMemory:
        return self.memories[self._name_to_idx[name]]

    def index_of(self, memory: StimMemory) -> int:
        if memory.array is not self:
            raise ValueError("Memory does not belong to this array")
        return memory.index

    def memory_expire(self, memory: StimMemory) -> None:
        if self.owner is not None and hasattr(self.owner, "memory_expire"):
            try:
                self.owner.memory_expire(memory)
            except Exception:
                pass

    # NEW: Chapter-5 compatible param update
    def update_memory_params(self, field: str, value: Any) -> None:
        """
        Update an attribute on all StimMemory slots.
        Ignores unknown fields (e.g., raw_fidelity from tutorials).
        """
        updated = 0
        for m in self.memories:
            if hasattr(m, field):
                setattr(m, field, value)
                updated += 1
        # Optional: you may log if nothing updated, but stay silent to keep it lean.


# =============================================================================
# Tier-2 Link model: pair-level record + tiny Pauli error budget
# =============================================================================

class LinkStatus(Enum):
    PENDING = auto()
    ENTANGLED = auto()
    CONSUMED = auto()
    FAILED = auto()

@dataclass
class LocalEndpoint:
    array_name: str
    index: int
    key: Optional[Union[int, str]]
    memory: StimMemory

@dataclass
class RemoteEndpoint:
    node: str
    key: Optional[Union[int, str]]

@dataclass
class PauliBudget:
    p_ix: float = 0.0
    p_xi: float = 0.0
    p_iz: float = 0.0
    p_zi: float = 0.0
    p_xx: float = 0.0
    p_zz: float = 0.0

    def clamp(self) -> None:
        for f in ("p_ix","p_xi","p_iz","p_zi","p_xx","p_zz"):
            v = getattr(self, f)
            v = 0.0 if v < 0.0 else 1.0 if v > 1.0 else v
            setattr(self, f, v)

    def add_local_channel(self, px: float = 0.0, pz: float = 0.0) -> None:
        self.p_ix = min(1.0, self.p_ix + px)
        self.p_iz = min(1.0, self.p_iz + pz)

    def add_remote_channel(self, px: float = 0.0, pz: float = 0.0) -> None:
        self.p_xi = min(1.0, self.p_xi + px)
        self.p_zi = min(1.0, self.p_zi + pz)

    def add_correlated(self, p_xx: float = 0.0, p_zz: float = 0.0) -> None:
        self.p_xx = min(1.0, self.p_xx + p_xx)
        self.p_zz = min(1.0, self.p_zz + p_zz)

    def as_dict(self) -> Dict[str, float]:
        self.clamp()
        return asdict(self)

@dataclass
class EntanglementLink:
    link_id: str
    local: LocalEndpoint
    remote: RemoteEndpoint
    status: LinkStatus
    born_ps: int
    deadline_ps: Optional[int] = None
    pauli_frame: Tuple[int, int] = (0, 0)
    budget: PauliBudget = field(default_factory=PauliBudget)

    def is_live(self) -> bool:
        return self.status == LinkStatus.ENTANGLED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "link_id": self.link_id,
            "status": self.status.name,
            "born_ps": self.born_ps,
            "deadline_ps": self.deadline_ps,
            "pauli_frame": tuple(self.pauli_frame),
            "local": {
                "array": self.local.array_name,
                "index": self.local.index,
                "key": self.local.key
            },
            "remote": {
                "node": self.remote.node,
                "key": self.remote.key
            },
            "budget": self.budget.as_dict(),
        }


# =============================================================================
# LinkLedger: manages EntanglementLinks and their TTL via kernel events
# =============================================================================

class LinkLedger(Entity):
    def __init__(self, name: str, timeline):
        super().__init__(name, timeline)
        self._links: Dict[str, EntanglementLink] = {}
        self._by_remote: Dict[str, List[str]] = defaultdict(list)
        self._by_local_key: Dict[Tuple[str, int], List[str]] = defaultdict(list)
        self._ttls: Dict[str, Event] = {}
        self._counter = 0
    
    def init(self) -> None:
        """Initialize the link ledger (Entity abstract method implementation)."""
        pass

    def _now(self) -> int:
        return int(self.timeline.now())

    def _mk_link_id(self) -> str:
        lid = f"{self.name}:L{self._counter}"
        self._counter += 1
        return lid

    def create(
        self,
        local_mem: StimMemory,
        remote_node: str,
        remote_key: Optional[Union[int, str]],
        *,
        pauli_frame: Tuple[int, int] = (0, 0),
        budget: Optional[PauliBudget] = None,
        ttl_ps: Optional[int] = None,
    ) -> EntanglementLink:
        lid = self._mk_link_id()
        local = LocalEndpoint(array_name=local_mem.array.name, index=local_mem.index,
                              key=local_mem.get_key(), memory=local_mem)
        remote = RemoteEndpoint(node=remote_node, key=remote_key)
        link = EntanglementLink(
            link_id=lid,
            local=local,
            remote=remote,
            status=LinkStatus.ENTANGLED,
            born_ps=self._now(),
            deadline_ps=(self._now() + int(ttl_ps)) if ttl_ps else None,
            pauli_frame=pauli_frame,
            budget=budget or PauliBudget()
        )
        self._links[lid] = link
        self._by_remote[remote_node].append(lid)
        self._by_local_key[(local.array_name, local.index)].append(lid)

        if ttl_ps:
            self._schedule_ttl(lid, int(ttl_ps))
        return link

    def get(self, link_id: str) -> EntanglementLink:
        return self._links[link_id]

    def links_by_remote(self, remote_node: str, live_only: bool = True) -> List[EntanglementLink]:
        out = []
        for lid in self._by_remote.get(remote_node, []):
            link = self._links.get(lid)
            if link and (link.is_live() or not live_only):
                out.append(link)
        return out

    def links_for_local(self, array_name: str, index: int, live_only: bool = True) -> List[EntanglementLink]:
        out = []
        for lid in self._by_local_key.get((array_name, index), []):
            link = self._links.get(lid)
            if link and (link.is_live() or not live_only):
                out.append(link)
        return out

    def mark_consumed(self, link_id: str) -> None:
        link = self._links.get(link_id)
        if not link:
            return
        if link.status in (LinkStatus.CONSUMED, LinkStatus.FAILED):
            return
        link.status = LinkStatus.CONSUMED
        self._cancel_ttl(link_id)

    def mark_failed(self, link_id: str) -> None:
        link = self._links.get(link_id)
        if not link:
            return
        if link.status == LinkStatus.FAILED:
            return
        link.status = LinkStatus.FAILED
        self._cancel_ttl(link_id)
        try:
            link.local.memory.expire()
        except Exception:
            pass

    def set_pauli_frame(self, link_id: str, frame: Tuple[int, int]) -> None:
        link = self._links.get(link_id)
        if link and link.is_live():
            link.pauli_frame = (int(frame[0]) & 1, int(frame[1]) & 1)

    def add_local_dwell(self, link_id: str, dwell_ps: float) -> None:
        link = self._links.get(link_id)
        if not (link and link.is_live()):
            return
        ch = link.local.memory.pauli_channel_from_dwell(dwell_ps)
        if ch:
            link.budget.add_local_channel(px=ch.get("px", 0.0), pz=ch.get("pz", 0.0))

    def add_remote_dwell(self, link_id: str, px: float = 0.0, pz: float = 0.0) -> None:
        link = self._links.get(link_id)
        if not (link and link.is_live()):
            return
        link.budget.add_remote_channel(px=px, pz=pz)

    def add_correlated_noise(self, link_id: str, p_xx: float = 0.0, p_zz: float = 0.0) -> None:
        link = self._links.get(link_id)
        if not (link and link.is_live()):
            return
        link.budget.add_correlated(p_xx=p_xx, p_zz=p_zz)

    def set_deadline(self, link_id: str, ttl_ps: Optional[int]) -> None:
        link = self._links.get(link_id)
        if not link:
            return
        self._cancel_ttl(link_id)
        if ttl_ps is None:
            link.deadline_ps = None
            return
        link.deadline_ps = self._now() + int(ttl_ps)
        self._schedule_ttl(link_id, int(ttl_ps))

    def _schedule_ttl(self, link_id: str, ttl_ps: int) -> None:
        when = self._now() + int(ttl_ps)
        proc = Process(self, "_on_ttl_expire", [link_id])
        evt = Event(when, proc)
        self.timeline.schedule(evt)
        self._ttls[link_id] = evt

    def _cancel_ttl(self, link_id: str) -> None:
        if link_id in self._ttls:
            self._ttls.pop(link_id, None)

    def _on_ttl_expire(self, link_id: str) -> None:
        link = self._links.get(link_id)
        if not link:
            return
        if link.status == LinkStatus.ENTANGLED:
            self.mark_failed(link_id)

    def to_dict(self, link_id: str) -> Dict[str, Any]:
        return self._links[link_id].to_dict()

    def list_links(self, live_only: bool = False) -> List[EntanglementLink]:
        if not live_only:
            return list(self._links.values())
        return [L for L in self._links.values() if L.is_live()]
