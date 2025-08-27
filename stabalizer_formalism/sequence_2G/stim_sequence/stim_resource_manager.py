
# =====================================================================
# stim_resource_manager.py
# ---------------------------------------------------------------------
# Coordinator that glues StimMemoryManager and StimRuleManager together.
# Ensures TTL is set on link creation if missing.
# =====================================================================

from __future__ import annotations
from typing import Union

from stim_memory_manager import StimMemoryManager, StimMemoryInfo, MemState
from stim_memory import LinkLedger, EntanglementLink, StimMemory


class StimResourceManager:
    def __init__(self, node, memory_manager: StimMemoryManager, rule_manager):
        self.node = node
        self.mm = memory_manager
        self.rm = rule_manager
        self.mm.resource_manager = self
        if hasattr(self.rm, "_mm") and self.rm._mm is not self.mm:
            self.rm._mm = self.mm

    def load(self, rule) -> None:
        self.rm.load(rule)

    def expire(self, rule) -> None:
        self.rm.expire(rule)

    def evaluate_rules(self) -> None:
        self.rm.evaluate(self)

    def update(self, protocol, memory: StimMemory, state: Union[str, MemState]) -> None:
        self.mm.update(memory, state)

    # --- Link events from controller/ledger ---

    def on_link_created(self, link_id: str) -> None:
        ledger = getattr(self.node, "link_ledger", self.mm.links)
        link = ledger.get(link_id)
        mem = link.local.memory

        # Ensure a deadline exists: derive from key if absent
        if link.deadline_ps is None:
            ct = getattr(mem, "coherence_time", None)
            cr = getattr(mem, "cutoff_ratio", None)
            if ct and cr:
                ttl = int(ct * cr)
                ledger.set_deadline(link_id, ttl)

        self.mm.bind_link(mem, link_id)
        
        # Notify the node's app about the new entanglement
        info = self.mm.get_info_by_memory(mem)
        self.node.get_memory(info)

    def on_link_consumed(self, link_id: str) -> None:
        link = self.mm.links.get(link_id)
        self.mm.clear_link(link.local.memory)

    def on_link_failed(self, link_id: str) -> None:
        link = self.mm.links.get(link_id)
        self.mm.clear_link(link.local.memory)

    # --- Node → RM (expiration callback path) ---

    def memory_expire(self, memory: StimMemory) -> None:
        self.mm.update(memory, MemState.RAW)
