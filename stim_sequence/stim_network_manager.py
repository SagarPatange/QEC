
# =====================================================================
# stim_network_manager.py
# ---------------------------------------------------------------------
# Minimal Chapter-5-compatible shim to drive reservations/windows and
# expose swapping tunables while delegating work to StimResourceManager.
# =====================================================================

from __future__ import annotations
from typing import List, Optional, Callable

from sequence.kernel.entity import Entity
from sequence.kernel.event import Event
from sequence.kernel.process import Process

# local managers
from stim_resource_manager import StimResourceManager


class SwappingProtocolShim:
    """Holds swapping tunables to satisfy Chapter-5 examples."""
    def __init__(self):
        self._success_rate: float = 1.0
        self._degradation: float = 0.0

    def set_swapping_success_rate(self, p: float) -> None:
        self._success_rate = float(p)

    def set_swapping_degradation(self, d: float) -> None:
        self._degradation = float(d)

    # Optional accessors for your controller:
    @property
    def success_rate(self) -> float: return self._success_rate
    @property
    def degradation(self) -> float: return self._degradation


class StimNetworkManager(Entity):
    """
    A lean network manager:
      - owns a protocol_stack where index 1 provides swapping tunables
      - schedules a reservation window [start_ps, end_ps]
      - on start: installs rules via a factory callback, then triggers RM.evaluate_rules()
      - on end: removes/unloads rules via a callback (or leaves to caller) and frees idle keys
    """
    def __init__(self, name: str, timeline, resource_manager: StimResourceManager):
        super().__init__(name, timeline)
        self.rm = resource_manager
        # protocol_stack[1] is consulted by tutorials for swapping knobs
        self.protocol_stack = [None, SwappingProtocolShim()]
        # Track installed rules per-window so we can unload them
        self._active_rules: List = []
    
    def init(self) -> None:
        """Initialize the network manager (Entity abstract method implementation)."""
        pass

    # Chapter-5 shape: request(responder, start_time, end_time, memory_size, target_fidelity)
    def request(
        self,
        responder: str,
        start_time_ps: int,
        end_time_ps: int,
        memory_size: int,
        target_fidelity: Optional[float] = None,
        rules_factory: Optional[Callable[[StimResourceManager, float], List[object]]] = None,
    ) -> None:
        """
        Schedule a reservation window and auto-install rules at start.
        - responder: node id or label (kept for API parity; routing is out of scope here)
        - memory_size: number of memories requested (unused here; selection is in rules)
        - target_fidelity: accepted and passed to rules_factory; can be ignored by rules
        - rules_factory: callback returning a list of rule instances to load
        """
        # install at start
        start_proc = Process(self, "_on_request_start",
                             [rules_factory, target_fidelity])
        self.timeline.schedule(Event(int(start_time_ps), start_proc))

        # cleanup at end
        end_proc = Process(self, "_on_request_end", [])
        self.timeline.schedule(Event(int(end_time_ps), end_proc))

        # Notify node's app about reservation result
        self.rm.node.get_reservation_result(None, True)

    # ---- scheduled callbacks ----

    def _on_request_start(self, rules_factory, target_fidelity):
        # Install rules if provided
        if callable(rules_factory):
            rules = list(rules_factory(self.rm, target_fidelity))
            for r in rules:
                self.rm.load(r)
                self._active_rules.append(r)
        
        # Evaluate rules once to start the process
        self.rm.evaluate_rules()

    def _on_request_end(self):
        # Unload rules we installed (caller can also manage explicitly)
        for r in self._active_rules:
            self.rm.expire(r)
        self._active_rules.clear()

        # Optionally: force evaluation to release idle OCCUPIED keys (if rules do so)
        self.rm.evaluate_rules()
