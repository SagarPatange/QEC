"""RequestLogicalPairApp module.

This module defines a lightweight application controller for launching
logical Bell-pair generation across a linear path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from QREProtocol import QREProtocol
from css_codes import get_css_code
from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.utils import log
from TeleportedCNOT import TeleportedCNOTProtocol


if TYPE_CHECKING:
    from sequence.network_management.reservation import Reservation
    from sequence.resource_management.memory_manager import MemoryInfo
    from sequence.topology.node import QuantumRouter2ndGeneration


class RequestLogicalPairApp:
    """Application entry point for logical Bell-pair generation.

    Args:
        node: Owner router node.
        css_code: CSS code label for this app instance.
        required_end_to_end_logical_fidelity: Default target fidelity.

    Notes:
        This app does not inherit RequestApp and does not use the reservation
        pipeline. It schedules protocol phases directly on the timeline.
    """

    def __init__(self, node: "QuantumRouter2ndGeneration", css_code: str = "[[7,1,3]]", required_end_to_end_logical_fidelity: float = 0.8, run_config: Optional[dict[str, object]] = None):
        self.node = node  # Owner router for this app instance.
        self.node.set_app(self)  # Register app for reservation and memory callbacks.
        self.node.request_logical_pair_app = self  # Node-level handle for protocol lookup.

        self.name = f"{self.node.name}.RequestLogicalPairApp"  # Stable app identifier.
        self.css_code = css_code  # Requested CSS code label (e.g. [[7,1,3]]).
        self.code = get_css_code(css_code)  # Parsed code object used by protocol phases.
        self.n = self.code.n  # Physical qubits per logical block.
        self.required_end_to_end_logical_fidelity = float(required_end_to_end_logical_fidelity)  # Default chain target.
        self.run_config = dict(run_config) if run_config is not None else {}  # Parsed run configuration for this app.
        if self.run_config.get("path_node_names") is not None:
            self.run_config["path_node_names"] = list(self.run_config["path_node_names"])
        path_from_config = self.run_config.get("path_node_names")
        self._path_node_names = list(path_from_config) if isinstance(path_from_config, list) and len(path_from_config) > 0 else [self.node.name]  # Canonical path view for this app.

        # Precompute local path geometry so start() does minimal work.
        if self.node.name in self._path_node_names:
            self._path_position = self._path_node_names.index(self.node.name)  # Local index in chain path.
        else:
            self._path_position = 0  # Fallback when node is missing from provided path.
        last_index = len(self._path_node_names) - 1
        self._path_role = "edge" if self._path_position == 0 or self._path_position == last_index else "middle"  # Local role in chain.
        self._left_peer_name = self._path_node_names[self._path_position - 1] if self._path_position > 0 else None  # Immediate left neighbor.
        self._right_peer_name = (self._path_node_names[self._path_position + 1] if self._path_position < last_index else None)  # Immediate right neighbor.

        # Per-link qubit mapping, keyed by neighbor node name.
        self.data_qubits: dict[str, list] = {}  # neighbor -> list of n data Memory objects
        self.comm_qubits: dict[str, list] = {}  # neighbor -> list of n entangled comm Memory objects
        self.ancilla_qubits: dict[str, list] = {}  # neighbor -> list of n ancilla Memory objects

        # Per-link runtime state, keyed by neighbor node name.
        self.qre_protocols: dict[str, "QREProtocol"] = {}
        self.tcnot_protocols: dict[str, "TeleportedCNOTProtocol"] = {}
        self.link_ready: set[str] = set()  # Links that are ready to produce one adjacent logical Bell pair.
        self.tcnot_done: set[str] = set()  # Links whose teleported CNOT phase has completed.

        # Reservation tracking for physical-link callbacks.
        self.memo_to_reservation: dict[int, "Reservation"] = {}

        # Temporary compatibility alias while we migrate call sites.
        self.link_protocols = self.qre_protocols

        # Reservation callback aliases expected by node callback dispatch.
        self.get_reservation_result = self.get_physical_reservation_results
        self.get_other_reservation = self.get_physical_other_reservation
        self.get_memory = self.get_physical_memory

        # Backward-compatible report fields, pre-seeded until real values are computed.
        self._initial_link_fidelities: dict[str, float] = {}
        self._link_logical_fidelities: dict[str, float] = {}
        self._post_idle_pair_fidelity_rows: dict[str, list[dict[str, object]]] = {}
        self._final_end_to_end_fidelity: Optional[float] = None

        # Keep FT/prep fields available while QRE protocol phases are scaffolded.
        self.ft_prep_mode = getattr(node, "ft_prep_mode", "none")
        data_array_name = f"{self.node.name}.DataMemoryArray"
        data_array = self.node.components.get(data_array_name)  # May be absent in minimal test setups.
        if data_array and len(data_array.memories) > 0:
            self.prep_fidelity = data_array.memories[0].raw_fidelity
        else:
            self.prep_fidelity = 1.0

        # Compatibility flag used by existing setup helper.
        self._is_final_action_node = False

    def start(self, responder: str, start_t: int, end_t: int, fidelity: float) -> None:
        """Start one link attempt by enqueueing adjacent-link generation.

        Args:
            responder: Neighbor endpoint name.
            start_t: Link start time in picoseconds.
            end_t: Link end time in picoseconds.
            fidelity: Requested target fidelity.

        Returns:
            None

        Notes:
            This method only schedules the first protocol phase. It does not run
            the full protocol synchronously.
        """

        # Use init-derived path geometry; fall back only when no chain path was provided.
        if len(self._path_node_names) > 1:
            path = list(self._path_node_names)
            position = self._path_position
            role = self._path_role
            left_peer_name = self._left_peer_name
            right_peer_name = self._right_peer_name
        else:
            path = [self.node.name, responder]
            position = 0
            role = "edge"
            left_peer_name = None
            right_peer_name = responder

        metadata = {
            "responder": responder,
            "start_time": start_t,
            "end_time": end_t,
            "target_fidelity": fidelity,
            "css_code": self.css_code,
            "path_node_names": list(path),
            "position": position,
            "role": role,
            "left_peer_name": left_peer_name,
            "right_peer_name": right_peer_name,
        }

        # Allocate data qubits for this link
        data_array_name = f"{self.node.name}.DataMemoryArray"
        data_array = self.node.components[data_array_name]
        data_qubits = data_array.memories[0:self.n]

        # Pre-create TeleportedCNOT (comm qubits filled later by get_physical_memory)
        name = f"TeleportedCNOT_{self.node.name}_to_{responder}"
        tcnot = TeleportedCNOTProtocol(
            owner=self.node, name=name, role="alice",
            remote_node_name=responder,
            data_qubits=data_qubits,
            communication_qubits=[]) # Communication qubits will be assigned when physical Bell pair is ready.
        
        self.tcnot_protocols[responder] = tcnot
        self.node.reserve_net_resource(responder, start_t, end_t, self.n, fidelity)

    def get_physical_memory(self, info: "MemoryInfo") -> None:
        """Route physical Bell-pair memory events to the matching protocol.

        Args:
            info: Memory update callback payload.

        Returns:
            None
        """
        if info.state != "ENTANGLED":
            return
        if info.index not in self.memo_to_reservation:
            return

        reservation = self.memo_to_reservation[info.index]
        if reservation.initiator == self.node.name:
            neighbor = reservation.responder
        else:
            neighbor = reservation.initiator

        protocol = self.link_protocols.get(neighbor)
        if protocol is None:
            log.logger.warning(f"{self.name}: no protocol for neighbor={neighbor}")
            return
        if not protocol.is_running:
            protocol.start()
        protocol.on_entangled_memory(info, reservation)

    def received_message(self, src: str, msg: object) -> bool:
        """Route incoming messages to active protocols.

        Args:
            src: Source node name.
            msg: Incoming message object.

        Returns:
            ``True`` if handled, else ``False``.
        """
        # Delegate until one active link protocol accepts the message.
        for protocol in list(self.link_protocols.values()):
            if protocol.received_message(src, msg):
                return True
        return False

    def set_swap_config(self, config: dict[str, object]) -> None:
        """Compatibility hook for existing setup code.

        Args:
            config: Swap configuration payload.

        Returns:
            None
        """
        # Compatibility shim; full swap config handling will move into the new flow.
        self._swap_config = dict(config)

    def set_final_action_node(self) -> None:
        """Compatibility hook for existing setup code.

        Returns:
            None
        """
        self._is_final_action_node = True

    @staticmethod
    def build_swap_schedule(node_names: list[str]) -> tuple[dict[str, dict[str, object]], Optional[str]]:
        """Return empty swap schedule placeholder.

        Args:
            node_names: Ordered chain node names.

        Returns:
            Tuple of (configs, final_swap_node). Currently returns ({}, None).
        """
        # Placeholder to keep existing setup code callable.
        _ = node_names
        return {}, None

    def set_name(self, name: str) -> None:
        """Set app name.

        Args:
            name: New app name.

        Returns:
            None
        """
        self.name = name

    def __str__(self) -> str:
        """Return app name."""
        return self.name

    def _get_single_link_logical_pair(self, neighbor: str) -> None:
        """Start post-TCNOT QREProtocol for one adjacent link.

        Args:
            neighbor: Neighbor endpoint name for this link.

        Returns:
            None
        """
        protocol = self.qre_protocols.get(neighbor)
        if protocol is None:
            protocol = QREProtocol(
                owner=self.node,
                app=self,
                remote_node_name=neighbor,
                memory_keys=None,
                metadata={"phase_entry": "post_tcnot"},
            )
            self.qre_protocols[neighbor] = protocol

        if protocol.is_running:
            return

        time_now = self.node.timeline.now()
        process = Process(protocol, "start", [])
        priority = self.node.timeline.schedule_counter
        event = Event(time_now, process, priority)
        self.node.timeline.schedule(event)

    def _begin_adjacent_link_generation(self, responder: str) -> None:
        """Begin adjacent physical Bell-pair generation for one link.

        Args:
            responder: Neighbor endpoint name.

        Returns:
            None
        """
        # Guard against stale events or partially initialized links.
        protocol = self.link_protocols.get(responder)
        if protocol is None:
            log.logger.warning(f"{self.name}: no protocol for responder={responder}")
            return

        # Current protocol.start() is phase-1 entry only.
        protocol.start()

    def schedule_physical_bell_pair_reservation_window(self, reservation: "Reservation") -> None:
        """Schedule memory-index mappings for physical Bell-pair reservation.

        Args:
            reservation: Reservation object.

        Returns:
            None
        """
        reservation_protocol = self.node.network_manager.protocol_stack[1]
        for card in reservation_protocol.timecards:
            if reservation in card.reservations:
                add_process = Process(self.memo_to_reservation, "__setitem__", [card.memory_index, reservation])
                add_event = Event(reservation.start_time, add_process)
                self.node.timeline.schedule(add_event)

                remove_process = Process(self.memo_to_reservation, "pop", [card.memory_index, None])
                remove_event = Event(reservation.end_time, remove_process)
                self.node.timeline.schedule(remove_event)

    def get_physical_reservation_results(self, reservation: "Reservation", result: bool) -> None:
        """Handle initiator-side physical reservation result.

        Args:
            reservation: Reservation object.
            result: ``True`` if reservation is approved.

        Returns:
            None
        """
        if not result:
            return

        if reservation.initiator == self.node.name:
            neighbor = reservation.responder
        else:
            neighbor = reservation.initiator

        protocol = self.link_protocols.get(neighbor)
        if protocol is None:
            metadata = {
                "responder": neighbor,
                "start_time": reservation.start_time,
                "end_time": reservation.end_time,
                "target_fidelity": reservation.fidelity,
                "css_code": self.css_code,
                "path_node_names": list(self._path_node_names),
                "position": self._path_position,
                "role": self._path_role,
                "left_peer_name": self._left_peer_name,
                "right_peer_name": self._right_peer_name,
            }
            protocol = QREProtocol(owner=self.node, app=self, remote_node_name=neighbor, memory_keys=None, metadata=metadata)
            self.link_protocols[neighbor] = protocol

        self.schedule_physical_bell_pair_reservation_window(reservation)

    def _on_teleported_cnot_complete(self, reservation: "Reservation") -> None:
        """Handle responder-side approved physical reservation.

        Args:
            reservation: Reservation object created by remote initiator.

        Returns:
            None
        """
        if reservation.initiator == self.node.name:
            neighbor = reservation.responder
        else:
            neighbor = reservation.initiator

        protocol = self.link_protocols.get(neighbor)
        if protocol is None:
            metadata = {
                "responder": neighbor,
                "start_time": reservation.start_time,
                "end_time": reservation.end_time,
                "target_fidelity": reservation.fidelity,
                "css_code": self.css_code,
                "path_node_names": list(self._path_node_names),
                "position": self._path_position,
                "role": self._path_role,
                "left_peer_name": self._left_peer_name,
                "right_peer_name": self._right_peer_name,
            }
            protocol = QREProtocol(owner=self.node, app=self, remote_node_name=neighbor, memory_keys=None, metadata=metadata)
            self.link_protocols[neighbor] = protocol

        self.schedule_physical_bell_pair_reservation_window(reservation)
