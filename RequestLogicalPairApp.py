"""RequestLogicalPairApp module.

This module defines a lightweight application controller for launching
logical Bell-pair generation across a linear path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import stim
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
        self.ancilla_qubits: dict[str, list] = {}  # neighbor -> list of 4 ancilla Memory objects (FT Prep)
        self._data_alloc_cursor: int = 0   # Next free index in DataMemoryArray.
        self._ancilla_alloc_cursor: int = 0  # Next free index in AncillaMemoryArray.

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

        self.frame_updates_by_src: dict[str, tuple[int, int]] = {}  # src -> (frame_bx, frame_bz)

    def start(self, responder: str, start_t: int, end_t: int, fidelity: float) -> None:
        """Start one link attempt as the initiator (lower path index).

        Args:
            responder: Neighbor endpoint name.
            start_t: Link start time in picoseconds.
            end_t: Link end time in picoseconds.
            fidelity: Requested target fidelity.

        Returns:
            None
        """
        self._allocate_data_and_ancilla(responder) # allocate data and ancilla qubits for this link before starting reservation to ensure they're ready when needed for TCNOT start.

        if responder not in self.qre_protocols:
            self.qre_protocols[responder] = QREProtocol(
                owner=self.node,
                app=self,
                remote_node_name=responder,
                memory_keys=None)
            
        if responder not in self.tcnot_protocols:
            self.tcnot_protocols[responder] = TeleportedCNOTProtocol(
                owner=self.node,
                name=f"TeleportedCNOT_{self.node.name}_to_{responder}",
                role="alice",
                remote_node_name=responder,
                data_qubits=self.data_qubits[responder],
                communication_qubits=self.comm_qubits[responder])  # Shared list; filled by get_physical_memory().

        self.node.reserve_net_resource(responder, start_t, end_t, self.n, fidelity) # reserve physical Bell pair generation with neighbor; callback will trigger TCNOT start once reservation is approved and memories are ready.

    def get_physical_memory(self, info: "MemoryInfo") -> None:
        """Collect entangled comm qubits; launch TeleportedCNOT when all n arrive.

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

        # Collect one comm qubit per callback; start TCNOT once exactly n are ready.
        memory = self.node.resource_manager.memory_manager[info.index]
        self.comm_qubits[neighbor].append(memory)

        if len(self.comm_qubits[neighbor]) == self.n:
            self._initial_link_fidelities[neighbor] = self.calculate_pair_fidelity(self.node.name, neighbor, "physical")
            self._initalize_teleported_cnot(neighbor, reservation)

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

        self._link_logical_fidelities[neighbor] = self.calculate_pair_fidelity(
            self.node.name,
            neighbor,
            "logical_link")
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

    def _allocate_data_and_ancilla(self, neighbor: str) -> None:
        """Allocate n data and 4 ancilla qubits for one link from local arrays.

        Args:
            neighbor: Remote node name this allocation is for.

        Returns:
            None
        """
        n_ancilla = 4  # TODO: derive from code/config when FT prep requirements change.

        data_array = self.node.components[f"{self.node.name}.DataMemoryArray"]
        d = self._data_alloc_cursor
        self.data_qubits[neighbor] = data_array.memories[d:d + self.n]
        self._data_alloc_cursor = d + self.n

        ancilla_array = self.node.components[f"{self.node.name}.AncillaMemoryArray"]
        a = self._ancilla_alloc_cursor
        self.ancilla_qubits[neighbor] = ancilla_array.memories[a:a + n_ancilla]
        self._ancilla_alloc_cursor = a + n_ancilla

        self.comm_qubits[neighbor] = []

    def _initalize_teleported_cnot(self, neighbor: str, reservation: "Reservation") -> None:
        """Create and start TeleportedCNOT once all comm qubits are collected.

        Args:
            neighbor: Remote node name.
            reservation: The reservation that produced these Bell pairs.

        Returns:
            None
        """
        is_initiator = reservation.initiator == self.node.name
        role = "alice" if is_initiator else "bob"

        tcnot = self.tcnot_protocols.get(neighbor)
        if tcnot is None:
            tcnot = TeleportedCNOTProtocol(
                owner=self.node,
                name=f"TeleportedCNOT_{self.node.name}_to_{neighbor}",
                role=role,
                remote_node_name=neighbor,
                data_qubits=self.data_qubits[neighbor],
                communication_qubits=self.comm_qubits[neighbor])
            self.tcnot_protocols[neighbor] = tcnot
        elif tcnot.role != role:
            raise RuntimeError(f"{self.name}: TCNOT role mismatch for {neighbor}: have {tcnot.role}, expected {role}")

        if tcnot.started:
            return

        tcnot.start()

    def _on_qre_complete(self, neighbor: str, result: dict[str, object] | None) -> None:
        """Handle QRE completion.

        Args:
            neighbor: Neighbor associated with the completed QRE protocol.
            result: Optional result payload from the protocol.

        Returns:
            None
        """
        self._final_end_to_end_fidelity = self.calculate_pair_fidelity(
            self._path_node_names[0],
            self._path_node_names[-1],
            "logical_end")

      
    def calculate_pair_fidelity(self, left_node: str, right_node: str, pair_type: str) -> float:
        """Compute non-LOCC fidelity for a physical, one-link logical, or end-to-end logical pair.

        Args:
            left_node: Left node name.
            right_node: Right node name.
            pair_type: One of "physical", "logical_link", or "logical_end".

        Returns:
            float: Pair fidelity.
        """
        left_app = self.node.timeline.get_entity_by_name(left_node).request_logical_pair_app
        right_app = self.node.timeline.get_entity_by_name(right_node).request_logical_pair_app

        if pair_type == "physical":
            left_keys = [left_app.comm_qubits[right_node][0].qstate_key]
            right_keys = [right_app.comm_qubits[left_node][0].qstate_key]
            px, py, pz = "X", "Y", "Z"
        elif pair_type == "logical_link":
            left_keys = [m.qstate_key for m in left_app.data_qubits[right_node]]
            right_keys = [m.qstate_key for m in right_app.data_qubits[left_node]]
            px = self.code.get_logical_x_string()
            pz = self.code.get_logical_z_string()
            py = "".join("Y" if x == "X" and z == "Z" else x if x == "X" else z for x, z in zip(px, pz))
        elif pair_type == "logical_end":
            left_keys = [m.qstate_key for m in left_app.data_qubits[self._path_node_names[1]]]
            right_keys = [m.qstate_key for m in right_app.data_qubits[self._path_node_names[-2]]]
            px = self.code.get_logical_x_string()
            pz = self.code.get_logical_z_string()
            py = "".join("Y" if x == "X" and z == "Z" else x if x == "X" else z for x, z in zip(px, pz))
        else:
            raise ValueError(f"{self.name}: unknown pair_type {pair_type}")

        sim = self.node.timeline.quantum_manager.states[left_keys[0]].tableau

        def corr(p: str) -> float:
            """Evaluate one correlator.

            Args:
                p: Pauli support string.

            Returns:
                float: Correlator value.
            """
            obs = stim.PauliString(sim.num_qubits)
            for key, pauli in zip(left_keys, p):
                if pauli != "I":
                    obs[key] = pauli
            for key, pauli in zip(right_keys, p):
                if pauli != "I":
                    obs[key] = pauli
            return float(sim.peek_observable_expectation(obs))

        return (1.0 + corr(px) - corr(py) + corr(pz)) / 4.0
