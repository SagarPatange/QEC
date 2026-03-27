"""RequestLogicalPairApp module.

This module defines a lightweight application controller for launching
logical Bell-pair generation across a linear path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import stim
from QREProtocol import QREMessage, QREProtocol
from css_codes import get_css_code
from sequence.components.circuit import Circuit
from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.utils import log
from TeleportedCNOT import TeleportedCNOTMessage, TeleportedCNOTProtocol


if TYPE_CHECKING:
    from sequence.kernel.quantum_state import TableauState
    from sequence.network_management.reservation import Reservation
    from sequence.resource_management.memory_manager import MemoryInfo
    from sequence.topology.node import QuantumRouter2ndGeneration


class RequestLogicalPairApp:
    """Application entry point for logical Bell-pair generation.

    Args:
        node: Owner router node.
        css_code: CSS code label for this app instance.
        required_end_to_end_logical_fidelity: Default target fidelity.
        path_node_names: Ordered node names for this app's linear path view.

    Notes:
        This app does not inherit RequestApp and does not use the reservation
        pipeline. It schedules protocol phases directly on the timeline.
    """

# ---------- Initialization and configuration ----------
    def __init__(self, node: "QuantumRouter2ndGeneration", css_code: str = "[[7,1,3]]", required_end_to_end_logical_fidelity: float = 0.8, path_node_names: list[str] | None = None):
        self.node = node  # Owner router for this app instance.
        self.node.set_app(self)  # Register app for reservation and memory callbacks.
        self.node.request_logical_pair_app = self  # Node-level handle for protocol lookup.

        self.name = f"{self.node.name}.RequestLogicalPairApp"  # Stable app identifier.
        self.css_code = css_code  # Requested CSS code label (e.g. [[7,1,3]]).
        self.code = get_css_code(css_code)  # Parsed code object used by protocol phases.
        self.n = self.code.n  # Physical qubits per logical block.
        self.required_end_to_end_logical_fidelity = float(required_end_to_end_logical_fidelity)  # Default chain target.
        self._path_node_names = list(path_node_names) if path_node_names else [self.node.name]  # Canonical path view for this app.

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

        # Per-link runtime state, keyed by neighbor node name.
        self.qre_protocols: dict[str, "QREProtocol"] = {}
        self.tcnot_protocols: dict[str, "TeleportedCNOTProtocol"] = {}
        self.link_ready: set[str] = set()  # Links that are ready to produce one adjacent logical Bell pair.
        self.tcnot_done: set[str] = set()  # Links whose TeleportedCNOT phase has completed.

        # Reservation tracking for physical-link callbacks.
        self.memo_to_reservation: dict[int, "Reservation"] = {}

        # Reservation callback aliases expected by node callback dispatch.
        self.get_reservation_result = self.get_physical_reservation_results
        self.get_other_reservation = self.get_other_physical_reservation
        self.get_memory = self.get_physical_memory

        # Backward-compatible report fields, pre-seeded until real values are computed.
        self._initial_link_fidelities: dict[str, float] = {}
        self._link_logical_fidelities: dict[str, float] = {}
        self._physical_pair_fidelity_rows: dict[str, list[dict[str, object]]] = {}
        self._post_idle_pair_fidelity_rows: dict[str, list[dict[str, object]]] = {}
        self._final_end_to_end_fidelity: Optional[float] = None
        self._attempt_start_time_ps: int | None = None
        self._final_completion_time_ps: int | None = None

        # FT preparation setting used by encode_data_qubits.
        self.ft_prep_mode = getattr(node, "ft_prep_mode", "none")

        # Idle-decoherence settings consumed by TCNOT/QRE phase operations.
        self.idle_pauli_weights: dict[str, float] = dict(getattr(node, "idle_pauli_weights", {"x": 0.05, "y": 0.05, "z": 0.90}))  # Biased idle Pauli weights.
        self.idle_data_coherence_time_sec = float(getattr(node, "idle_data_coherence_time_sec", 1e12))  # Data-qubit idle coherence.
        self.idle_comm_coherence_time_sec = float(getattr(node, "idle_comm_coherence_time_sec", 1e12))  # Comm-qubit idle coherence.
        
        # Baseline prep fidelity for logical encoding, derived from data memory quality or defaulting to 1.0 if not available.
        data_array_name = f"{self.node.name}.DataMemoryArray"
        data_array = self.node.components.get(data_array_name)  
        if data_array and len(data_array.memories) > 0:
            self.prep_fidelity = data_array.memories[0].raw_fidelity
        else:
            self.prep_fidelity = 1.0

        self.frame_updates_by_src: dict[str, tuple[int, int]] = {}  # src -> (frame_bx, frame_bz)

    def start(self, responder: str, start_t: int, end_t: int, fidelity: float) -> None:
        """Start one link reservation as the initiator.

        Args:
            responder: Neighbor endpoint name.
            start_t: Link start time in picoseconds.
            end_t: Link end time in picoseconds.
            fidelity: Requested target fidelity.

        Returns:
            None
        """
        log.logger.info(f"{self.name}: start responder={responder} window=({int(start_t)},{int(end_t)}) fidelity={fidelity:.4f}")
        if int(end_t) <= int(start_t):
            raise RuntimeError(f"{self.name}: end_t must be > start_t")

        # Track earliest request start for run-level latency reporting.
        if self._attempt_start_time_ps is None or int(start_t) < self._attempt_start_time_ps:
            self._attempt_start_time_ps = int(start_t)

        # Ensure local data/ancilla blocks exist for links this node must serve.
        if responder not in self.data_qubits:
            self._allocate_data_and_ancilla(responder)  # allocate local block for initiator-side link
        if self._path_role == "middle" and self._left_peer_name is not None:
            if self._left_peer_name not in self.data_qubits:
                self._allocate_data_and_ancilla(self._left_peer_name)  # pre-allocate other adjacent block for middle node

        # Create per-link QRE object once; QRE starts only after TCNOT barrier.
        if responder not in self.qre_protocols:
            self.qre_protocols[responder] = QREProtocol(
                owner=self.node,
                app=self,
                remote_node_name=responder)

        self.node.reserve_net_resource(responder, int(start_t), int(end_t), self.n, fidelity)

    def received_message(self, src: str, msg: object) -> bool:
        """Route incoming messages to active protocols.

        Args:
            src: Source node name.
            msg: Incoming message object.

        Returns:
            bool: True if any protocol was called.
        """
        msg_type = getattr(msg, "msg_type", type(msg).__name__)
        log.logger.debug(f"{self.name}: received_message src={src} type={msg_type}")

        # TCNOT routing is link-local: source node identifies the protocol.
        if isinstance(msg, TeleportedCNOTMessage):
            protocol = self.tcnot_protocols.get(src)
            if protocol is None:
                log.logger.warning(f"{self.name}: missing TCNOT protocol for src={src}")
                return False
            protocol.received_message(src, msg)
            return True

        # QRE frame updates are consumed by the currently running local QRE instance.
        if isinstance(msg, QREMessage):
            for protocol in self.qre_protocols.values():
                if protocol.is_running:
                    protocol.received_message(src, msg)
                    return True
            log.logger.warning(f"{self.name}: no running QRE protocol for src={src}")
            return False

        return False

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
        qm = self.node.timeline.quantum_manager

        def readonly_simulator_from_keys(keys: list[int]) -> tuple[stim.TableauSimulator, dict[int, int]]:
            """Build a temporary simulator view for keys without mutating manager state.

            Args:
                keys: Quantum-manager keys to inspect.

            Returns:
                tuple[stim.TableauSimulator, dict[int, int]]: Temporary simulator and key-to-local map.
            """
            unique_states: list["TableauState"] = []
            seen_state_ids: set[int] = set()
            for key in keys:
                qstate = qm.states[key]
                state_id = id(qstate)
                if state_id not in seen_state_ids:
                    seen_state_ids.add(state_id)
                    unique_states.append(qstate)

            if len(unique_states) == 1:
                state_obj = unique_states[0]
                simulator = qm._sim_from_tableau(
                    state_obj.current_tableau(),
                    num_keys=len(state_obj.keys),
                )
                key_to_local = {key: idx for idx, key in enumerate(state_obj.keys)}
                return simulator, key_to_local

            merged_keys: list[int] = []
            merged_tableau = None
            for qstate in unique_states:
                merged_keys.extend(qstate.keys)
                block_tableau = qstate.current_tableau()
                merged_tableau = (
                    block_tableau
                    if merged_tableau is None
                    else merged_tableau + block_tableau
                )

            if len(set(merged_keys)) != len(merged_keys):
                raise RuntimeError(
                    f"{self.name}: duplicate keys in read-only tableau merge: {merged_keys}"
                )

            simulator = qm._sim_from_tableau(merged_tableau, num_keys=len(merged_keys))
            key_to_local = {key: idx for idx, key in enumerate(merged_keys)}
            return simulator, key_to_local

        def fidelity_from_keys(left_keys: list[int], right_keys: list[int], px: str, py: str, pz: str) -> float:
            """Evaluate fidelity from key sets and Pauli support strings.

            Args:
                left_keys: Left-side keys.
                right_keys: Right-side keys.
                px: X-support string.
                py: Y-support string.
                pz: Z-support string.

            Returns:
                float: Fidelity value.
            """
            all_keys = left_keys + right_keys
            sim, key_to_local = readonly_simulator_from_keys(all_keys)

            def corr(pauli_support: str) -> float:
                """Evaluate one correlator.

                Args:
                    pauli_support: Pauli support string.

                Returns:
                    float: Correlator value.
                """
                obs = stim.PauliString(sim.num_qubits)
                for key, pauli in zip(left_keys, pauli_support):
                    if pauli != "I":
                        obs[key_to_local[key]] = pauli
                for key, pauli in zip(right_keys, pauli_support):
                    if pauli != "I":
                        obs[key_to_local[key]] = pauli
                return float(sim.peek_observable_expectation(obs))

            return (1.0 + corr(px) - corr(py) + corr(pz)) / 4.0

        if pair_type == "physical":
            # Physical pair fidelity is computed pair-by-pair, then averaged over n pairs.
            left_memories = left_app.comm_qubits[right_node]
            if len(left_memories) == 0:
                raise RuntimeError(f"{self.name}: no physical pairs for {left_node}<->{right_node}")

            right_router = self.node.timeline.get_entity_by_name(right_node)
            right_array = right_router.components[right_router.memo_arr_name]
            right_by_name = {memory.name: memory for memory in right_array.memories}
            pair_values: list[float] = []
            pair_rows: list[dict[str, object]] = []
            for pair_index, left_memory in enumerate(left_memories):
                remote_name = str(left_memory.entangled_memory["memo_id"])
                right_memory = right_by_name.get(remote_name)
                if right_memory is None:
                    raise RuntimeError(f"{self.name}: missing remote memory {remote_name}")
                pair_f = fidelity_from_keys([left_memory.qstate_key], [right_memory.qstate_key], "X", "Y", "Z")
                log.logger.info(
                    f"{self.name}: physical_pair left={left_memory.name}:{left_memory.qstate_key} "
                    f"right={right_memory.name}:{right_memory.qstate_key} f={pair_f:.6f}"
                )
                pair_values.append(pair_f)
                pair_rows.append(
                    {
                        "pair_index": pair_index,
                        "left_memory": left_memory.name,
                        "right_memory": right_memory.name,
                        "left_key": int(left_memory.qstate_key),
                        "right_key": int(right_memory.qstate_key),
                        "pair_fidelity": float(pair_f),
                    }
                )
            left_app._physical_pair_fidelity_rows[right_node] = pair_rows
            avg_f = float(sum(pair_values) / len(pair_values))
            log.logger.info(
                f"{self.name}: physical_avg left={left_node} right={right_node} "
                f"pairs={len(pair_values)} avg={avg_f:.6f}"
            )
            return avg_f

        # Logical link/end fidelities use code-level logical Pauli supports over data blocks.
        if pair_type == "logical_link":
            left_keys = [m.qstate_key for m in left_app.data_qubits[right_node]]
            right_keys = [m.qstate_key for m in right_app.data_qubits[left_node]]
        elif pair_type == "logical_end":
            left_keys = [m.qstate_key for m in left_app.data_qubits[self._path_node_names[1]]]
            right_keys = [m.qstate_key for m in right_app.data_qubits[self._path_node_names[-2]]]
        else:
            raise ValueError(f"{self.name}: unknown pair_type {pair_type}")

        px = self.code.get_logical_x_string()
        pz = self.code.get_logical_z_string()
        py = "".join("Y" if x == "X" and z == "Z" else x if x == "X" else z for x, z in zip(px, pz))
        return float(fidelity_from_keys(left_keys, right_keys, px, py, pz))

    def _allocate_data_and_ancilla(self, neighbor: str) -> None:
        """Allocate data and required ancilla qubits for one link from local arrays.

        Args:
            neighbor: Remote node name this allocation is for.

        Returns:
            None
        """
        required_ancillas = self.code.get_ft_required_ancillas(self.ft_prep_mode)

        # Allocation is monotonic by slice: next free block starts after already-assigned blocks.
        data_array = self.node.components[f"{self.node.name}.DataMemoryArray"]
        d = sum(len(block) for block in self.data_qubits.values())
        data_block = data_array.memories[d:d + self.n]
        if len(data_block) != self.n:
            raise RuntimeError(f"{self.name}: insufficient data memories for {neighbor} (need {self.n}, got {len(data_block)})")
        self.data_qubits[neighbor] = data_block

        ancilla_array = self.node.components[f"{self.node.name}.AncillaMemoryArray"]
        a = sum(len(block) for block in self.ancilla_qubits.values())
        ancilla_block = ancilla_array.memories[a:a + required_ancillas]
        if required_ancillas > 0 and len(ancilla_block) < required_ancillas:
            raise RuntimeError(f"{self.name}: insufficient ancilla memories for {neighbor} (need {required_ancillas}, got {len(ancilla_block)})")
        self.ancilla_qubits[neighbor] = ancilla_block

        self.comm_qubits[neighbor] = []

# ---------- Physical bell pair generation phase ----------

    def get_physical_memory(self, info: "MemoryInfo") -> None:
        """Collect entangled communication qubits and trigger data encoding.

        Args:
            info: Memory update callback payload.

        Returns:
            None
        """
        if info.state != "ENTANGLED":
            return
        if info.index not in self.memo_to_reservation:
            return

        # Route this callback to the corresponding adjacent link via reservation mapping.
        reservation = self.memo_to_reservation[info.index]
        if reservation.initiator == self.node.name:
            neighbor = reservation.responder
        else:
            neighbor = reservation.initiator

        # Cache local comm memory; once n are collected, schedule link-finalization barrier check.
        self.comm_qubits[neighbor].append(info.memory)
        # Comm key became active now; initialize idle-time baseline at current sim time.
        now_ps = int(self.node.timeline.now())
        self.node.timeline.quantum_manager.last_idle_time_ps_by_key[info.memory.qstate_key] = now_ps
        local_names = [m.name for m in self.comm_qubits[neighbor]]
        remote_names = [str(m.entangled_memory["memo_id"]) for m in self.comm_qubits[neighbor]]
        log.logger.info(
            f"{self.name}: physical_progress neighbor={neighbor} "
            f"count={len(local_names)}/{self.n} "
            f"unique_local={len(set(local_names))} unique_remote={len(set(remote_names))} "
            f"last={info.memory.name}->{info.memory.entangled_memory['memo_id']}"
        )
        log.logger.debug(f"{self.name}: entangled comm memory neighbor={neighbor} count={len(self.comm_qubits[neighbor])}/{self.n} index={info.index}")

        if len(self.comm_qubits[neighbor]) != self.n:
            return

        process = Process(self, "_finalize_physical_link_ready", [neighbor])
        event = Event(self.node.timeline.now(), process, self.node.timeline.schedule_counter)
        self.node.timeline.schedule(event)

    def _finalize_physical_link_ready(self, neighbor: str) -> None:
        """Finalize one physical link after both endpoints have n ready memories.

        Args:
            neighbor: Adjacent peer for this physical link.

        Returns:
            None
        """
        if neighbor in self.link_ready:
            return

        local_count = len(self.comm_qubits.get(neighbor, []))
        if local_count != self.n:
            return

        # Symmetric barrier: wait until both endpoints have n comm memories for this link.
        remote_router = self.node.timeline.get_entity_by_name(neighbor)
        remote_app = remote_router.request_logical_pair_app
        remote_count = len(remote_app.comm_qubits.get(self.node.name, []))
        if remote_count != self.n:
            process = Process(self, "_finalize_physical_link_ready", [neighbor])
            event = Event(self.node.timeline.now() + 10, process, self.node.timeline.schedule_counter)
            self.node.timeline.schedule(event)
            return

        self._initial_link_fidelities[neighbor] = self.calculate_pair_fidelity(self.node.name, neighbor, "physical")
        self.link_ready.add(neighbor)
        log.logger.info(f"{self.name}: link_ready neighbor={neighbor} ready_links={sorted(self.link_ready)}")

        expected_ready = {n for n in (self._left_peer_name, self._right_peer_name) if n is not None}
        if not self.link_ready.issuperset(expected_ready):
            return

        # Build node-local data blocks for encode stage once all required links are physically ready.
        if self._path_role == "edge":
            edge_neighbor = self._right_peer_name if self._right_peer_name is not None else self._left_peer_name
            data_blocks = [[m.qstate_key for m in self.data_qubits[edge_neighbor]]]
        else:
            data_blocks = [[m.qstate_key for m in self.data_qubits[self._left_peer_name]], [m.qstate_key for m in self.data_qubits[self._right_peer_name]]]

        # Encode only after both adjacent links are finalized as ready.
        log.logger.info(f"{self.name}: triggering encode path_role={self._path_role} expected_ready={sorted(expected_ready)}")
        self.encode_data_qubits(data_blocks=data_blocks, ft_prep_mode=self.ft_prep_mode, max_ft_prep_shots=8)

    def get_physical_reservation_results(self, reservation: "Reservation", result: bool) -> None:
        """Handle initiator-side physical reservation result and index mapping.

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

        # Sanity check: initiator path expects QRE object to exist from start().
        # Initiator path expects per-link QRE protocol pre-created in start().
        protocol = self.qre_protocols.get(neighbor)
        if protocol is None:
            raise RuntimeError(f"{self.name}: missing QREProtocol for {neighbor}; initialize in start()")

        # Reservation window bookkeeping:
        # map memory_index -> reservation at start, then remove at end.
        reservation_protocol = self.node.network_manager.protocol_stack[1]
        for card in reservation_protocol.timecards:
            if reservation in card.reservations:
                add_process = Process(self.memo_to_reservation, "__setitem__", [card.memory_index, reservation])
                add_event = Event(reservation.start_time, add_process)
                self.node.timeline.schedule(add_event)

                remove_process = Process(self.memo_to_reservation, "pop", [card.memory_index, None])
                remove_event = Event(reservation.end_time, remove_process)
                self.node.timeline.schedule(remove_event)

    def get_other_physical_reservation(self, reservation: "Reservation") -> None:
        """Handle responder-side approved reservation and index mapping.

        Args:
            reservation: Reservation object where this node is responder.

        Returns:
            None
        """
        if reservation.initiator == self.node.name:
            neighbor = reservation.responder
        else:
            neighbor = reservation.initiator

        # Responder lazily allocates local qubits and QRE object when counterpart request is approved.
        # Responder path allocates local qubits for this link if not present yet.
        if neighbor not in self.data_qubits:
            self._allocate_data_and_ancilla(neighbor)

        # Responder path also needs a local QRE protocol object for later phases.
        if neighbor not in self.qre_protocols:
            self.qre_protocols[neighbor] = QREProtocol(
                owner=self.node,
                app=self,
                remote_node_name=neighbor,
            )

        # Same reservation window bookkeeping as initiator path.
        reservation_protocol = self.node.network_manager.protocol_stack[1]
        for card in reservation_protocol.timecards:
            if reservation in card.reservations:
                add_process = Process(self.memo_to_reservation, "__setitem__", [card.memory_index, reservation])
                add_event = Event(reservation.start_time, add_process)
                self.node.timeline.schedule(add_event)

                remove_process = Process(self.memo_to_reservation, "pop", [card.memory_index, None])
                remove_event = Event(reservation.end_time, remove_process)
                self.node.timeline.schedule(remove_event)

# ---------- Encoding and fault-tolerant preparation phase ----------

    def encode_data_qubits(self, data_blocks: list[list[int]], ft_prep_mode: str = "none", max_ft_prep_shots: int = 8) -> None:
        """Encode one or more local logical data blocks.

        Args:
            data_blocks: List of local blocks; each block is a list of data-qubit keys.
            ft_prep_mode: Fault-tolerant preparation mode.
            max_ft_prep_shots: Maximum number of shots for fault-tolerant preparation.
        Returns:
            None.

        Notes:
            Edge-node usage: encode_data_qubits([edge_block_keys])
            Middle-node usage: encode_data_qubits([left_block_keys, right_block_keys])
        """
        log.logger.info(f"{self.name}: encode_data_qubits blocks={len(data_blocks)} ft_prep_mode={ft_prep_mode} max_shots={max_ft_prep_shots}")
        if self._path_role == "edge" and len(data_blocks) != 1:
            raise RuntimeError(f"{self.name}: edge node expects exactly 1 data block")
        if self._path_role == "middle" and len(data_blocks) != 2:
            raise RuntimeError(f"{self.name}: middle node expects exactly 2 data blocks")
        if max_ft_prep_shots < 1:
            raise RuntimeError(f"{self.name}: max_ft_prep_shots must be >= 1")

        qm = self.node.timeline.quantum_manager

        # Validate each local block, then encode/FT-prep block-by-block.
        for block_keys in data_blocks:
            if len(block_keys) != self.n:
                raise RuntimeError(f"{self.name}: expected {self.n} data keys, got {len(block_keys)}")

            for key in block_keys:
                if key not in qm.states:
                    raise RuntimeError(f"{self.name}: missing data key {key}")

            if ft_prep_mode not in {"none", "minimal", "standard"}:
                raise RuntimeError(f"{self.name}: unknown ft_prep_mode {ft_prep_mode}")

            # Resolve which neighbor owns this block so logical-state orientation is deterministic.
            block_neighbor = ""
            for neighbor_name, memories in self.data_qubits.items():
                if [m.qstate_key for m in memories] == block_keys:
                    block_neighbor = neighbor_name
                    break
            if block_neighbor == "":
                raise RuntimeError(f"{self.name}: could not map data block to neighbor")

            # Logical-state plan for Bell-generation pipeline:
            # block facing right peer is control-side (+), block facing left peer is target-side (0).
            logical_state = "+" if block_neighbor == self._right_peer_name else "0"

            # Select ancillas from local metadata only; no remote-state dependency (LOCC-safe).
            ancilla_keys: list[int] = []
            need = self.code.get_ft_required_ancillas(ft_prep_mode)
            if need > 0:
                if len(self.ancilla_qubits) == 0:
                    raise RuntimeError(f"{self.name}: no ancilla qubits available for FT prep")
                first_neighbor = next(iter(self.ancilla_qubits))
                ancilla_memories = self.ancilla_qubits[first_neighbor]
                if len(ancilla_memories) < need:
                    raise RuntimeError(f"{self.name}: {ft_prep_mode} FT mode requires >={need} ancillas")
                ancilla_keys = [ancilla_memories[i].qstate_key for i in range(need)]

            # Merge data and ancillas into one tableau block before local-index addressing.
            merge_keys = list(block_keys) + ancilla_keys
            merge_circuit = Circuit(len(merge_keys))
            qm.run_circuit(merge_circuit, merge_keys)
            log.logger.debug(f"{self.name}: merged block keys={len(block_keys)} ancillas={len(ancilla_keys)} total={len(merge_keys)}")

            state_obj = qm.states[block_keys[0]]
            key_to_local = {key: idx for idx, key in enumerate(state_obj.keys)}
            for key in merge_keys:
                if key not in key_to_local:
                    raise RuntimeError(f"{self.name}: key {key} not present in merged tableau state")

            q = [key_to_local[key] for key in block_keys]
            ancilla_locals = [key_to_local[key] for key in ancilla_keys]

            accepted = self.code.run_encode_ft_prep(
                simulator=state_obj.state,
                data_locals=q,
                ancilla_locals=ancilla_locals,
                ft_prep_mode=ft_prep_mode,
                max_ft_prep_shots=max_ft_prep_shots,
                logical_state=logical_state)
            log.logger.info(f"{self.name}: block encode accepted={accepted} ft_prep_mode={ft_prep_mode}")
            
            if not accepted:
                raise RuntimeError(f"{self.name}: FT prep failed after {max_ft_prep_shots} shots")

            # Encoding consumed this block now; reset idle-time baseline for involved keys.
            now_ps = int(self.node.timeline.now())
            for key in block_keys:
                qm.last_idle_time_ps_by_key[key] = now_ps
            for key in ancilla_keys:
                qm.last_idle_time_ps_by_key[key] = now_ps

        # TCNOT starts per ready link after local encode/FT-prep finishes.
        for ready_neighbor in list(self.link_ready):
            reservation_for_neighbor = None
            for res in self.memo_to_reservation.values():
                other = res.responder if res.initiator == self.node.name else res.initiator
                if other == ready_neighbor:
                    reservation_for_neighbor = res
                    break

            if reservation_for_neighbor is None:
                raise RuntimeError(f"{self.name}: missing reservation for ready link {ready_neighbor}")

            self.initialize_teleported_cnot(ready_neighbor, reservation_for_neighbor)

        log.logger.debug(f"{self.name}: encoded {len(data_blocks)} block(s) with ft_prep_mode={ft_prep_mode}")

# ---------- Teleported CNOT phase: reservation and memory callbacks for physical Bell pair generation ----------
    def initialize_teleported_cnot(self, neighbor: str, reservation: "Reservation") -> None:
        """Create and start TeleportedCNOT once all comm qubits are collected.

        Args:
            neighbor: Remote node name.
            reservation: The reservation that produced these Bell pairs.

        Returns:
            None
        """
        is_initiator = reservation.initiator == self.node.name
        role = "alice" if is_initiator else "bob"
        log.logger.info(f"{self.name}: initialize_tcnot neighbor={neighbor} role={role} comm={len(self.comm_qubits[neighbor])} data={len(self.data_qubits[neighbor])}")

        tcnot = self.tcnot_protocols.get(neighbor)
        if tcnot is None:
            tcnot = TeleportedCNOTProtocol(
                owner=self.node,
                name=f"TeleportedCNOT_{self.node.name}_to_{neighbor}",
                role=role,
                remote_node_name=neighbor,
                data_qubit_keys=[m.qstate_key for m in self.data_qubits[neighbor]],
                communication_qubit_keys=[m.qstate_key for m in self.comm_qubits[neighbor]])
            self.tcnot_protocols[neighbor] = tcnot
        elif tcnot.role != role:
            raise RuntimeError(f"{self.name}: TCNOT role mismatch for {neighbor}: have {tcnot.role}, expected {role}")

        if tcnot.started:
            return

        tcnot.start()

    def on_teleported_cnot_complete(self, reservation_or_neighbor: "Reservation | str") -> None:
        """Handle TCNOT completion bookkeeping for one adjacent link.

        Args:
            reservation_or_neighbor: Reservation object or neighbor node name.

        Returns:
            None
        """
        # Support both callback shapes: reservation object or direct neighbor string.
        if isinstance(reservation_or_neighbor, str):
            neighbor = reservation_or_neighbor
        else:
            reservation = reservation_or_neighbor
            if reservation.initiator == self.node.name:
                neighbor = reservation.responder
            else:
                neighbor = reservation.initiator
            self.get_other_physical_reservation(reservation)
        log.logger.info(f"{self.name}: tcnot_complete neighbor={neighbor}")

        left_keys_dbg = [m.qstate_key for m in self.data_qubits[neighbor]]
        right_app_dbg = self.node.timeline.get_entity_by_name(neighbor).request_logical_pair_app
        right_keys_dbg = [m.qstate_key for m in right_app_dbg.data_qubits[self.node.name]]

        qm_dbg = self.node.timeline.quantum_manager
        state_groups: dict[int, list[int]] = {}
        for k in left_keys_dbg + right_keys_dbg:
            sid = id(qm_dbg.states[k].state)
            state_groups.setdefault(sid, []).append(k)

        log.logger.info(
            f"{self.name}: fidelity_debug neighbor={neighbor} "
            f"left={left_keys_dbg} right={right_keys_dbg} "
            f"state_groups={state_groups}"
        )

        # Record one-link logical fidelity only on one designated side.
        neighbor_pos = self._path_node_names.index(neighbor)
        if self._path_position < neighbor_pos:
            self._link_logical_fidelities[neighbor] = self.calculate_pair_fidelity(
                self.node.name,
                neighbor,
                "logical_link",
            )

        # QRE barrier: this node launches QRE only after all its adjacent TCNOT links complete.
        self.tcnot_done.add(neighbor)
        required_tcnot = {n for n in (self._left_peer_name, self._right_peer_name) if n is not None}
        if not self.tcnot_done.issuperset(required_tcnot):
            log.logger.info(
                f"{self.name}: waiting_for_tcnot_barrier "
                f"done={self.tcnot_done} required={required_tcnot}")
            return

        for peer in required_tcnot:
            have = len(self.data_qubits.get(peer, []))
            if have != self.n:
                log.logger.info(f"{self.name}: waiting_for_data_block peer={peer} have={have} need={self.n}")
                return

        # Barrier passed: middle nodes launch one local swap; edge nodes launch their single link QRE.
        if self._path_role == "middle":
            swap_launch_neighbor = self._left_peer_name if self._left_peer_name is not None else self._right_peer_name
            if swap_launch_neighbor is None:
                raise RuntimeError(f"{self.name}: missing neighbor for middle-node QRE launch")
            self.start_qre_for_link(swap_launch_neighbor)
            return

        # Edge nodes launch their single link-level QRE.
        for qre_neighbor in required_tcnot:
            self.start_qre_for_link(qre_neighbor)

    def start_qre_for_link(self, neighbor: str) -> None:
        """Start QRE exactly once for one adjacent link.

        Args:
            neighbor: Adjacent node name for this link.

        Returns:
            None
        """
        log.logger.info(f"{self.name}: start_qre_for_link neighbor={neighbor}")
        protocol = self.qre_protocols.get(neighbor)
        if protocol is None:
            raise RuntimeError(f"{self.name}: missing QREProtocol for {neighbor}")
        if protocol.is_running:
            return

        # Event-driven QRE launch on the local node timeline.
        time_now = self.node.timeline.now()
        process = Process(protocol, "start", [])
        priority = self.node.timeline.schedule_counter
        event = Event(time_now, process, priority)
        self.node.timeline.schedule(event)

# ---------- QRE phase: protocol launch and completion handling ----------

    def logical_pair_complete(self, neighbor: str, result: dict[str, object] | None) -> None:
        """Store per-link QRE result and finalize when all links complete.

        Args:
            neighbor: Neighbor associated with completed QRE protocol.
            result: Optional result payload from QRE protocol.

        Returns:
            None
        """
        log.logger.info(f"{self.name}: logical_pair_complete neighbor={neighbor} result_keys={list(result.keys()) if result else []}")
        # Cache per-link QRE output if provided.
        if result is not None:
            self._post_idle_pair_fidelity_rows[neighbor] = [result]

        # Only protocols that actually started belong to the current attempt.
        # Middle nodes keep a second neighbor-scoped QRE object in IDLE, and
        # that object should not block attempt completion.
        active_protocols = [
            protocol
            for protocol in self.qre_protocols.values()
            if protocol.is_running or protocol.current_phase == "COMPLETE"
        ]
        if len(active_protocols) == 0:
            return
        if not all(
            protocol.current_phase == "COMPLETE"
            for protocol in active_protocols
        ):
            return

        # Initiator-only: compute end-to-end logical fidelity for this attempt.
        if self._path_position == 0:
            self._final_end_to_end_fidelity = self.calculate_pair_fidelity(self._path_node_names[0], self._path_node_names[-1], "logical_end")
            self._final_completion_time_ps = int(self.node.timeline.now())

        # Non-initiator nodes stop after completion bookkeeping.
        if self._path_position != 0:
            return

        # Compare against configured target and log outcome.
        meets_target = self._final_end_to_end_fidelity >= self.required_end_to_end_logical_fidelity
        if meets_target:
            log.logger.info(f"{self.name}: logical pair complete with end-to-end fidelity {self._final_end_to_end_fidelity:.4f} (target={self.required_end_to_end_logical_fidelity:.4f})")
        else:
            log.logger.warning(f"{self.name}: logical pair fidelity below target (final={self._final_end_to_end_fidelity:.4f}, target={self.required_end_to_end_logical_fidelity:.4f})")
