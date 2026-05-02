"""RequestLogicalPairApp module.

This module defines a lightweight application controller for launching
logical Bell-pair generation across a linear path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import stim
import time
from QREProtocol import QREMessage, QREProtocol
from css_codes import get_css_code
from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.utils import log
from TeleportedCNOT import TeleportedCNOTMessage, TeleportedCNOTProtocol
from sequence.constants import MILLISECOND

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
        debug: Whether to emit detailed debug logging.
        path_node_names: Ordered node names for this app's linear path view.

    Notes:
        This app does not inherit RequestApp and does not use the reservation
        pipeline. It schedules protocol phases directly on the timeline.
    """

# ---------- Initialization and configuration ----------
    def __init__(self,node: "QuantumRouter2ndGeneration", css_code: str = "[[7,1,3]]", 
                 required_end_to_end_logical_fidelity: float = 0.8, debug: bool = False, path_node_names: list[str] | None = None):
        """Initialize the logical-pair request application.

        Args:
            node: Owner router node.
            css_code: CSS code label for this app instance.
            required_end_to_end_logical_fidelity: Default target fidelity.
            debug: Whether to emit detailed debug logging.
            path_node_names: Ordered node names for this app's linear path view.

        Returns:
            None.
        """
        self.node = node
        self.node.set_app(self)
        self.node.request_logical_pair_app = self

        self.name = f"{self.node.name}.RequestLogicalPairApp"
        self.code = get_css_code(css_code)
        self.n = self.code.n
        self.required_end_to_end_logical_fidelity = float(required_end_to_end_logical_fidelity)
        self.debug = bool(debug)
        self._path_node_names = list(path_node_names) if path_node_names else [self.node.name]

        # Precompute local path geometry so start() does minimal work.
        if self.node.name in self._path_node_names:
            self._path_position = self._path_node_names.index(self.node.name)
        else:
            self._path_position = 0
        last_index = len(self._path_node_names) - 1
        self._path_role = "edge" if self._path_position == 0 or self._path_position == last_index else "middle"
        self._left_peer_name = self._path_node_names[self._path_position - 1] if self._path_position > 0 else None
        self._right_peer_name = self._path_node_names[self._path_position + 1] if self._path_position < last_index else None

        self.data_qubits: dict[str, list] = {}
        self.ancilla_qubits: dict[str, list] = {}

        # Reservation tracking for physical-link callbacks.
        self.memo_to_reservation: dict[int, "Reservation"] = {}

        # Reservation callback aliases expected by node callback dispatch.
        self.get_reservation_result = self.get_physical_reservation_results
        self.get_other_reservation = self.get_other_physical_reservation
        self.get_memory = self.get_physical_memory

        # Current-run state and metadata.
        self.current_run: dict[str, object] = {
            "run_id": 0,
            "status": "idle",
            "comm_qubits": {},
            "qre_protocols": {},
            "tcnot_protocols": {},
            "pending_qre_messages": {},
            "pending_tcnot_messages": {},
            "link_ready": set(),
            "tcnot_done": set(),
            "frame_updates_by_src": {},
            "link_logical_fidelities": {},
            "link_logical_fidelities_corrected": {},
            "last_corrected_recovery": None,
            "final_end_to_end_fidelity": None,
            "final_end_to_end_fidelity_corrected": None,
            "pre_frame_end_to_end_fidelity_corrected": None,
            "final_end_to_end_corrected_bell_state": None,
            "success": None,
            "start_time": None,
            "end_time": None,
            "completion_time": None,
        }

        self.run_stats: dict[int, dict[str, object]] = {}
        self.completed_run_count = 0
        self.sum_final_fidelity_corrected = 0.0
        self.sum_latency_ps = 0
        self.scheduled_run_starts: list[int] = []
        self.next_run_id = 1
        self._initiator_responder: str | None = None
        self._initiator_fidelity = 0.0
        self._initiator_run_duration_ps = 0
        self._initiator_round_spacing_ps = 0
        self._initiator_runs_remaining = 0
        self._initiator_next_start_t: int | None = None

        # FT preparation setting used by encode_data_qubits.
        self.ft_prep_mode = getattr(node, "ft_prep_mode", "none")

        # Idle-decoherence settings consumed by TCNOT/QRE phase operations.
        self.idle_pauli_weights: dict[str, float] = dict(getattr(node, "idle_pauli_weights", {"x": 0.05, "y": 0.05, "z": 0.90}))  # Biased idle Pauli weights.
        self.idle_t1_sec = float(getattr(node, "idle_t1_sec", 1e12))  # Shared idle T1.
        self.idle_t2_sec = float(getattr(node, "idle_t2_sec", 1e12))  # Shared idle T2.
        self.idle_decoherence_enabled = bool(getattr(node, "idle_decoherence_enabled", True))
        
        # Correction mode setting.
        self.correction_mode = str(getattr(node, "correction_mode"))
        if self.correction_mode not in {"none", "cec", "qec", "qec+cec"}:
            raise RuntimeError(f"{self.name}: unknown correction_mode {self.correction_mode}")
        
        # Baseline prep fidelity for logical encoding, derived from data memory quality or defaulting to 1.0 if not available.
        data_array_name = f"{self.node.name}.DataMemoryArray"
        data_array = self.node.components.get(data_array_name)  
        if data_array and len(data_array.memories) > 0:
            self.prep_fidelity = data_array.memories[0].raw_fidelity
        else:
            self.prep_fidelity = 1.0

        self.time_begin_of_run = 0

    def reset_run(self) -> None:
        """Reset all live state for the completed run.

        Args:
            None.

        Returns:
            None
        """
        self.time_begin_of_run = time.time()
        # log.logger.info(f"{self.name}: reset_run_execute run_id={self.current_run['run_id']} start={self.current_run['start_time']} end={self.current_run['end_time']} now={int(self.node.timeline.now())} data_neighbors={list(self.data_qubits.keys())} comm_neighbors={list(self.current_run['comm_qubits'].keys())}")
        resource_manager = self.node.resource_manager
        quantum_manager = self.node.timeline.quantum_manager
        reset_time_ps = int(self.node.timeline.now())

        # Return communication memories to the resource pool for the next run window.
        for memories in self.current_run["comm_qubits"].values():
            for memory in memories:
                quantum_manager.set_to_zero(memory.qstate_key)
                quantum_manager.last_idle_time_ps_by_key[memory.qstate_key] = reset_time_ps
                resource_manager.update(None, memory, "RAW")

        # Reset local data and ancilla state carried by this run.
        for memories in self.data_qubits.values():
            for memory in memories:
                quantum_manager.set_to_zero(memory.qstate_key)
                quantum_manager.last_idle_time_ps_by_key[memory.qstate_key] = reset_time_ps

        for memories in self.ancilla_qubits.values():
            for memory in memories:
                quantum_manager.set_to_zero(memory.qstate_key)
                quantum_manager.last_idle_time_ps_by_key[memory.qstate_key] = reset_time_ps

        self.data_qubits.clear()
        self.ancilla_qubits.clear()

        # Reinitialize per-run bookkeeping to the idle baseline.
        self.current_run = {
            "run_id": 0,
            "status": "idle",
            "comm_qubits": {},
            "qre_protocols": {},
            "tcnot_protocols": {},
            "pending_qre_messages": {},
            "pending_tcnot_messages": {},
            "link_ready": set(),
            "tcnot_done": set(),
            "frame_updates_by_src": {},
            "link_logical_fidelities": {},
            "link_logical_fidelities_corrected": {},
            "last_corrected_recovery": None,
            "final_end_to_end_fidelity": None,
            "final_end_to_end_fidelity_corrected": None,
            "pre_frame_end_to_end_fidelity_corrected": None,
            "final_end_to_end_corrected_bell_state": None,
            "success": None,
            "start_time": None,
            "end_time": None,
            "completion_time": None,
        }

    def begin_run(self, start_t: int, end_t: int) -> None:
        """Initialize live state for one scheduled run window.

        Args:
            start_t: Run start time in picoseconds.
            end_t: Run end time in picoseconds.

        Returns:
            None
        """
        self.time_begin_of_run = time.time()
        # log.logger.warning(f'{self.name} begin run, runtime = {time.time() - self.time_begin_of_run:.2f}s')

        # Prevents overlapping runs   
        if self.current_run["start_time"] == int(start_t) and self.current_run["status"] != "idle":
            return # NOTE: delete the "begin_run" event scheduled in get_other_physical_reservation()?

        self.node.timeline.quantum_manager.reset_error_statistics()

        # Initialize fresh per-run bookkeeping for the new scheduled execution window.
        self.current_run = {
            "run_id": self.next_run_id,
            "status": "active",
            "comm_qubits": {},
            "qre_protocols": {},
            "tcnot_protocols": {},
            "pending_qre_messages": {},
            "pending_tcnot_messages": {},
            "link_ready": set(),
            "tcnot_done": set(),
            "frame_updates_by_src": {},
            "link_logical_fidelities": {},
            "link_logical_fidelities_corrected": {},
            "last_corrected_recovery": None,
            "final_end_to_end_fidelity": None,
            "final_end_to_end_fidelity_corrected": None,
            "pre_frame_end_to_end_fidelity_corrected": None,
            "final_end_to_end_corrected_bell_state": None,
            "success": None,
            "start_time": int(start_t),
            "end_time": int(end_t),
            "completion_time": None,
        }
        self.next_run_id += 1

        # Allocate resources for active neighbors
        active_neighbors = [neighbor for neighbor in (self._left_peer_name, self._right_peer_name) if neighbor is not None]
        for neighbor in active_neighbors:
            if neighbor not in self.data_qubits:
                self.allocate_data_and_ancilla(neighbor)
            if neighbor not in self.current_run["qre_protocols"]:
                self.current_run["qre_protocols"][neighbor] = QREProtocol(owner=self.node, app=self, remote_node_name=neighbor)

    def record_pre_final_frame_fidelity(self) -> None:
        """Store the end-to-end fidelity right before final frame correction.

        Args:
            None.

        Returns:
            None.
        """
        return

    def _build_key_label_maps(self) -> tuple[dict[int, str], dict[int, str]]:
        """Build per-key block and qubit labels for critical summaries.

        Args:
            None.

        Returns:
            tuple[dict[int, str], dict[int, str]]: Block-label and qubit-label maps keyed by quantum-manager key.
        """
        block_labels: dict[int, str] = {}
        qubit_labels: dict[int, str] = {}
        for node_name in self._path_node_names:
            router = self.node.timeline.get_entity_by_name(node_name)
            app = router.request_logical_pair_app

            for peer_name, memories in app.data_qubits.items():
                for idx, memory in enumerate(memories):
                    block_label = f"{node_name}:data:{peer_name}"
                    block_labels[int(memory.qstate_key)] = block_label
                    qubit_labels[int(memory.qstate_key)] = f"{block_label}:q{idx}:key{int(memory.qstate_key)}"

            for peer_name, memories in app.ancilla_qubits.items():
                for idx, memory in enumerate(memories):
                    block_label = f"{node_name}:ancilla:{peer_name}"
                    block_labels[int(memory.qstate_key)] = block_label
                    qubit_labels[int(memory.qstate_key)] = f"{block_label}:q{idx}:key{int(memory.qstate_key)}"

            for peer_name, memories in app.current_run["comm_qubits"].items():
                for idx, memory in enumerate(memories):
                    block_label = f"{node_name}:comm:{peer_name}"
                    block_labels[int(memory.qstate_key)] = block_label
                    qubit_labels[int(memory.qstate_key)] = f"{block_label}:q{idx}:key{int(memory.qstate_key)}"

        return block_labels, qubit_labels

    def _log_critical_run_summary(self, run_id: int, latency_ps: int | None) -> None:
        """Emit the final corrected end-to-end fidelity summary.

        Args:
            run_id: Logical-pair run identifier.
            latency_ps: End-to-end latency in picoseconds, if known.

        Returns:
            None.
        """
        final_corrected = self.current_run["final_end_to_end_fidelity_corrected"]

        if not self.debug:
            log.logger.critical(f"{final_corrected}, {latency_ps}")
            return

        qm_stats = self.node.timeline.quantum_manager.get_error_statistics()
        log.logger.critical(
            f"critical_e2e run_id={run_id} latency_ps={latency_ps} fidelity_corrected={final_corrected} "
            f"qm_gate_count=(1q:{qm_stats['gate_1q_count']},2q:{qm_stats['gate_2q_count']}) "
            f"qm_gate_error_count=(1q:{qm_stats['gate_1q_error_count']},2q:{qm_stats['gate_2q_error_count']}) "
            f"qm_measurement=(count:{qm_stats['measurement_count']},error_count:{qm_stats['measurement_error_count']})"
        )
    

    def start(self, responder: str, start_t: int, end_t: int, fidelity: float, num_logical_pairs: int) -> None:
        """Schedule one or more logical-pair runs.

        Args:
            responder: Neighbor endpoint name.
            start_t: First run start time in picoseconds.
            end_t: First run end time in picoseconds.
            fidelity: Requested target fidelity.
            num_logical_pairs: Number of logical-pair runs to schedule.

        Returns:
            None
        """
        round_spacing_ps = int(getattr(self.node, "round_spacing_ps", int(1e9)))  # Gap between successive logical-pair run windows.
        
        if int(end_t) <= int(start_t):
            raise RuntimeError(f"{self.name}: end_t must be > start_t")
        if num_logical_pairs < 1:
            raise RuntimeError(f"{self.name}: num_logical_pairs must be >= 1")

        run_duration_ps = int(end_t) - int(start_t)
        self._initiator_responder = responder
        self._initiator_fidelity = float(fidelity)
        self._initiator_run_duration_ps = run_duration_ps
        self._initiator_round_spacing_ps = round_spacing_ps
        self._initiator_runs_remaining = int(num_logical_pairs)
        self._initiator_next_start_t = int(start_t)
        self._schedule_next_initiator_run()

    def _schedule_next_initiator_run(self) -> bool:
        """Schedule the next initiator-side run window if one remains.

        Args:
            None.

        Returns:
            bool: ``True`` if a new run window was scheduled.
        """
        if self._initiator_responder is None or self._initiator_next_start_t is None:
            return False
        if self._initiator_runs_remaining < 1:
            return False

        run_start_t = int(self._initiator_next_start_t)
        run_end_t = run_start_t + int(self._initiator_run_duration_ps)
        self.scheduled_run_starts.append(run_start_t)

        process = Process(self, "begin_run", [run_start_t, run_end_t])
        event = Event(run_start_t, process, self.node.timeline.schedule_counter)
        self.node.timeline.schedule(event)
        self.node.reserve_net_resource(
            self._initiator_responder,
            run_start_t,
            run_end_t,
            self.n,
            float(self._initiator_fidelity),
        )

        self._initiator_runs_remaining -= 1
        self._initiator_next_start_t = run_start_t + int(self._initiator_run_duration_ps) + int(self._initiator_round_spacing_ps)
        return True

    def received_message(self, src: str, msg: object) -> bool:
        """Route incoming messages to active protocols.

        Args:
            src: Source node name.
            msg: Incoming message object.

        Returns:
            bool: True if any protocol was called.
        """
        msg_type = getattr(msg, "msg_type", type(msg).__name__)
        # log.logger.debug(f"{self.name}: received_message src={src} type={msg_type}")

        # TCNOT routing is link-local: source node identifies the protocol.
        if isinstance(msg, TeleportedCNOTMessage):
            protocol = self.current_run["tcnot_protocols"].get(src)
            if protocol is None:
                pending_tcnot_messages = self.current_run["pending_tcnot_messages"].setdefault(src, [])
                pending_tcnot_messages.append(msg)  # Preserve early TCNOT payload messages until the local protocol exists.
                # log.logger.info(f"{self.name}: buffered TCNOT message src={src} type={msg.msg_type} pending={len(pending_tcnot_messages)}")
                return True
            protocol.received_message(src, msg)
            return True

        # QRE frame updates are consumed by the currently running local QRE instance.
        if isinstance(msg, QREMessage):
            for protocol in self.current_run["qre_protocols"].values():
                if protocol.is_running:
                    protocol.received_message(src, msg)
                    return True
            pending_qre_messages = self.current_run["pending_qre_messages"].setdefault(src, [])
            pending_qre_messages.append(msg)  # Preserve early frame updates until the local QRE protocol is running.
            # log.logger.info(f"{self.name}: buffered QRE message src={src} type={msg.msg_type} pending={len(pending_qre_messages)}")
            return True

        return False

    def _calculate_pair_fidelity(self, left_node: str, right_node: str, pair_type: str, recover_endpoints: bool) -> float:
        """Compute pair fidelity with optional noiseless endpoint recovery on a copied tableau.

        Args:
            left_node: Left node name.
            right_node: Right node name.
            pair_type: One of "physical", "logical_link", or "logical_end".
            recover_endpoints: Whether to apply noiseless endpoint recovery before evaluating logical correlators.

        Returns:
            float: Pair fidelity.
        """
        qm = self.node.timeline.quantum_manager

        # Resolve the endpoint apps and the shared tableau manager for the pair being scored.
        left_app = self.node.timeline.get_entity_by_name(left_node).request_logical_pair_app
        right_app = self.node.timeline.get_entity_by_name(right_node).request_logical_pair_app

        # Build a read-only tableau copy for the requested qubits.
        # This keeps fidelity evaluation from mutating the live simulation state.
        def build_simulator_from_keys(keys: list[int]) -> tuple[stim.TableauSimulator, dict[int, int]]:
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
                # Fast path: all requested qubits already share one tableau block.
                state_obj = unique_states[0]
                simulator = stim.TableauSimulator()
                simulator.set_inverse_tableau(state_obj.current_inverse_tableau())
                key_to_local = {key: idx for idx, key in enumerate(state_obj.keys)}
                return simulator, key_to_local

            merged_keys: list[int] = []
            merged_tableau = None
            for qstate in unique_states:
                # When the pair spans multiple independent blocks, merge them into one temporary tableau.
                merged_keys.extend(qstate.keys)
                block_tableau = qstate.current_tableau()
                merged_tableau = block_tableau if merged_tableau is None else merged_tableau + block_tableau

            if len(set(merged_keys)) != len(merged_keys):
                raise RuntimeError(f"{self.name}: duplicate keys in read-only tableau merge: {merged_keys}")

            simulator = stim.TableauSimulator()
            simulator.set_inverse_tableau(merged_tableau.inverse())
            key_to_local = {key: idx for idx, key in enumerate(merged_keys)}
            return simulator, key_to_local

        # Build one logical Pauli observable that acts on both encoded endpoint blocks.
        def build_joint_observable(left_keys: list[int], right_keys: list[int], pauli_string: str, key_to_local: dict[int, int]) -> stim.PauliString:
            """Build one logical observable spanning both endpoint blocks.

            Args:
                left_keys: Left block keys in code order.
                right_keys: Right block keys in code order.
                pauli_string: Pauli support string to apply on each block.
                key_to_local: Map from quantum-manager keys to simulator-local indices.

            Returns:
                stim.PauliString: Joint logical observable on the temporary simulator.
            """
            observable = stim.PauliString(len(key_to_local))
            for keys in (left_keys, right_keys):
                for key, pauli in zip(keys, pauli_string):
                    if pauli != "I":
                        observable[key_to_local[key]] = pauli
            return observable

        # Run one ideal recovery pass on a copied endpoint block.
        # This is only used for the corrected end-to-end metric.
        def recover_block(sim: stim.TableauSimulator, block_keys: list[int], key_to_local: dict[int, int], block_label: str) -> dict[str, object]:
            """Apply one noiseless recovery round on a copied endpoint block.

            Args:
                sim: Temporary simulator copy used only for fidelity evaluation.
                block_keys: Quantum-manager keys for one encoded endpoint block.
                key_to_local: Map from quantum-manager keys to simulator-local indices.
                block_label: Human-readable block label for logging.

            Returns:
                dict[str, object]: Measured before/after syndromes and decoded correction indices.
            """
            decode_table = self.code.get_decode_table()
            if len(decode_table) == 0:
                return {
                    "x_syndrome_before": [],
                    "z_syndrome_before": [],
                    "x_syndrome_after": [],
                    "z_syndrome_after": [],
                    "x_error_qubit": None,
                    "z_error_qubit": None,
                }

            def syndrome_bit(stabilizer: str) -> int | None:
                """Compute one syndrome bit from a stabilizer expectation.

                Args:
                    stabilizer: Pauli stabilizer string over the block.

                Returns:
                    int | None: Syndrome bit, where +1 maps to 0 and -1 maps to 1.
                        Returns ``None`` when the stabilizer is non-deterministic.
                """
                # Peek the stabilizer on the copied tableau and convert the sign into a syndrome bit.
                observable = stim.PauliString(len(key_to_local))
                for key, pauli in zip(block_keys, stabilizer):
                    if pauli != "I":
                        observable[key_to_local[key]] = pauli
                expectation = float(sim.peek_observable_expectation(observable))
                if expectation > 0.5:
                    return 0
                if expectation < -0.5:
                    return 1
                if abs(expectation) <= 0.5:
                    # log.logger.warning(
                        # f"{self.name}: nondeterministic_stabilizer block_keys={block_keys} "
                        # f"stabilizer={stabilizer} expectation={expectation:.6f}"
                    # )
                    return None  # Non-deterministic stabilizer; treat as decode failure and skip correction.
                raise RuntimeError(f"{self.name}: unexpected stabilizer expectation {expectation}")

            def decode_syndrome(syndrome: list[int | None]) -> int | None:
                """Decode one syndrome when all bits are deterministic.

                Args:
                    syndrome: Syndrome bits for one error type.

                Returns:
                    int | None: Qubit index to correct, or ``None`` when the
                        syndrome is zero or non-deterministic.
                """
                if any(bit is None for bit in syndrome):
                    return None
                return decode_table[tuple(int(bit) for bit in syndrome)]

            # In a CSS code, Z stabilizers locate X errors and X stabilizers locate Z errors.
            x_syndrome_before = [syndrome_bit(stabilizer) for stabilizer in self.code.get_x_stabilizer_strings()]
            z_syndrome_before = [syndrome_bit(stabilizer) for stabilizer in self.code.get_z_stabilizer_strings()]
            x_error_qubit = decode_syndrome(z_syndrome_before)
            z_error_qubit = decode_syndrome(x_syndrome_before)
            if x_error_qubit is not None:
                sim.x(key_to_local[block_keys[int(x_error_qubit)]])
            if z_error_qubit is not None:
                sim.z(key_to_local[block_keys[int(z_error_qubit)]])

            x_syndrome_after = [syndrome_bit(stabilizer) for stabilizer in self.code.get_x_stabilizer_strings()]
            z_syndrome_after = [syndrome_bit(stabilizer) for stabilizer in self.code.get_z_stabilizer_strings()]
            # log.logger.warning(
                # f"{self.name}: ideal_recovery block={block_label} block_keys={block_keys} "
                # f"x_before={x_syndrome_before} z_before={z_syndrome_before} "
                # f"x_q={x_error_qubit} z_q={z_error_qubit} "
                # f"x_after={x_syndrome_after} z_after={z_syndrome_after}"
            # )

            return {
                "x_syndrome_before": x_syndrome_before,
                "z_syndrome_before": z_syndrome_before,
                "x_syndrome_after": x_syndrome_after,
                "z_syndrome_after": z_syndrome_after,
                "x_error_qubit": x_error_qubit,
                "z_error_qubit": z_error_qubit,
            }

        # Evaluate Bell-pair fidelity once the relevant endpoint keys and logical supports are known.
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
            def build_single_block_observable(block_keys: list[int], pauli_string: str, key_to_local: dict[int, int]) -> stim.PauliString:
                """Build one logical observable on a single encoded block.

                Args:
                    block_keys: Block keys in code order.
                    pauli_string: Pauli support string to apply on the block.
                    key_to_local: Map from quantum-manager keys to simulator-local indices.

                Returns:
                    stim.PauliString: Logical observable on one block.
                """
                observable = stim.PauliString(len(key_to_local))
                for key, pauli in zip(block_keys, pauli_string):
                    if pauli != "I":
                        observable[key_to_local[key]] = pauli
                return observable

            def all_logical_pauli_expectations(sim: stim.TableauSimulator, left_keys: list[int], right_keys: list[int], px: str, py: str, pz: str, key_to_local: dict[int, int]) -> dict[str, float]:
                """Return all 16 logical two-qubit Pauli expectations.

                Args:
                    sim: Temporary simulator copy used for fidelity evaluation.
                    left_keys: Left-side keys.
                    right_keys: Right-side keys.
                    px: X-support string.
                    py: Y-support string.
                    pz: Z-support string.
                    key_to_local: Map from quantum-manager keys to simulator-local indices.

                Returns:
                    dict[str, float]: Expectations for II, IX, ..., ZZ.
                """
                support_by_label = {
                    "I": "I" * len(px),
                    "X": px,
                    "Y": py,
                    "Z": pz,
                }

                values: dict[str, float] = {}
                labels = ["I", "X", "Y", "Z"]
                for left_label in labels:
                    for right_label in labels:
                        name = f"{left_label}{right_label}"
                        if name == "II":
                            values[name] = 1.0
                            continue

                        observable = stim.PauliString(len(key_to_local))
                        left_obs = build_single_block_observable(left_keys, support_by_label[left_label], key_to_local)
                        right_obs = build_single_block_observable(right_keys, support_by_label[right_label], key_to_local)
                        observable *= left_obs
                        observable *= right_obs
                        values[name] = float(sim.peek_observable_expectation(observable))
                return values

            def bell_fidelities_from_correlators(cx: float, cy: float, cz: float) -> dict[str, float]:
                """Return fidelities for the four logical Bell states.

                Args:
                    cx: Logical XX correlator expectation.
                    cy: Logical YY correlator expectation.
                    cz: Logical ZZ correlator expectation.

                Returns:
                    dict[str, float]: Bell-state fidelities keyed by label.
                """
                return {
                    "phi_plus": (1.0 + cx - cy + cz) / 4.0,
                    "phi_minus": (1.0 - cx + cy + cz) / 4.0,
                    "psi_plus": (1.0 + cx + cy - cz) / 4.0,
                    "psi_minus": (1.0 - cx - cy - cz) / 4.0,
                }

            def log_correlator_snapshot(stage: str, sim: stim.TableauSimulator, left_keys: list[int], right_keys: list[int], px: str, py: str, pz: str, key_to_local: dict[int, int]) -> None:
                """Log one Bell-correlator snapshot for the copied endpoint state.

                Args:
                    stage: Short stage label for the snapshot.
                    sim: Temporary simulator copy used for fidelity evaluation.
                    left_keys: Left-side keys.
                    right_keys: Right-side keys.
                    px: X-support string.
                    py: Y-support string.
                    pz: Z-support string.
                    key_to_local: Map from quantum-manager keys to simulator-local indices.

                Returns:
                    None
                """
                xx_obs = build_joint_observable(left_keys, right_keys, px, key_to_local)
                cx = float(sim.peek_observable_expectation(xx_obs))

                yy_obs = build_joint_observable(left_keys, right_keys, py, key_to_local)
                cy = float(sim.peek_observable_expectation(yy_obs))

                zz_obs = build_joint_observable(left_keys, right_keys, pz, key_to_local)
                cz = float(sim.peek_observable_expectation(zz_obs))

                pauli_values = all_logical_pauli_expectations(sim, left_keys, right_keys, px, py, pz, key_to_local)
                fidelities = bell_fidelities_from_correlators(cx, cy, cz)
                best_label = max(fidelities, key=fidelities.get)
                value = fidelities[best_label]
                # log.logger.warning(
                    # f"{self.name}: recovery_correlators stage={stage} pair_type={pair_type} "
                    # f"left={left_node} right={right_node} "
                    # f"cx={cx:.6f} cy={cy:.6f} cz={cz:.6f} "
                    # f"paulis={pauli_values} "
                    # f"phi_plus={fidelities['phi_plus']:.6f} phi_minus={fidelities['phi_minus']:.6f} "
                    # f"psi_plus={fidelities['psi_plus']:.6f} psi_minus={fidelities['psi_minus']:.6f} "
                    # f"best={best_label} value={value:.6f}"
                # )

            all_keys = left_keys + right_keys
            sim, key_to_local = build_simulator_from_keys(all_keys)
            left_recovery = {
                "x_syndrome_before": [],
                "z_syndrome_before": [],
                "x_syndrome_after": [],
                "z_syndrome_after": [],
                "x_error_qubit": None,
                "z_error_qubit": None,
            }
            right_recovery = {
                "x_syndrome_before": [],
                "z_syndrome_before": [],
                "x_syndrome_after": [],
                "z_syndrome_after": [],
                "x_error_qubit": None,
                "z_error_qubit": None,
            }
            if pair_type in {"logical_end", "logical_link"} and recover_endpoints:
                log_correlator_snapshot("before_recovery", sim, left_keys, right_keys, px, py, pz, key_to_local)

                # Corrected fidelity: recover each endpoint block on the copied tableau before scoring it.
                left_recovery = recover_block(sim, left_keys, key_to_local, "left")
                log_correlator_snapshot("after_left_recovery", sim, left_keys, right_keys, px, py, pz, key_to_local)
                right_recovery = recover_block(sim, right_keys, key_to_local, "right")
                log_correlator_snapshot("after_right_recovery", sim, left_keys, right_keys, px, py, pz, key_to_local)
                if pair_type == "logical_end":
                    self.current_run["last_corrected_recovery"] = {
                        "left_recovery": dict(left_recovery),
                        "right_recovery": dict(right_recovery),
                    }

            # Score the final copied state with logical XX, YY, and ZZ Bell correlators.
            xx_obs = build_joint_observable(left_keys, right_keys, px, key_to_local)
            cx = float(sim.peek_observable_expectation(xx_obs))

            yy_obs = build_joint_observable(left_keys, right_keys, py, key_to_local)
            cy = float(sim.peek_observable_expectation(yy_obs))

            zz_obs = build_joint_observable(left_keys, right_keys, pz, key_to_local)
            cz = float(sim.peek_observable_expectation(zz_obs))

            pauli_values = all_logical_pauli_expectations(sim, left_keys, right_keys, px, py, pz, key_to_local)
            fidelities = bell_fidelities_from_correlators(cx, cy, cz)
            if pair_type == "logical_end" and recover_endpoints:
                best_label = max(fidelities, key=fidelities.get)
                value = fidelities["phi_plus"]
                self.current_run["final_end_to_end_corrected_bell_state"] = best_label
            else:
                best_label = "phi_plus"
                value = fidelities["phi_plus"]
            # log.logger.warning(
                # f"{self.name}: fidelity_components pair_type={pair_type} recover_endpoints={recover_endpoints} left={left_node} right={right_node} "
                # f"left_x_before={left_recovery['x_syndrome_before']} left_z_before={left_recovery['z_syndrome_before']} "
                # f"left_x_after={left_recovery['x_syndrome_after']} left_z_after={left_recovery['z_syndrome_after']} "
                # f"right_x_before={right_recovery['x_syndrome_before']} right_z_before={right_recovery['z_syndrome_before']} "
                # f"right_x_after={right_recovery['x_syndrome_after']} right_z_after={right_recovery['z_syndrome_after']} "
                # f"cx={cx:.6f} cy={cy:.6f} cz={cz:.6f} "
                # f"paulis={pauli_values} "
                # f"phi_plus={fidelities['phi_plus']:.6f} phi_minus={fidelities['phi_minus']:.6f} "
                # f"psi_plus={fidelities['psi_plus']:.6f} psi_minus={fidelities['psi_minus']:.6f} "
                # f"best={best_label} value={value:.6f}"
            # )
            return value

        if pair_type == "physical":
            raise ValueError(f"{self.name}: unsupported pair_type {pair_type}")

        # Select the encoded data blocks to score.
        if pair_type == "logical_link":
            left_keys = [m.qstate_key for m in left_app.data_qubits[right_node]]
            right_keys = [m.qstate_key for m in right_app.data_qubits[left_node]]
        elif pair_type == "logical_end":
            left_keys = [m.qstate_key for m in left_app.data_qubits[self._path_node_names[1]]]
            right_keys = [m.qstate_key for m in right_app.data_qubits[self._path_node_names[-2]]]
        else:
            raise ValueError(f"{self.name}: unknown pair_type {pair_type}")

        # Build the logical Pauli supports once, then pass them to the shared evaluator.
        px = self.code.get_logical_x_string()
        pz = self.code.get_logical_z_string()
        py = "".join("Y" if x == "X" and z == "Z" else x if x == "X" else z for x, z in zip(px, pz))
        return float(fidelity_from_keys(left_keys, right_keys, px, py, pz))

    def calculate_pair_fidelity(self, left_node: str, right_node: str, pair_type: str) -> float:
        """Compute raw pair fidelity without endpoint recovery.

        Args:
            left_node: Left node name.
            right_node: Right node name.
            pair_type: One of "physical", "logical_link", or "logical_end".

        Returns:
            float: Raw pair fidelity.
        """
        return self._calculate_pair_fidelity(left_node, right_node, pair_type, recover_endpoints=False)

    def calculate_pair_fidelity_corrected(self, left_node: str, right_node: str, pair_type: str) -> float:
        """Compute corrected pair fidelity with noiseless recovery on endpoint blocks.

        Args:
            left_node: Left node name.
            right_node: Right node name.
            pair_type: One of "physical", "logical_link", or "logical_end".

        Returns:
            float: Corrected pair fidelity.
        """
        return self._calculate_pair_fidelity(left_node, right_node, pair_type, recover_endpoints=True)

    def allocate_data_and_ancilla(self, neighbor: str) -> None:
        """Allocate data and required ancilla qubits for one link from local arrays.

        Args:
            neighbor: Remote node name this allocation is for.

        Returns:
            None
        """
        qec_ancillas = 3 if self.correction_mode in {"qec", "qec+cec"} else 0
        ft_ancillas = self.code.get_ft_required_ancillas(self.ft_prep_mode)
        required_ancillas = max(qec_ancillas, ft_ancillas)  # Reserve enough ancillas for the larger of QEC or FT prep.

        # Carve out the next logical data block from the node-local data memory array.
        data_array = self.node.components[f"{self.node.name}.DataMemoryArray"]
        data_offset = sum(len(block) for block in self.data_qubits.values())
        data_block = data_array.memories[data_offset:data_offset + self.n]
        if len(data_block) != self.n:
            raise RuntimeError(f"{self.name}: insufficient data memories for {neighbor} (need {self.n}, got {len(data_block)})")
        self.data_qubits[neighbor] = data_block

        # Reserve the next ancilla slice for this neighbor's local FT/QEC work.
        ancilla_array = self.node.components[f"{self.node.name}.AncillaMemoryArray"]
        ancilla_offset = sum(len(block) for block in self.ancilla_qubits.values())
        ancilla_block = ancilla_array.memories[ancilla_offset:ancilla_offset + required_ancillas]
        if required_ancillas > 0 and len(ancilla_block) < required_ancillas:
            raise RuntimeError(f"{self.name}: insufficient ancilla memories for {neighbor} (need {required_ancillas}, got {len(ancilla_block)})")
        self.ancilla_qubits[neighbor] = ancilla_block

        # Communication memories are populated later as physical Bell pairs arrive.
        self.current_run["comm_qubits"][neighbor] = []

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

        # Create the run-local comm bucket on first memory arrival for this neighbor.
        if neighbor not in self.current_run["comm_qubits"]:
            self.current_run["comm_qubits"][neighbor] = []

        # Cache local comm memory; once n are collected, schedule link-finalization barrier check.
        self.current_run["comm_qubits"][neighbor].append(info.memory)
        # Comm key became active now; initialize idle-time baseline at current sim time.
        now_ps = int(self.node.timeline.now())
        self.node.timeline.quantum_manager.last_idle_time_ps_by_key[info.memory.qstate_key] = now_ps
        local_names = [m.name for m in self.current_run["comm_qubits"][neighbor]]
        remote_names = [str(m.entangled_memory["memo_id"]) for m in self.current_run["comm_qubits"][neighbor]]
        # log.logger.info(f"{self.name}: physical_progress neighbor={neighbor} count={len(local_names)}/{self.n} unique_local={len(set(local_names))} unique_remote={len(set(remote_names))} last={info.memory.name}->{info.memory.entangled_memory['memo_id']}")
        # log.logger.debug(f"{self.name}: entangled comm memory neighbor={neighbor} count={len(self.current_run['comm_qubits'][neighbor])}/{self.n} index={info.index}")

        if len(self.current_run["comm_qubits"][neighbor]) != self.n:
            return

        process = Process(self, "finalize_physical_link_ready", [neighbor])
        event = Event(self.node.timeline.now(), process, self.node.timeline.schedule_counter)
        self.node.timeline.schedule(event)

    def finalize_physical_link_ready(self, neighbor: str) -> None:
        """Finalize one physical link after both endpoints have n ready memories.

        Args:
            neighbor: Adjacent peer for this physical link.

        Returns:
            None
        """
        if neighbor in self.current_run["link_ready"]:
            return

        local_count = len(self.current_run["comm_qubits"].get(neighbor, []))
        if local_count != self.n:
            return

        self.current_run["link_ready"].add(neighbor)
        # log.logger.info(f"{self.name}: link_ready neighbor={neighbor} ready_links={sorted(self.current_run['link_ready'])}")

        expected_ready = {n for n in (self._left_peer_name, self._right_peer_name) if n is not None}
        if not self.current_run["link_ready"].issuperset(expected_ready):
            return

        # Build node-local data blocks for encode stage once all required links are physically ready.
        if self._path_role == "edge":
            edge_neighbor = self._right_peer_name if self._right_peer_name is not None else self._left_peer_name
            data_blocks = [[m.qstate_key for m in self.data_qubits[edge_neighbor]]]
        else:
            data_blocks = [[m.qstate_key for m in self.data_qubits[self._left_peer_name]], [m.qstate_key for m in self.data_qubits[self._right_peer_name]]]

        # log.logger.warning(f'{self.name}: physical link ready, runtime = {time.time() - self.time_begin_of_run:.2f}s')

        # Encode only after both adjacent links are finalized as ready.
        # log.logger.info(f"{self.name}: triggering encode path_role={self._path_role} expected_ready={sorted(expected_ready)}")
        self.encode_data_qubits(data_blocks=data_blocks, ft_prep_mode=self.ft_prep_mode, max_ft_prep_shots=15)

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

        # Reservation window bookkeeping:
        # map memory_index -> reservation at start, then remove at end.
        reservation_protocol = self.node.network_manager.protocol_stack[1]
        for card in reservation_protocol.timecards:
            if reservation in card.reservations:
                add_process = Process(self.memo_to_reservation, "__setitem__", [card.memory_index, reservation])
                add_event = Event(reservation.start_time, add_process, self.node.timeline.schedule_counter)
                self.node.timeline.schedule(add_event)

                remove_process = Process(self.memo_to_reservation, "pop", [card.memory_index, None])
                remove_event = Event(reservation.end_time, remove_process, self.node.timeline.schedule_counter)
                self.node.timeline.schedule(remove_event)

    def get_other_physical_reservation(self, reservation: "Reservation") -> None:
        """Handle responder-side approved reservation and index mapping.

        Args:
            reservation: Reservation object where this node is responder.

        Returns:
            None
        """

        if int(reservation.start_time) not in self.scheduled_run_starts:
            self.scheduled_run_starts.append(int(reservation.start_time))  # Keep responder reset timing aligned with initiators.

        process = Process(self, "begin_run", [int(reservation.start_time), int(reservation.end_time)])
        event = Event(int(reservation.start_time), process, self.node.timeline.schedule_counter)
        self.node.timeline.schedule(event)

        # Same reservation window bookkeeping as initiator path.
        reservation_protocol = self.node.network_manager.protocol_stack[1]
        for card in reservation_protocol.timecards:
            if reservation in card.reservations:
                add_process = Process(self.memo_to_reservation, "__setitem__", [card.memory_index, reservation])
                add_event = Event(reservation.start_time, add_process, self.node.timeline.schedule_counter)
                self.node.timeline.schedule(add_event)

                remove_process = Process(self.memo_to_reservation, "pop", [card.memory_index, None])
                remove_event = Event(reservation.end_time, remove_process, self.node.timeline.schedule_counter)
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
        # log.logger.info(f"{self.name}: encode_data_qubits blocks={len(data_blocks)} ft_prep_mode={ft_prep_mode} max_shots={max_ft_prep_shots}")
        if self._path_role == "edge" and len(data_blocks) != 1:
            raise RuntimeError(f"{self.name}: edge node expects exactly 1 data block")
        if self._path_role == "middle" and len(data_blocks) != 2:
            raise RuntimeError(f"{self.name}: middle node expects exactly 2 data blocks")
        if max_ft_prep_shots < 1:
            raise RuntimeError(f"{self.name}: max_ft_prep_shots must be >= 1")

        qm = self.node.timeline.quantum_manager
        total_prep_duration_ps = 0

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
                ancilla_memories = self.ancilla_qubits[block_neighbor]
                if len(ancilla_memories) < need:
                    raise RuntimeError(f"{self.name}: {ft_prep_mode} FT mode requires >={need} ancillas")
                ancilla_keys = [ancilla_memories[i].qstate_key for i in range(need)]

            accepted, prep_duration_ps, ft_prep_shots_used = self.code.run_encode_ft_prep(qm=qm, data_keys=block_keys, ancilla_keys=ancilla_keys, ft_prep_mode=ft_prep_mode, max_ft_prep_shots=max_ft_prep_shots, meas_samp=self.node.get_generator().random(), logical_state=logical_state)
            total_prep_duration_ps += prep_duration_ps
            ft_prep_retries = ft_prep_shots_used - 1
            # log.logger.warning(
                # f"{self.name}: ft_prep_result neighbor={block_neighbor} logical_state={logical_state} "
                # f"mode={ft_prep_mode} accepted={accepted} shots_used={ft_prep_shots_used} "
                # f"retries={ft_prep_retries}"
            # )
            # log.logger.info(f"{self.name}: block encode accepted={accepted} ft_prep_mode={ft_prep_mode}")
            
            if not accepted:
                raise RuntimeError(f"{self.name}: FT prep failed after {max_ft_prep_shots} shots")

            # Encoding consumed this block now; reset idle-time baseline for involved keys.
            now_ps = int(self.node.timeline.now()) + prep_duration_ps
            for key in block_keys:
                qm.last_idle_time_ps_by_key[key] = now_ps
            for key in ancilla_keys:
                qm.last_idle_time_ps_by_key[key] = now_ps
        # Delay TCNOT launch by the total local FT-prep processing time.
        tcnot_start_t = int(self.node.timeline.now()) + total_prep_duration_ps
        # log.logger.info(f"{self.name}: schedule_tcnot_start now={int(self.node.timeline.now())} prep_duration_ps={total_prep_duration_ps} tcnot_start_t={tcnot_start_t}")
        for ready_neighbor in list(self.current_run["link_ready"]):
            reservation_for_neighbor = None
            for res in self.memo_to_reservation.values():
                other = res.responder if res.initiator == self.node.name else res.initiator
                if other == ready_neighbor:
                    reservation_for_neighbor = res
                    break

            if reservation_for_neighbor is None:
                raise RuntimeError(f"{self.name}: missing reservation for ready link {ready_neighbor}")

            process = Process(self, "initialize_teleported_cnot", [ready_neighbor, reservation_for_neighbor])
            event = Event(tcnot_start_t, process, self.node.timeline.schedule_counter)
            self.node.timeline.schedule(event)

        # log.logger.debug(f"{self.name}: encoded {len(data_blocks)} block(s) with ft_prep_mode={ft_prep_mode}")
        # log.logger.warning(f'{self.name}: encode data qubits done, runtime = {time.time() - self.time_begin_of_run:.2f}s')

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
        # log.logger.info(f"{self.name}: initialize_tcnot neighbor={neighbor} role={role} comm={len(self.current_run['comm_qubits'][neighbor])} data={len(self.data_qubits[neighbor])}")

        tcnot = self.current_run["tcnot_protocols"].get(neighbor)
        if tcnot is None:
            tcnot = TeleportedCNOTProtocol(
                owner=self.node,
                name=f"TeleportedCNOT_{self.node.name}_to_{neighbor}",
                role=role,
                remote_node_name=neighbor,
                data_qubit_keys=[m.qstate_key for m in self.data_qubits[neighbor]],
                communication_qubit_keys=[m.qstate_key for m in self.current_run["comm_qubits"][neighbor]])
            self.current_run["tcnot_protocols"][neighbor] = tcnot
        elif tcnot.role != role:
            raise RuntimeError(f"{self.name}: TCNOT role mismatch for {neighbor}: have {tcnot.role}, expected {role}")

        if tcnot.started:
            return

        tcnot.start()

        pending_tcnot_messages = self.current_run["pending_tcnot_messages"].pop(neighbor, [])
        if pending_tcnot_messages:
            # log.logger.info(f"{self.name}: replay buffered TCNOT messages neighbor={neighbor} count={len(pending_tcnot_messages)}")
            for pending_msg in pending_tcnot_messages:
                tcnot.received_message(neighbor, pending_msg)  # Deliver queued TCNOT payload messages after local start().

    def on_teleported_cnot_complete(self, reservation_or_neighbor: "Reservation | str") -> None:
        """Handle TCNOT completion bookkeeping for one adjacent link.

        Args:
            reservation_or_neighbor: Reservation object or neighbor node name.

        Returns:
            None
        """
        # log.logger.warning(f'{self.name} TCNOT complete, runtime = {time.time() - self.time_begin_of_run:.2f}s')
        
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
        # log.logger.info(f"{self.name}: tcnot_complete neighbor={neighbor}")
        # Record one-link logical fidelity only on one designated side.
        neighbor_pos = self._path_node_names.index(neighbor)
        if self._path_position < neighbor_pos:
            # measured_fidelity = self.calculate_pair_fidelity(self.node.name, neighbor, "logical_link")
            # self.current_run["link_logical_fidelities"][neighbor] = measured_fidelity
            # self.current_run["link_logical_fidelities_corrected"][neighbor] = measured_fidelity
            pass

        # QRE barrier: this node launches QRE only after all its adjacent TCNOT links complete.
        self.current_run["tcnot_done"].add(neighbor)
        required_tcnot = {n for n in (self._left_peer_name, self._right_peer_name) if n is not None}
        # log.logger.info(f"{self.name}: tcnot_barrier run_id={self.current_run['run_id']} done={self.current_run['tcnot_done']} required={required_tcnot} role={self._path_role}")
        if not self.current_run["tcnot_done"].issuperset(required_tcnot):
            # log.logger.info(f"{self.name}: waiting_for_tcnot_barrier done={self.current_run['tcnot_done']} required={required_tcnot}")
            return

        for peer in required_tcnot:
            have = len(self.data_qubits.get(peer, []))
            if have != self.n:
                # log.logger.info(f"{self.name}: waiting_for_data_block peer={peer} have={have} need={self.n}")
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
        # log.logger.info(f"{self.name}: start_qre_for_link neighbor={neighbor}")
        protocol = self.current_run["qre_protocols"].get(neighbor)
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

        pending_qre_messages_by_src = dict(self.current_run["pending_qre_messages"])
        self.current_run["pending_qre_messages"].clear()
        if pending_qre_messages_by_src:
            total_pending = sum(len(messages) for messages in pending_qre_messages_by_src.values())
            # log.logger.info(f"{self.name}: replay buffered QRE messages neighbor={neighbor} count={total_pending}")
            for src, pending_messages in pending_qre_messages_by_src.items():
                for pending_msg in pending_messages:
                    protocol.received_message(src, pending_msg)  # Deliver queued frame updates once the local QRE protocol is active.

# ---------- QRE phase: protocol launch and completion handling ----------

    def logical_pair_complete(self, neighbor: str, result: dict[str, object] | None) -> None:
        """Store per-link QRE result and finalize when all links complete.

        Args:
            neighbor: Neighbor associated with completed QRE protocol.
            result: Optional result payload from QRE protocol.

        Returns:
            None
        """
        if result is not None:
            pass

        # Only protocols that actually started belong to the current attempt.
        # Middle nodes keep a second neighbor-scoped QRE object in IDLE, and
        # that object should not block attempt completion.
        active_protocols = [protocol for protocol in self.current_run["qre_protocols"].values() if protocol.is_running or protocol.current_phase == "COMPLETE"]
        if len(active_protocols) == 0:
            return
        if not all(protocol.current_phase == "COMPLETE" for protocol in active_protocols):
            return

        # Initiator-only: compute end-to-end logical fidelity for this attempt.
        if self._path_position == 0:
            left_keys = [m.qstate_key for m in self.data_qubits[self._path_node_names[1]]]
            right_router = self.node.timeline.get_entity_by_name(self._path_node_names[-1])
            right_app = right_router.request_logical_pair_app
            right_keys = [m.qstate_key for m in right_app.data_qubits[self._path_node_names[-2]]]
            final_fidelity_corrected = self.calculate_pair_fidelity_corrected(self._path_node_names[0], self._path_node_names[-1], "logical_end")
            self.current_run["final_end_to_end_fidelity_corrected"] = final_fidelity_corrected
            self.current_run["final_end_to_end_fidelity"] = final_fidelity_corrected  # Persist the corrected run result as the default metric.
            self.current_run["completion_time"] = int(self.node.timeline.now())
            self.current_run["status"] = "complete"

        # Record compact stats for this completed local run.
        final_fidelity = self.current_run["final_end_to_end_fidelity"]
        meets_target = (final_fidelity >= self.required_end_to_end_logical_fidelity if final_fidelity is not None else True)
        self.current_run["success"] = bool(meets_target)

        run_id = int(self.current_run["run_id"])
        latency_ps = (None if self.current_run["start_time"] is None or self.current_run["completion_time"] is None else int(self.current_run["completion_time"]) - int(self.current_run["start_time"]))

        self.completed_run_count += 1
        if self.current_run["final_end_to_end_fidelity_corrected"] is not None:
            self.sum_final_fidelity_corrected += float(self.current_run["final_end_to_end_fidelity_corrected"])
        if latency_ps is not None:
            self.sum_latency_ps += int(latency_ps)

        if self._path_position == 0:
            self._log_critical_run_summary(run_id, latency_ps)

        if self._initiator_responder is not None:
            self._schedule_next_initiator_run()

        current_start_t = int(self.current_run["start_time"]) if self.current_run["start_time"] is not None else -1
        current_end_t = int(self.current_run["end_time"]) if self.current_run["end_time"] is not None else int(self.node.timeline.now())
        next_start_t = min((start_t for start_t in self.scheduled_run_starts if start_t > current_start_t), default=None)
        now_ps = int(self.node.timeline.now())
        if next_start_t is None:
            cleanup_time = max(now_ps, current_end_t)
        else:
            cleanup_time = max(now_ps, int(next_start_t) - 1)

        process = Process(self, "reset_run", [])
        event = Event(cleanup_time, process, self.node.timeline.schedule_counter)
        self.node.timeline.schedule(event)

        if self._path_position != 0:
            return

        if meets_target:
            pass
        else:
            pass
