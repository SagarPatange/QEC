"""
Request Logical Pair Application — Multi-link logical entanglement.

Each node runs one instance of this app. The app tracks per-link qubit
allocation using dicts keyed by neighbor node name. An endpoint node
has 1 link, a middle node has 2 links. All operations are LOCC
compliant — each node only operates on its own local qubits and
communicates classically.

Pipeline per link:
1. Generate n physical Bell pairs (communication memories)
2. Calculate physical Bell pair fidelities via Pauli tomography
3. Encode n data qubits into logical qubit using CSS code
4. Teleported CNOT to create logical entanglement across link
5. Simultaneous entanglement swapping at all middle nodes (single round)
"""

import numpy as np
import stim
from enum import Enum, auto
from typing import TYPE_CHECKING
from sequence.resource_management.memory_manager import MemoryInfo
from sequence.app.request_app import RequestApp
from sequence.utils import log
from sequence.kernel.process import Process
from sequence.kernel.event import Event
from sequence.message import Message
from TeleportedCNOT import TeleportedCNOTProtocol
from css_codes import get_css_code
from sequence.kernel.quantum_state import StabilizerState

if TYPE_CHECKING:
    from sequence.topology.node import QuantumRouter
    from sequence.network_management.reservation import Reservation


# ======================================================================
# Classical message types for LOCC-compliant swap coordination
# ======================================================================

class SwapMsgType(Enum):
    LINKS_READY = auto()      # Middle node → Coordinator
    SWAP_COMPLETE = auto()    # Middle node → R0 (correction endpoint)


class SwapNotificationMessage(Message):
    """Classical message for LOCC-compliant swap coordination.

    LINKS_READY:  Non-coordinator middle node → Coordinator.
                  Signals that both links are complete and qubits are ready.
    SWAP_COMPLETE: Any middle node → Correction endpoint (R0).
                   Signals that local Bell measurement is done.
    """

    def __init__(self, msg_type: SwapMsgType, receiver_name: str, sender_node: str):
        super().__init__(msg_type, receiver_name)
        self.sender_node = sender_node


# ======================================================================
# Main application
# ======================================================================

class RequestLogicalPairApp(RequestApp):
    """Application for generating logical Bell pairs across multiple links.

    LOCC compliant: only tracks local state. Neighbor information comes
    from the memory manager callbacks (MemoryInfo.remote_node) and
    reservation objects.

    Per-link qubit mapping is built as links are established via
    start() (initiator) or get_other_reservation() (responder).
    """

    # ------------------------------------------------------------------
    # Construction & configuration
    # ------------------------------------------------------------------

    def __init__(self, node: "QuantumRouter", css_code: str = "[[7,1,3]]"):
        super().__init__(node)

        # Register on node so protocols can find this app
        self.node.request_logical_pair_app = self

        # Protocol attributes for SeQUeNCe message dispatch
        self.protocol_type = "RequestLogicalPairApp"
        self.name = f"RequestLogicalPairApp.{node.name}"
        self.node.protocols.append(self)

        # CSS code parameters
        self.code = get_css_code(css_code)
        self.n = self.code.n
        self.css_code = css_code

        # Per-link qubit mapping (keyed by neighbor node name)
        self.data_qubits = {}    # neighbor -> list of n Memory objects
        self.comm_qubits = {}    # neighbor -> list of n Memory objects
        self._next_data_index = 0
        self._tcnot_protocols = {}  # neighbor -> TeleportedCNOTProtocol
        self._initial_link_fidelities = {}  # neighbor -> average physical Bell-pair fidelity
        self._link_logical_fidelities = {}  # neighbor -> logical Bell-pair fidelity after TCNOT
        self._final_end_to_end_fidelity = None
        self._last_idle_update_ps = {}  # qstate_key -> last idle-noise update time (ps)
        self._memory_entangled_since_ps = {}  # qstate_key -> time memory became entangled for current link (ps)
        self._initial_pair_fidelity_rows = {}  # neighbor -> list[dict]
        self._pre_encoding_pair_idle_rows = {}  # neighbor -> list[dict]
        self._post_idle_pair_fidelity_rows = {}  # neighbor -> list[dict]

        # Swap coordination state (all instance-level, LOCC compliant)
        self._completed_link_neighbors = set()
        self._swapping_done = False
        self._swap_config = None
        self._ready_nodes = set()
        self._is_final_action_node = False

        # Simulation config
        self.encoding_enabled = True
        self.teleported_cnot_enabled = False

        # FT prep config (set by RouterNetTopo2G from JSON config)
        self.ft_prep_mode = getattr(node, 'ft_prep_mode', 'none')
        self.ft_max_retries = getattr(node, 'ft_max_retries', 1)
        self.ft_postselect = getattr(node, 'ft_postselect', False)
        self.idle_decoherence_enabled = getattr(node, 'idle_decoherence_enabled', True)
        self.idle_decoherence_debug = getattr(node, 'idle_decoherence_debug', False)

        # Read prep_fidelity from data memory hardware (set via config template)
        data_array_name = f"{self.node.name}.DataMemoryArray"
        data_array = self.node.components.get(data_array_name)
        if data_array and len(data_array.memories) > 0:
            self.prep_fidelity = data_array.memories[0].raw_fidelity
        else:
            self.prep_fidelity = 1.0

    def _get_idle_t2_seconds(self, memory):
        """Return coherence time in seconds for coarse memory idling."""
        t2 = getattr(memory, "coherence_time", None)
        if t2 is None or t2 <= 0:
            return None
        return float(t2)

    def _get_memory_qm_key(self, memory):
        """Return the quantum-manager key backing a memory."""
        return getattr(memory, "qstate_key", None)

    def _reference_idle_time_ps(self, memory, now_ps: int):
        """Best available reference time for the first idle update."""
        key = self._get_memory_qm_key(memory)
        if key in self._last_idle_update_ps:
            return self._last_idle_update_ps[key]

        last_update = getattr(memory, "last_update_time", -1)
        if last_update is not None and last_update >= 0:
            return int(last_update)

        generation_time = getattr(memory, "generation_time", -1)
        if generation_time is not None and generation_time >= 0:
            return int(generation_time)

        return int(now_ps)

    def _apply_idle_decoherence_to_memories(self, memories, stage: str = ""):
        """Apply coarse dephasing-only idle decoherence since the last touch."""
        if not self.idle_decoherence_enabled:
            return

        now_ps = self.node.timeline.now()
        qm = self.node.timeline.quantum_manager
        seen_keys = set()

        for memory in memories:
            if memory is None:
                continue

            key = self._get_memory_qm_key(memory)
            if key is None or key in seen_keys:
                continue
            seen_keys.add(key)

            t2_sec = self._get_idle_t2_seconds(memory)
            if t2_sec is None:
                self._last_idle_update_ps[key] = now_ps
                continue

            last_ps = self._reference_idle_time_ps(memory, now_ps)
            dt_ps = now_ps - last_ps
            if dt_ps <= 0:
                self._last_idle_update_ps[key] = now_ps
                continue

            dt_sec = dt_ps * 1e-12
            p_idle = 0.5 * (1 - np.exp(-dt_sec / t2_sec))
            if p_idle > 0:
                state = qm.states[key]
                state.circuit.append("Z_ERROR", [key], [float(p_idle)])
                if hasattr(state, "_tableau"):
                    state._tableau = None
                if self.idle_decoherence_debug:
                    debug_msg = (
                        f"{self.node.name}: idle decoherence stage={stage or 'unknown'} "
                        f"memory={getattr(memory, 'name', key)} key={key} "
                        f"dt_sec={dt_sec:.6e} T2={t2_sec:.6e} p_idle={p_idle:.6e}"
                    )
                    print(debug_msg)
                    log.logger.info(debug_msg)
            elif self.idle_decoherence_debug:
                debug_msg = (
                    f"{self.node.name}: idle decoherence stage={stage or 'unknown'} "
                    f"memory={getattr(memory, 'name', key)} key={key} "
                    f"dt_sec={dt_sec:.6e} T2={t2_sec:.6e} p_idle=0.0"
                )
                print(debug_msg)
                log.logger.info(debug_msg)

            self._last_idle_update_ps[key] = now_ps

    def _apply_pre_encoding_idle_decoherence_to_comm_memories(self, neighbor, memories):
        """Apply comm-memory idling from Bell-pair ready time until encoding starts."""
        if not self.idle_decoherence_enabled:
            return

        now_ps = self.node.timeline.now()
        qm = self.node.timeline.quantum_manager
        seen_keys = set()
        pair_rows = []

        for pair_index, memory in enumerate(memories):
            if memory is None:
                continue

            key = self._get_memory_qm_key(memory)
            if key is None or key in seen_keys:
                continue
            seen_keys.add(key)

            t2_sec = self._get_idle_t2_seconds(memory)
            if t2_sec is None:
                self._last_idle_update_ps[key] = now_ps
                continue

            entangled_ps = self._memory_entangled_since_ps.get(key, None)
            if entangled_ps is None:
                generation_time = getattr(memory, "generation_time", -1)
                entangled_ps = generation_time if generation_time is not None and generation_time >= 0 else now_ps

            dt_ps = now_ps - int(entangled_ps)
            if dt_ps <= 0:
                if self.idle_decoherence_debug:
                    print(
                        f"{self.node.name}: idle decoherence stage=pre_encoding "
                        f"memory={getattr(memory, 'name', key)} key={key} "
                        f"dt_sec=0.0 T2={t2_sec:.6e} p_idle=0.0"
                    )
                self._last_idle_update_ps[key] = now_ps
                pair_rows.append(
                    {
                        "pair_index": pair_index,
                        "memory_name": getattr(memory, "name", str(key)),
                        "memory_key": key,
                        "wait_time_sec": 0.0,
                        "p_idle": 0.0,
                    }
                )
                continue

            dt_sec = dt_ps * 1e-12
            p_idle = 0.5 * (1 - np.exp(-dt_sec / t2_sec))
            if p_idle > 0:
                state = qm.states[key]
                state.circuit.append("Z_ERROR", [key], [float(p_idle)])
                if hasattr(state, "_tableau"):
                    state._tableau = None

            if self.idle_decoherence_debug:
                print(
                    f"{self.node.name}: idle decoherence stage=pre_encoding "
                    f"memory={getattr(memory, 'name', key)} key={key} "
                    f"dt_sec={dt_sec:.6e} T2={t2_sec:.6e} p_idle={float(max(p_idle, 0.0)):.6e}"
                )

            self._last_idle_update_ps[key] = now_ps
            pair_rows.append(
                {
                    "pair_index": pair_index,
                    "memory_name": getattr(memory, "name", str(key)),
                    "memory_key": key,
                    "wait_time_sec": float(dt_sec),
                    "p_idle": float(max(p_idle, 0.0)),
                }
            )

        self._pre_encoding_pair_idle_rows[neighbor] = pair_rows

    def _estimate_bell_pair_fidelity(self, local_key, remote_key, shots: int = 10000, seed_tag: str = "") -> float:
        """Estimate Bell-pair fidelity by Pauli tomography for one physical pair."""
        qm = self.node.timeline.quantum_manager
        keys = [local_key, remote_key]

        state = qm.states[local_key]
        base_seed = hash(f"{self.node.name}_{local_key}_{remote_key}_{seed_tag}") % (2**31)

        correlations = {}
        for basis in ["X", "Y", "Z"]:
            meas_circuit = state.circuit.copy()

            for key in keys:
                if basis == "X":
                    meas_circuit.append("H", [key])
                elif basis == "Y":
                    meas_circuit.append("S_DAG", [key])
                    meas_circuit.append("H", [key])

            for key in keys:
                meas_circuit.append("M", [key])

            basis_seed = abs(base_seed + hash(basis)) % (2**63)
            sampler = meas_circuit.compile_sampler(seed=basis_seed)
            measurements = sampler.sample(shots=shots)

            num_measurements = measurements.shape[1]
            m0 = measurements[:, num_measurements - 2]
            m1 = measurements[:, num_measurements - 1]

            eigenvalues = (1 - 2*m0) * (1 - 2*m1)
            correlations[basis] = float(np.mean(eigenvalues))

        return float(np.clip((1 + correlations["X"] - correlations["Y"] + correlations["Z"]) / 4, 0.0, 1.0))

    def _mark_memories_touched(self, memories):
        """Advance idle-decoherence reference time after active use."""
        now_ps = self.node.timeline.now()
        for memory in memories:
            if memory is None:
                continue
            key = self._get_memory_qm_key(memory)
            if key is not None:
                self._last_idle_update_ps[key] = now_ps

    def set_swap_config(self, config):
        """Set this node's swap configuration (called during setup)."""
        self._swap_config = config
        role = config.get("role", "unknown")
        if role == "coordinator":
            log.logger.info(f"{self.node.name}: Swap config — coordinator, "
                            f"swap_nodes={config['swap_nodes']}, "
                            f"wait_for={config['wait_for']}")
        elif role == "swap_node":
            log.logger.info(f"{self.node.name}: Swap config — swap_node, "
                            f"reports to {config['coordinator']}")
        elif role == "correction_endpoint":
            log.logger.info(f"{self.node.name}: Swap config — correction_endpoint, "
                            f"expects {config['num_swap_nodes']} swap measurements")

    def set_final_action_node(self):
        """Mark this node as responsible for stopping the simulation."""
        self._is_final_action_node = True
        log.logger.info(f"{self.node.name}: Designated as final action node")

    @staticmethod
    def build_swap_schedule(node_names):
        """Build single-round simultaneous swap schedule for an N-node chain.

        One coordinator node merges ALL link circuits and performs ALL
        swap Bell measurements simultaneously, then applies corrections
        in one pass — matching the paper's protocol (PhysRevA.79.032325).

        For N=5 chain [R0, R1, R2, R3, R4]:
            Coordinator: R2 (center middle node)
            Swap nodes: R1, R2, R3 (all swap simultaneously)
            R1, R3 send LINKS_READY to R2; R2 performs all swaps at once.

        For N=3 chain [R0, R1, R2]:
            Coordinator: R1 (only middle node, empty wait_for)

        Args:
            node_names: Ordered list of node names in the chain

        Returns:
            (configs, final_node_name): configs maps node_name -> swap
            config dict; final_node_name is the coordinator (or None).
        """
        configs = {}
        N = len(node_names)
        if N <= 2:
            return configs, None

        middle_nodes = node_names[1:-1]
        coordinator = middle_nodes[len(middle_nodes) // 2]

        # Correction endpoint (left endpoint, R0)
        configs[node_names[0]] = {
            "role": "correction_endpoint",
            "correction_key": node_names[1],
            "num_swap_nodes": len(middle_nodes),
            "left_endpoint": node_names[0],
            "left_endpoint_key": node_names[1],
            "right_endpoint": node_names[-1],
            "right_endpoint_key": node_names[-2],
        }

        # Middle nodes
        for node in middle_nodes:
            if node == coordinator:
                configs[node] = {
                    "role": "coordinator",
                    "chain_node_names": list(node_names),
                    "swap_nodes": list(middle_nodes),
                    "wait_for": [nd for nd in middle_nodes if nd != coordinator],
                    "correction_node": node_names[0],
                }
            else:
                configs[node] = {
                    "role": "swap_node",
                    "coordinator": coordinator,
                    "correction_node": node_names[0],
                }

        return configs, node_names[0]

    # ------------------------------------------------------------------
    # Public interface (SeQUeNCe callbacks)
    # ------------------------------------------------------------------

    def start(self, responder: str, start_time: int, end_time: int,
              target_fidelity: float = 0.8,
              teleported_cnot_enabled: bool = False):
        """Initiate Bell pair generation with a neighbor (initiator/Alice role).

        Called from the test function for each link this node initiates.
        """
        self.teleported_cnot_enabled = teleported_cnot_enabled
        self._allocate_qubits_for_link(responder)

        super().start(
            responder=responder,
            start_t=start_time,
            end_t=end_time,
            memo_size=self.n,
            fidelity=target_fidelity
        )

        log.logger.info(f"{self.node.name}: Initiated link to {responder}, "
                        f"requesting {self.n} Bell pairs for {self.css_code}")

    def get_other_reservation(self, reservation: "Reservation") -> None:
        """Called when this node is the responder (Bob) for a reservation."""
        neighbor = self._get_neighbor(reservation)
        self._allocate_qubits_for_link(neighbor)
        super().get_other_reservation(reservation)

        log.logger.info(f"{self.node.name}: Configured as responder for "
                        f"{neighbor}'s reservation")

    def get_memory(self, info: "MemoryInfo") -> None:
        """Process a completed Bell pair.

        Called by the resource manager when a communication memory
        becomes entangled and is idle (no more rules match).
        """
        if info.state != "ENTANGLED":
            return
        if info.index not in self.memo_to_reservation:
            return

        reservation = self.memo_to_reservation[info.index]
        neighbor = self._get_neighbor(reservation)
        if getattr(info.memory, "qstate_key", None) is not None:
            self._memory_entangled_since_ps[info.memory.qstate_key] = self.node.timeline.now()

        log.logger.info(f"{self.node.name}: Bell pair on memory {info.index} "
                        f"with {neighbor} at t={self.node.timeline.now()*1e-12:.6f}s")

        ready_count = len(self._get_entangled_memories(reservation))
        log.logger.info(f"{self.node.name}: Link to {neighbor}: "
                        f"{ready_count}/{self.n} Bell pairs ready")

        if ready_count >= self.n:
            process = Process(self, '_calculate_physical_bell_pair_fidelities',
                              [reservation])
            event = Event(self.node.timeline.now() + 1000, process, priority=0)
            self.node.timeline.schedule(event)

            if self.encoding_enabled:
                process = Process(self, '_start_encoding', [reservation])
                event = Event(self.node.timeline.now() + 1000, process, priority=1)
                self.node.timeline.schedule(event)

    def received_message(self, src: str, msg) -> bool:
        """Handle incoming classical messages from other nodes.

        Dispatched by SeQUeNCe's node.receive_message() when
        msg.receiver matches this protocol's name.
        """
        if isinstance(msg, SwapNotificationMessage):
            if msg.msg_type == SwapMsgType.LINKS_READY:
                log.logger.info(f"{self.node.name}: Received LINKS_READY from "
                                f"{msg.sender_node} (via {src})")
                self._on_node_links_ready(msg.sender_node)
            elif msg.msg_type == SwapMsgType.SWAP_COMPLETE:
                log.logger.info(f"{self.node.name}: Received SWAP_COMPLETE from "
                                f"{msg.sender_node} (via {src})")
                self._on_swap_measurement_received(msg.sender_node)
            return True
        return False

    # ------------------------------------------------------------------
    # Stage 1: Physical Bell pair fidelity
    # ------------------------------------------------------------------

    def _calculate_physical_bell_pair_fidelities(self, reservation,
                                                  shots: int = 10000):
        """Calculate fidelities for all Bell pairs on a given link.

        Uses Pauli tomography: F = (1 + <XX> - <YY> + <ZZ>) / 4
        for each physical Bell pair. Updates the memory manager's
        fidelity with the measured value.
        """
        qm = self.node.timeline.quantum_manager
        neighbor = self._get_neighbor(reservation)
        entangled_memories = self._get_entangled_memories(reservation)

        log.logger.info(f"{self.node.name}: Calculating tomography fidelities for "
                        f"{len(entangled_memories)} Bell pairs")

        fidelities = []
        pair_rows = []
        for pair_id, mem_info in enumerate(entangled_memories):
            local_key = mem_info.memory.qstate_key
            remote_memory = self.node.timeline.get_entity_by_name(mem_info.remote_memo)
            remote_key = remote_memory.qstate_key
            fidelity = self._estimate_bell_pair_fidelity(local_key, remote_key, shots=shots, seed_tag=f"initial_{pair_id}")

            mem_info.fidelity = fidelity
            fidelities.append(fidelity)
            pair_rows.append(
                {
                    "pair_index": pair_id,
                    "local_memory": mem_info.memory.name,
                    "remote_memory": mem_info.remote_memo,
                    "initial_pair_fidelity": float(fidelity),
                }
            )
            log.logger.info(f"{self.node.name}: Pair {pair_id} fidelity = "
                            f"{fidelity:.4f}")

        if fidelities:
            avg_fidelity = float(np.mean(fidelities))
            self._initial_link_fidelities[neighbor] = avg_fidelity
        self._initial_pair_fidelity_rows[neighbor] = pair_rows

    # ------------------------------------------------------------------
    # Stage 2: Encoding
    # ------------------------------------------------------------------

    def _start_encoding(self, reservation):
        """Encode data qubits into a logical qubit for a specific link.

        The logical state depends on role:
        initiator (Alice) encodes |+>_L, responder (Bob) encodes |0>_L.
        """
        neighbor = self._get_neighbor(reservation)
        is_initiator = reservation.initiator == self.node.name
        if self.code.cx_reversed:
            # Non-self-dual codes: physical CX(A→B) = CX_L(B→A)
            # Swap encoding: initiator (Alice) = |0⟩_L, responder (Bob) = |+⟩_L
            logical_state = '0' if is_initiator else '+'
        else:
            logical_state = '+' if is_initiator else '0'

        log.logger.info(f"{self.node.name}: Starting {self.css_code} encoding "
                        f"to |{logical_state}>_L for link to {neighbor}")
        log.logger.info(f"{self.node.name}: Data prep fidelity parameter = "
                        f"{self.prep_fidelity:.6f}")

        try:
            qm = self.node.timeline.quantum_manager
            data_qubits = self.data_qubits[neighbor]
            self._get_comm_qubits_for_link(reservation)
            comm_qubits = self.comm_qubits.get(neighbor, [])
            self._apply_pre_encoding_idle_decoherence_to_comm_memories(neighbor, comm_qubits)
            if len(data_qubits) != self.n:
                raise ValueError(f"Expected {self.n} data qubits, "
                                 f"got {len(data_qubits)}")

            data_keys = [memory.qstate_key for memory in data_qubits]

            # Preparation noise
            if self.prep_fidelity < 1.0:
                noise_param = 1.0 - self.prep_fidelity
                for i in range(len(data_qubits)):
                    noise_circuit = stim.Circuit()
                    noise_circuit.append("DEPOLARIZE1", [data_keys[i]], noise_param)
                    qm.run_circuit(noise_circuit, [data_keys[i]])

            # Prepare logical state (with optional FT verification)
            prep_circuit = stim.Circuit()
            ancilla_offset = self.n if self.ft_prep_mode != "none" else None
            meta = self.code.prepare_logical_state(
                prep_circuit, logical_state, offset=0,
                ft_mode=self.ft_prep_mode,
                ancilla_offset=ancilla_offset,
                ft_kwargs={"add_detectors": self.ft_postselect},
            )

            # Allocate ancilla QM keys for FT verification qubits
            anc_used = meta.get("ancilla_used", 0)
            all_keys = data_keys
            if anc_used > 0:
                anc_keys = [qm.new() for _ in range(anc_used)]
                all_keys = data_keys + anc_keys

            if self.ft_prep_mode != "none" and self.ft_postselect and meta.get("detector_count", 0) > 0:
                if self.ft_max_retries > 1:
                    log.logger.warning(
                        f"{self.node.name}: ft_max_retries={self.ft_max_retries} requested, "
                        "but link-level regeneration is not implemented; using a single postselected attempt."
                    )

                event_res = qm.run_circuit_with_events(
                    prep_circuit,
                    all_keys,
                    shots=1,
                    commit=True,
                )
                dets = event_res.get("detectors")
                accepted = dets is None or int(dets[0].sum()) == 0
                log.logger.info(
                    f"{self.node.name}: FT prep postselection "
                    f"{'accepted' if accepted else 'rejected'} for link to {neighbor}"
                )
                if not accepted:
                    raise RuntimeError(
                        f"FT preparation rejected for link to {neighbor}; "
                        "regeneration/retry of the underlying entangled block is not implemented."
                    )
            else:
                qm.run_circuit(prep_circuit, all_keys)

            log.logger.info(f"{self.node.name}: FT prep meta: {meta}")
            self._mark_memories_touched(data_qubits)

            log.logger.info(f"{self.node.name}: Encoding complete for link to {neighbor}")

            process = Process(self, '_encoding_completed', [reservation])
            event = Event(self.node.timeline.now() + 1, process)
            self.node.timeline.schedule(event)

        except Exception as e:
            log.logger.error(f"{self.node.name}: Encoding FAILED: {e}")
            import traceback
            log.logger.error(traceback.format_exc())
            raise

    def _encoding_completed(self, reservation):
        """Called when QEC encoding is complete for a link.

        If teleported CNOT is enabled and this node is the initiator,
        proceeds to TCNOT. Otherwise logs completion.
        """
        neighbor = self._get_neighbor(reservation)
        is_initiator = (reservation.initiator == self.node.name)

        log.logger.info(f"{self.node.name}: Encoding completed for link to {neighbor} "
                        f"(role: {'initiator' if is_initiator else 'responder'})")

        if self.teleported_cnot_enabled and is_initiator:
            self._get_comm_qubits_for_link(reservation)
            self._initialize_teleported_cnot(reservation)

    # ------------------------------------------------------------------
    # Stage 3: Teleported CNOT
    # ------------------------------------------------------------------

    def _initialize_teleported_cnot(self, reservation):
        """Initialize and start the teleported CNOT protocol for a link.

        Only the initiator (Alice) calls this. Creates protocol instances for
        both Alice and Bob, registers them on their respective nodes, and
        starts the protocol. Alice holds a reference to Bob's protocol so
        that Stim can group all qubits into a single stabilizer tableau
        (simulation workaround — in reality each node would initialize
        its own protocol via classical coordination).
        """
        neighbor = self._get_neighbor(reservation)

        if reservation.initiator != self.node.name:
            return

        alice_data = self.data_qubits[neighbor]
        alice_comm = self.comm_qubits[neighbor]
        self._apply_idle_decoherence_to_memories(alice_data + alice_comm, stage="pre_tcnot_alice")

        bob_node = self.node.timeline.get_entity_by_name(neighbor)
        if bob_node is None:
            raise RuntimeError(f"{self.node.name}: Cannot find node {neighbor}")

        bob_app = bob_node.request_logical_pair_app
        bob_app._get_comm_qubits_for_link(reservation)
        bob_data = bob_app.data_qubits[self.node.name]
        bob_comm_unordered = bob_app.comm_qubits[self.node.name]
        bob_app._apply_idle_decoherence_to_memories(bob_data + bob_comm_unordered, stage="pre_tcnot_bob")

        # Reorder Bob's comm qubits to match Alice's entanglement pairing
        bob_comm = self._reorder_bob_comm_qubits(alice_comm, bob_comm_unordered)
        bob_app.comm_qubits[self.node.name] = bob_comm
        self._record_post_idle_pair_fidelities(neighbor, bob_app, alice_comm, bob_comm)

        # Create protocol instances
        alice_protocol_name = f"TeleportedCNOT_{self.node.name}_to_{neighbor}"
        bob_protocol_name = f"TeleportedCNOT_{neighbor}_to_{self.node.name}"

        bob_protocol = TeleportedCNOTProtocol(
            owner=bob_node, name=bob_protocol_name, role='bob',
            remote_node_name=self.node.name, data_qubits=bob_data,
            communication_qubits=bob_comm
        )
        bob_node.protocols.append(bob_protocol)
        bob_app._tcnot_protocols[self.node.name] = bob_protocol

        alice_protocol = TeleportedCNOTProtocol(
            owner=self.node, name=alice_protocol_name, role='alice',
            remote_node_name=neighbor, data_qubits=alice_data,
            communication_qubits=alice_comm, bob_protocol=bob_protocol
        )
        self.node.protocols.append(alice_protocol)
        self._tcnot_protocols[neighbor] = alice_protocol
        self._mark_memories_touched(alice_data + alice_comm)
        bob_app._mark_memories_touched(bob_data + bob_comm)

        log.logger.info(f"{self.node.name}: TCNOT protocols created. "
                        f"Alice={alice_protocol_name}, Bob={bob_protocol_name}")
        alice_protocol.start()

    def _record_post_idle_pair_fidelities(self, neighbor, bob_app, alice_comm, bob_comm, shots: int = 4000):
        """Store per-pair wait diagnostics and post-idle physical Bell fidelities."""
        alice_initial = {
            row["pair_index"]: row for row in self._initial_pair_fidelity_rows.get(neighbor, [])
        }
        alice_waits = {
            row["pair_index"]: row for row in self._pre_encoding_pair_idle_rows.get(neighbor, [])
        }
        bob_waits = {
            row["pair_index"]: row for row in bob_app._pre_encoding_pair_idle_rows.get(self.node.name, [])
        }

        pair_rows = []
        for pair_index, (alice_mem, bob_mem) in enumerate(zip(alice_comm, bob_comm)):
            post_idle_fidelity = self._estimate_bell_pair_fidelity(
                alice_mem.qstate_key,
                bob_mem.qstate_key,
                shots=shots,
                seed_tag=f"post_idle_{neighbor}_{pair_index}",
            )
            a_wait = alice_waits.get(pair_index, {})
            b_wait = bob_waits.get(pair_index, {})
            init = alice_initial.get(pair_index, {})
            pair_rows.append(
                {
                    "pair_index": pair_index,
                    "alice_memory": alice_mem.name,
                    "bob_memory": bob_mem.name,
                    "alice_wait_time_sec": float(a_wait.get("wait_time_sec", float("nan"))),
                    "bob_wait_time_sec": float(b_wait.get("wait_time_sec", float("nan"))),
                    "alice_p_idle": float(a_wait.get("p_idle", float("nan"))),
                    "bob_p_idle": float(b_wait.get("p_idle", float("nan"))),
                    "initial_pair_fidelity": float(init.get("initial_pair_fidelity", float("nan"))),
                    "post_idle_pair_fidelity": float(post_idle_fidelity),
                }
            )

        self._post_idle_pair_fidelity_rows[neighbor] = pair_rows

    def _on_teleported_cnot_complete(self, neighbor):
        """Called when teleported CNOT protocol finishes for a link.

        Called on BOTH Alice's and Bob's apps. Each app tracks which
        of its links are complete. When a middle node has both links done:
        - Non-coordinator: sends LINKS_READY to the coordinator
        - Coordinator: checks if all wait_for nodes reported ready, then swaps
        """
        protocol = self._tcnot_protocols.get(neighbor)
        is_initiator = (protocol is not None and protocol.role == 'alice')

        log.logger.info(f"{self.node.name}: Teleported CNOT complete for "
                        f"link to {neighbor}")

        self._completed_link_neighbors.add(neighbor)

        if is_initiator:
            fidelity = self._calculate_logical_bell_pair_fidelity(neighbor)
            self._link_logical_fidelities[neighbor] = fidelity
            log.logger.info(f"{self.node.name}: Logical fidelity = {fidelity:.6f}")

        # Check if this middle node has both links done
        num_links = len(self.data_qubits)
        num_complete = len(self._completed_link_neighbors)
        role = self._swap_config.get("role") if self._swap_config else None

        if (num_links == 2 and num_complete == 2
                and not self._swapping_done
                and role in ("coordinator", "swap_node")):
            if role == "coordinator":
                # Coordinator: check if ready to trigger all swaps
                if self._can_swap():
                    self._swapping_done = True
                    log.logger.info(f"{self.node.name}: All nodes ready. "
                                    f"Triggering simultaneous swapping.")
                    process = Process(self, '_begin_simultaneous_swap', [])
                    event = Event(self.node.timeline.now() + 1000, process)
                    self.node.timeline.schedule(event)
                else:
                    log.logger.info(f"{self.node.name}: Both links complete but "
                                    f"waiting for other nodes to report ready.")
            else:
                # Non-coordinator swap node: notify the coordinator
                coordinator = self._swap_config["coordinator"]
                log.logger.info(f"{self.node.name}: Both links complete. "
                                f"Sending LINKS_READY to coordinator {coordinator}.")
                msg = SwapNotificationMessage(
                    msg_type=SwapMsgType.LINKS_READY,
                    receiver_name=f"RequestLogicalPairApp.{coordinator}",
                    sender_node=self.node.name
                )
                self.node.send_message(coordinator, msg)
        else:
            self._check_simulation_stop()

    # ------------------------------------------------------------------
    # Stage 4: Entanglement swapping
    # ------------------------------------------------------------------

    def _can_swap(self):
        """Check if this coordinator node is ready to trigger all swaps.

        Returns True if this node is the coordinator, its own links are
        complete, and all wait_for nodes have reported LINKS_READY.
        """
        if self._swapping_done or self._swap_config is None:
            return False
        if self._swap_config.get("role") != "coordinator":
            return False
        if len(self._completed_link_neighbors) < 2:
            return False
        wait_for = set(self._swap_config.get("wait_for", []))
        return wait_for.issubset(self._ready_nodes)

    def _on_node_links_ready(self, ready_node):
        """Handle notification that a non-coordinator node's links are ready.

        Called from received_message() when a LINKS_READY classical
        message arrives at the coordinator. Triggers simultaneous
        swapping if all nodes are ready.
        """
        self._ready_nodes.add(ready_node)

        log.logger.info(f"{self.node.name}: Node {ready_node} links ready. "
                        f"Ready nodes: {self._ready_nodes}")

        if self._can_swap():
            self._swapping_done = True
            log.logger.info(f"{self.node.name}: All nodes ready. "
                            f"Triggering simultaneous swapping.")
            process = Process(self, '_begin_simultaneous_swap', [])
            event = Event(self.node.timeline.now() + 1000, process)
            self.node.timeline.schedule(event)

    def _begin_simultaneous_swap(self):
        """Coordinator: merge circuits and schedule local Bell measurements.

        LOCC flow:
        1. Merge all link circuits into one Stim tableau (simulation bookkeeping)
        2. Schedule _perform_local_bell_measurement on each swap node's app
           with sequential timing to ensure deterministic measurement ordering

        Each middle node then independently appends gates on its OWN qubits
        and sends SWAP_COMPLETE to R0. R0 applies corrections on its own qubits.
        """
        swap_nodes = self._swap_config["swap_nodes"]

        log.logger.info(f"{self.node.name}: Starting simultaneous swapping. "
                        f"Swap nodes: {swap_nodes}")

        # Step 1: Merge ALL link circuits into one tableau (simulation bookkeeping)
        self._merge_all_link_circuits()

        # Step 2: Schedule local Bell measurements on each swap node
        # Sequential timing (t+1000, t+1100, ...) ensures measurements are
        # appended in swap_nodes order for correct target_rec indexing.
        for order, swap_name in enumerate(swap_nodes):
            swap_node = self.node.timeline.get_entity_by_name(swap_name)
            swap_app = swap_node.request_logical_pair_app
            process = Process(swap_app, '_perform_local_bell_measurement', [])
            event = Event(self.node.timeline.now() + 1000 + order * 100, process)
            self.node.timeline.schedule(event)

    def _perform_local_bell_measurement(self):
        """Perform Bell measurement on this node's own left/right data qubits.

        LOCC compliant: only appends gates on this node's local qubits.
        After measurement, sends SWAP_COMPLETE to the correction endpoint (R0).

        For self-dual codes (Steane): CX(left→right), H on all left, M all.
        For non-self-dual codes (Shor, cx_reversed=True):
          CX(right→left), no H on left, H on x_support of right, M all.
        """
        n = self.n
        neighbors = sorted(self.data_qubits.keys())
        self._apply_idle_decoherence_to_memories(
            self.data_qubits[neighbors[0]] + self.data_qubits[neighbors[1]],
            stage="pre_swap_bell_measurement",
        )
        left_keys = [m.qstate_key for m in self.data_qubits[neighbors[0]]]
        right_keys = [m.qstate_key for m in self.data_qubits[neighbors[1]]]

        circuit = self.node.timeline.quantum_manager.states[left_keys[0]].circuit

        if self.code.cx_reversed:
            # CX_L(left→right) requires physical CX(right→left) for reversed codes
            for i in range(n):
                circuit.append('CX', [right_keys[i], left_keys[i]])
            # X_L = Z on z_support → measure Z on left (no H needed)
            # Z_L = X on x_support → measure X on right x_support (H needed)
            for i in self.code.x_support:
                circuit.append('H', [right_keys[i]])
        else:
            # Standard: CX(left→right), H on x_support of left
            for i in range(n):
                circuit.append('CX', [left_keys[i], right_keys[i]])
            for i in self.code.x_support:
                circuit.append('H', [left_keys[i]])

        # Measure left block then right block
        for i in range(n):
            circuit.append('M', [left_keys[i]])
        for i in range(n):
            circuit.append('M', [right_keys[i]])

        log.logger.info(f"{self.node.name}: Local Bell measurement complete "
                        f"(2x{n} measurements appended)")
        self._mark_memories_touched(
            self.data_qubits[neighbors[0]] + self.data_qubits[neighbors[1]]
        )

        # Send classical notification to correction endpoint (R0)
        corr_node = self._swap_config["correction_node"]
        msg = SwapNotificationMessage(
            msg_type=SwapMsgType.SWAP_COMPLETE,
            receiver_name=f"RequestLogicalPairApp.{corr_node}",
            sender_node=self.node.name
        )
        self.node.send_message(corr_node, msg)

    def _on_swap_measurement_received(self, sender_node):
        """R0: track swap measurement notifications from middle nodes.

        Called from received_message() when SWAP_COMPLETE arrives.
        Once all middle nodes have reported, triggers correction application.
        """
        self._ready_nodes.add(sender_node)

        num_expected = self._swap_config["num_swap_nodes"]
        log.logger.info(f"{self.node.name}: Swap measurement received from {sender_node}. "
                        f"({len(self._ready_nodes)}/{num_expected})")

        if len(self._ready_nodes) >= num_expected:
            process = Process(self, '_apply_swap_corrections', [])
            event = Event(self.node.timeline.now() + 100, process)
            self.node.timeline.schedule(event)

    def _apply_swap_corrections(self):
        """R0: apply Pauli frame corrections on this endpoint's own data qubits.

        LOCC compliant: only appends gates on this node's local qubits.
        Uses target_rec references to the middle nodes' measurement results.
        After corrections, measures end-to-end fidelity and stops simulation.

        For self-dual codes (Steane):
          All left meas → CZ to z_support, all right meas → CX to x_support.
        For non-self-dual codes (Shor, cx_reversed=True):
          Left z_support meas → CX to x_support, right x_support meas → CZ to z_support.
        """
        n = self.n
        config = self._swap_config
        endpoint_keys = [m.qstate_key
                         for m in self.data_qubits[config["correction_key"]]]
        endpoint_memories = self.data_qubits[config["correction_key"]]
        self._apply_idle_decoherence_to_memories(endpoint_memories, stage="pre_swap_corrections")

        circuit = self.node.timeline.quantum_manager.states[endpoint_keys[0]].circuit

        num_swap_nodes = config["num_swap_nodes"]
        total_meas = num_swap_nodes * 2 * n

        for swap_idx in range(num_swap_nodes):
            x_start = -(total_meas - swap_idx * 2 * n)
            z_start = x_start + n

            if self.code.cx_reversed:
                # X_L meas from left z_support → Z_L correction on endpoint
                # Z_L = X on x_support → apply CX to endpoint x_support
                for k in self.code.z_support:
                    for j in self.code.x_support:
                        circuit.append('CX', [stim.target_rec(x_start + k),
                                              endpoint_keys[j]])
                # Z_L meas from right x_support → X_L correction on endpoint
                # X_L = Z on z_support → apply CZ to endpoint z_support
                for k in self.code.x_support:
                    for j in self.code.z_support:
                        circuit.append('CZ', [stim.target_rec(z_start + k),
                                              endpoint_keys[j]])
            else:
                # X_L meas from left x_support → Z_L correction (CZ to z_support)
                for k in self.code.x_support:
                    for j in self.code.z_support:
                        circuit.append('CZ', [stim.target_rec(x_start + k),
                                              endpoint_keys[j]])
                # Z_L meas from right z_support → X_L correction (CX to x_support)
                for k in self.code.z_support:
                    for j in self.code.x_support:
                        circuit.append('CX', [stim.target_rec(z_start + k),
                                              endpoint_keys[j]])

        self._swapping_done = True
        self._mark_memories_touched(endpoint_memories)

        log.logger.info(f"{self.node.name}: Swap corrections applied "
                        f"({num_swap_nodes} swaps, {total_meas} measurement refs)")

        # Measure end-to-end fidelity
        fidelity = self._calculate_chain_fidelity(
            config["left_endpoint"], config["left_endpoint_key"],
            config["right_endpoint"], config["right_endpoint_key"])
        self._final_end_to_end_fidelity = fidelity

        self._check_simulation_stop()

    # ------------------------------------------------------------------
    # Fidelity measurement (Pauli tomography)
    # ------------------------------------------------------------------

    def _calculate_chain_fidelity(self, left_node_name, left_key,
                                   right_node_name, right_key,
                                   shots: int = 10000):
        """Calculate logical Bell pair fidelity between two nodes via Pauli tomography.

        Measures the logical Bell state fidelity F = (1 + <X_L X_L> - <Y_L Y_L> + <Z_L Z_L>) / 4
        where X_L, Y_L, Z_L are the code's logical Pauli operators. Each logical
        operator acts on a subset of physical qubits (the "support"), so we:
          1. Apply per-qubit basis rotations only on the support
          2. Measure ALL physical qubits
          3. Compute parity only over the support qubits

        Used for both per-link fidelity (after TCNOT) and end-to-end fidelity
        (after swapping), with different left/right node arguments.

        Args:
            left_node_name: Name of left endpoint node
            left_key: Neighbor key into left node's data_qubits dict
            right_node_name: Name of right endpoint node
            right_key: Neighbor key into right node's data_qubits dict
            shots: Sampling shots per Pauli basis
        """
        qm = self.node.timeline.quantum_manager

        left_node = self.node.timeline.get_entity_by_name(left_node_name)
        left_app = left_node.request_logical_pair_app
        alice_keys = [mem.qstate_key for mem in left_app.data_qubits[left_key]]

        right_node = self.node.timeline.get_entity_by_name(right_node_name)
        right_app = right_node.request_logical_pair_app
        bob_keys = [mem.qstate_key for mem in right_app.data_qubits[right_key]]

        all_keys = alice_keys + bob_keys
        state = qm.states[alice_keys[0]]

        base_seed = hash(f"{left_node_name}_{right_node_name}_chain") % (2**31)
        y_info = self.code.y_basis_info()
        correlations = {}

        for basis in ['X', 'Y', 'Z']:
            meas_circuit = state.circuit.copy()

            # Basis rotation: only rotate qubits in the logical operator's support.
            # X basis: H on x_support qubits
            # Y basis: per-qubit rotation derived from X_L ∩ Z_L overlap
            # Z basis: no rotation needed (computational basis)
            if basis == 'X':
                for idx in self.code.x_support:
                    meas_circuit.append("H", [alice_keys[idx]])
                    meas_circuit.append("H", [bob_keys[idx]])
            elif basis == 'Y':
                for idx, pauli in y_info['basis_changes'].items():
                    for keys in [alice_keys, bob_keys]:
                        if pauli == 'Y':
                            meas_circuit.append("S_DAG", [keys[idx]])
                            meas_circuit.append("H", [keys[idx]])
                        elif pauli == 'X':
                            meas_circuit.append("H", [keys[idx]])

            # Measure all physical qubits (harmless to measure non-support qubits)
            for key in all_keys:
                meas_circuit.append("M", [key])

            basis_seed = abs(base_seed + hash(basis)) % (2**63)
            sampler = meas_circuit.compile_sampler(seed=basis_seed)
            measurements = sampler.sample(shots=shots)

            # Extract only the measurements we just appended (skip any prior M records)
            num_meas = measurements.shape[1]
            our = measurements[:, num_meas - len(all_keys):]

            # Compute logical parity over only the support qubits
            if basis == 'X':
                support = self.code.x_support
            elif basis == 'Z':
                support = self.code.z_support
            else:
                support = y_info['support']

            # Alice's qubits are columns 0..n-1, Bob's are columns n..2n-1
            a_parity = np.sum(our[:, support], axis=1) % 2
            b_parity = np.sum(our[:, [s + self.n for s in support]], axis=1) % 2

            # Convert to eigenvalues: 0 -> +1, 1 -> -1
            a_eig = 1 - 2 * a_parity
            b_eig = 1 - 2 * b_parity
            correlations[basis] = float(np.mean(a_eig * b_eig))

        xx, yy, zz = correlations['X'], correlations['Y'], correlations['Z']
        fidelity = (1 + xx - yy + zz) / 4

        log.logger.info(f"{self.node.name}: {left_node_name} <-> "
                        f"{right_node_name}: <XX>={xx:+.4f} <YY>={yy:+.4f} "
                        f"<ZZ>={zz:+.4f} F={fidelity:.6f}")

        return float(fidelity)

    def _calculate_logical_bell_pair_fidelity(self, neighbor,
                                               shots: int = 10000):
        """Calculate logical Bell pair fidelity for an adjacent link."""
        return self._calculate_chain_fidelity(
            self.node.name, neighbor, neighbor, self.node.name, shots=shots)

    # ------------------------------------------------------------------
    # Private utilities
    # ------------------------------------------------------------------

    def _get_neighbor(self, reservation):
        """Get the neighbor node name from a reservation."""
        if reservation.initiator == self.node.name:
            return reservation.responder
        return reservation.initiator

    def _allocate_qubits_for_link(self, neighbor_name):
        """Allocate the next n data qubits for a new link."""
        data_array_name = f"{self.node.name}.DataMemoryArray"
        data_array = self.node.components[data_array_name]

        if self._next_data_index + self.n > len(data_array.memories):
            raise RuntimeError(f"{self.node.name}: Not enough data qubits. "
                               f"Need {self.n} more, have "
                               f"{len(data_array.memories) - self._next_data_index}")

        self.data_qubits[neighbor_name] = data_array.memories[
            self._next_data_index : self._next_data_index + self.n
        ]

        log.logger.info(f"{self.node.name}: Allocated data qubits for link to {neighbor_name}: "
                        f"data[{self._next_data_index}:{self._next_data_index + self.n}]")
        self._next_data_index += self.n

    def _get_comm_qubits_for_link(self, reservation):
        """Derive communication qubits for a link from memo_to_reservation.

        Collects the actual Memory objects assigned to a given reservation,
        ensuring we use the correct Bell pair memories regardless of RSVP
        assignment order. Caches the result in self.comm_qubits.
        """
        neighbor = self._get_neighbor(reservation)
        comm_memories = []
        for mem_index, res in self.memo_to_reservation.items():
            if res is reservation:
                mem_info = self.node.resource_manager.memory_manager[mem_index]
                if mem_info.state == "ENTANGLED":
                    comm_memories.append(mem_info.memory)

        if len(comm_memories) != self.n:
            log.logger.warning(f"{self.node.name}: Expected {self.n} comm qubits "
                              f"for {neighbor}, found {len(comm_memories)}")

        self.comm_qubits[neighbor] = comm_memories
        return comm_memories

    def _get_entangled_memories(self, reservation):
        """Collect entangled MemoryInfo objects belonging to a reservation."""
        entangled = []
        for mem_index, res in self.memo_to_reservation.items():
            if res is reservation:
                mem_info = self.node.resource_manager.memory_manager[mem_index]
                if mem_info.state == "ENTANGLED":
                    entangled.append(mem_info)
        return entangled

    def _reorder_bob_comm_qubits(self, alice_comm, bob_comm_unordered):
        """Reorder Bob's comm qubits to match Alice's entanglement pairing.

        Alice's comm[i] is entangled with a specific Bob memory (from
        MemoryInfo.remote_memo). We must ensure bob_comm[i] is that
        exact memory so the transversal CX gates act on entangled pairs.
        """
        mm_alice = self.node.resource_manager.memory_manager
        bob_comm_by_name = {m.name: m for m in bob_comm_unordered}
        bob_comm = []
        for a_mem in alice_comm:
            a_info = None
            for info in mm_alice:
                if info.memory is a_mem:
                    a_info = info
                    break
            if a_info is None or a_info.remote_memo not in bob_comm_by_name:
                raise RuntimeError(
                    f"{self.node.name}: Cannot find Bob's entangled partner for {a_mem.name}")
            bob_comm.append(bob_comm_by_name[a_info.remote_memo])
        return bob_comm

    def _merge_all_link_circuits(self):
        """Merge ALL disjoint link Stim circuits across the entire chain.

        Collects every unique Stim circuit from every node's data qubits
        in the chain, merges them into a single stabilizer tableau using
        Stim's + operator, and updates all qubit state pointers.

        Returns:
            (merged_circuit, all_keys): the combined stim.Circuit and the
            sorted list of all qubit keys in the merged group.
        """
        qm = self.node.timeline.quantum_manager
        node_names = self._swap_config["chain_node_names"]

        # Collect unique circuits from all nodes' data qubits
        circuits_seen = {}  # id(circuit) -> StabilizerState
        for name in node_names:
            node = self.node.timeline.get_entity_by_name(name)
            app = node.request_logical_pair_app
            for neighbor in sorted(app.data_qubits.keys()):
                key = app.data_qubits[neighbor][0].qstate_key
                state = qm.states[key]
                if id(state.circuit) not in circuits_seen:
                    circuits_seen[id(state.circuit)] = state

        states = list(circuits_seen.values())
        if len(states) == 1:
            return states[0].circuit, sorted(states[0].keys)

        # Merge using Stim's + operator
        merged = states[0].circuit
        all_keys = set(states[0].keys)
        for s in states[1:]:
            merged = merged + s.circuit
            all_keys.update(s.keys)
        all_keys = sorted(all_keys)

        # Update quantum manager state pointers
        for key in all_keys:
            qm.states[key] = StabilizerState(
                original_key=key,
                keys=all_keys,
                circuit=merged,
                shots=qm.shots,
                truncation=qm.truncation,
                base_seed=qm.base_seed
            )

        log.logger.info(f"{self.node.name}: Merged {len(states)} link circuits. "
                        f"Combined {len(all_keys)} qubits into single tableau.")
        return merged, all_keys

    def _check_simulation_stop(self):
        """Stop the simulation if this node has completed its final action."""
        if not self._is_final_action_node:
            return
        if self._swap_config is not None and not self._swapping_done:
            return
        log.logger.info(f"{self.node.name}: Final action complete. Stopping simulation.")
        self.node.timeline.stop()
