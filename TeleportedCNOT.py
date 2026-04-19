"""Teleported CNOT protocol for one adjacent link.

This module implements a two-node control flow that applies a logical
teleported CNOT between encoded data blocks using pre-generated Bell pairs
on communication qubits.

Flow:
1. Phase A (Alice): CX(data->comm), measure comm in Z, send bits to Bob.
2. Phase B (Bob): CX(comm->data), apply X corrections, H(comm), measure comm
   in X, send bits to Alice.
3. Phase C (Alice): apply Z corrections from Bob's bits.
4. Completion: local app callback plus explicit TCNOT_COMPLETE to Bob.
"""

from enum import Enum, auto
from sequence.protocol import Protocol
from sequence.message import Message
from sequence.utils import log
from sequence.kernel.process import Process
from sequence.kernel.event import Event
from sequence.components.circuit import Circuit


class TeleportedCNOTMsgType(Enum):
    """Types of classical messages exchanged during the teleported CNOT protocol."""
    ALICE_MEASUREMENT = auto()  # Alice -> Bob after Phase A
    BOB_MEASUREMENT = auto()    # Bob -> Alice after Phase B
    TCNOT_COMPLETE = auto()     # Alice -> Bob explicit completion notification


class TeleportedCNOTMessage(Message):
    """Classical message container for teleported-CNOT control and payload bits.

    Message payloads by type:
    - ALICE_MEASUREMENT: alice_measurement_results (list[int]).
    - BOB_MEASUREMENT: bob_measurement_results (list[int]).
    - TCNOT_COMPLETE: no payload.
    """

    def __init__(self, msg_type: TeleportedCNOTMsgType, protocol_name: str, **kwargs):
        """Build one teleported-CNOT message.

        Args:
            msg_type: Message type enum value.
            protocol_name: Sender protocol name.
            **kwargs: Per-message payload fields.

        Returns:
            None
        """
        super().__init__(msg_type, "request_logical_pair_app")
        self.protocol_name = protocol_name

        if msg_type is TeleportedCNOTMsgType.ALICE_MEASUREMENT:
            self.alice_measurement_results = list(kwargs["alice_measurement_results"])
        elif msg_type is TeleportedCNOTMsgType.BOB_MEASUREMENT:
            self.bob_measurement_results = list(kwargs["bob_measurement_results"])
        elif msg_type is TeleportedCNOTMsgType.TCNOT_COMPLETE:
            pass
        else:
            raise ValueError(f"TeleportedCNOTMessage created with unknown msg_type: {msg_type}")


class TeleportedCNOTProtocol(Protocol):
    """Two-node teleported logical CNOT protocol for one adjacent link."""

    def __init__(self, owner, name: str, role: str, remote_node_name: str, data_qubit_keys: list[int], communication_qubit_keys: list[int]):
        """Initialize a TeleportedCNOT protocol instance.

        Args:
            owner: Quantum router node.
            name: Protocol identifier (TeleportedCNOT_{owner}_to_{remote}).
            role: 'alice' or 'bob'.
            remote_node_name: Partner node name.
            data_qubit_keys: List of n local data qstate keys.
            communication_qubit_keys: List of n local communication qstate keys.

        Returns:
            None
        """
        super().__init__(owner, name)

        if role not in ("alice", "bob"):
            raise RuntimeError(f"{self.name}: invalid role {role}")

        self.role = role
        self.remote_node_name = remote_node_name
        self.data_qubit_keys = list(data_qubit_keys)
        self.communication_qubit_keys = list(communication_qubit_keys)
        self.n = len(self.data_qubit_keys)

        if len(self.communication_qubit_keys) != self.n:
            raise RuntimeError(f"{self.name}: need {self.n} comm qubits, got {len(self.communication_qubit_keys)}")

        self.current_phase = "IDLE"
        self.start_time = None
        self.end_time = None
        self.started = False
        # Cache app handle once so protocol code has one anchor point.
        self.app = owner.request_logical_pair_app
        # Cache idle-noise settings from app at init; phase methods use local fields only.
        self.idle_pauli_weights: dict[str, float] = dict(self.app.idle_pauli_weights)
        self.idle_t1_sec = float(self.app.idle_t1_sec)
        self.idle_t2_sec = float(self.app.idle_t2_sec)

    def start(self) -> None:
        """Start teleported-CNOT execution once local resources are ready.

        Args:
            None

        Returns:
            None
        """
        if self.started:
            raise RuntimeError(f"{self.name}: start called more than once")

        if len(self.data_qubit_keys) != self.n or len(self.communication_qubit_keys) != self.n:
            raise RuntimeError(f"{self.name}: missing qubits for start")

        self.started = True
        if self.role == "alice":
            self.current_phase = "PHASE_A_SCHEDULED"
            process = Process(self, "alice_phase_a", [])
            event = Event(self.owner.timeline.now(), process, self.owner.timeline.schedule_counter)
            self.owner.timeline.schedule(event)
            log.logger.info(f"[{self.name}] start role=alice n={self.n} -> scheduled alice_phase_a")
            return

        self.current_phase = "WAITING_FOR_ALICE_MEASUREMENT"
        log.logger.info(f"[{self.name}] start role=bob n={self.n} phase={self.current_phase}")

    def received_message(self, src: str, msg: TeleportedCNOTMessage) -> None:
        """Dispatch incoming teleported-CNOT messages by type.

        Args:
            src: Source node name.
            msg: Incoming teleported-CNOT message.

        Returns:
            None
        """
        if not isinstance(msg, TeleportedCNOTMessage):
            return
        log.logger.debug(f"[{self.name}] recv src={src} type={msg.msg_type} phase={self.current_phase} role={self.role}")

        # Execution messages: ALICE_MEASUREMENT / BOB_MEASUREMENT.
        if msg.msg_type == TeleportedCNOTMsgType.ALICE_MEASUREMENT:
            if self.role != "bob":
                return

            # Save Alice's Z-basis outcomes, then schedule Bob's Phase B.
            self.alice_measurement_results = list(msg.alice_measurement_results)
            log.logger.info(f"[{self.name}] got ALICE_MEASUREMENT bits={len(self.alice_measurement_results)}")
            process = Process(self, "bob_phase_b", [])
            event = Event(self.owner.timeline.now(), process)
            self.owner.timeline.schedule(event)
            return

        if msg.msg_type == TeleportedCNOTMsgType.BOB_MEASUREMENT:
            if self.role != "alice":
                return

            # Save Bob's X-basis outcomes, then schedule Alice's Phase C.
            self.bob_measurement_results = list(msg.bob_measurement_results)
            log.logger.info(f"[{self.name}] got BOB_MEASUREMENT bits={len(self.bob_measurement_results)}")
            process = Process(self, "alice_phase_c", [])
            event = Event(self.owner.timeline.now(), process)
            self.owner.timeline.schedule(event)
            return

        # Completion message: TCNOT_COMPLETE.
        if msg.msg_type == TeleportedCNOTMsgType.TCNOT_COMPLETE:
            if self.role != "bob":
                return

            # Bob marks local completion and forwards completion to its app.
            self.end_time = self.owner.timeline.now()
            self.current_phase = "COMPLETE"
            log.logger.info(f"[{self.name}] TCNOT_COMPLETE received from {src}")

            process = Process(self.app, "on_teleported_cnot_complete", [src])
            event = Event(self.owner.timeline.now(), process)
            self.owner.timeline.schedule(event)
            return

        return

# -------- Phase implementations --------
    def alice_phase_a(self):
        """Execute Alice Phase A and send Z-basis communication bits to Bob.

        Args:
            None

        Returns:
            None
        """
        # Record protocol start on first quantum-action phase.
        self.start_time = self.owner.timeline.now()
        self.current_phase = 'PHASE_A'

        n = self.n
        qm = self.owner.timeline.quantum_manager

        # Phase A circuit: transversal CX(data_i -> comm_i), then measure comm in Z.
        circ = Circuit(2 * n)
        for i in range(n):
            circ.cx(i, n + i)      # CX(data_i, comm_i)
        for i in range(n):
            circ.measure(n + i)    # M(comm_i) — Z basis

        # Key order must match circuit indices: [data_0..data_n-1, comm_0..comm_n-1].
        keys = self.data_qubit_keys + self.communication_qubit_keys

        # Apply time-based idle decoherence before Phase A consumes data/comm qubits.
        now_ps = int(self.owner.timeline.now())
        qm.apply_idling_decoherence(keys=keys, now_ps=now_ps, t1_sec=self.idle_t1_sec, t2_sec=self.idle_t2_sec)

        rnd = self.owner.get_generator().random()
        results = qm.run_circuit(circ, keys, rnd)
        finish_t = int(self.owner.timeline.now()) + qm.get_circuit_duration(circ)
        log.logger.info(f"[{self.name}] phase_a_timing now={int(self.owner.timeline.now())} duration_ps={qm.get_circuit_duration(circ)} finish_t={finish_t}")
        for key in keys:
            qm.last_idle_time_ps_by_key[key] = finish_t

        # Extract Z-basis bits in qubit order from run_circuit results
        self.alice_measurement_results = [results[key] for key in self.communication_qubit_keys]
        log.logger.info(f"[{self.name}] Phase A: {n} CX + {n} M, z_bits={self.alice_measurement_results}")

        msg = TeleportedCNOTMessage(
            TeleportedCNOTMsgType.ALICE_MEASUREMENT,
            protocol_name=self.name,
            alice_measurement_results=self.alice_measurement_results)

        process = Process(self.owner, "send_message", [self.remote_node_name, msg])
        event = Event(finish_t, process, self.owner.timeline.schedule_counter)
        self.owner.timeline.schedule(event)
        self.current_phase = 'WAITING_FOR_BOB'

    def bob_phase_b(self):
        """Execute Bob Phase B and send X-basis communication bits to Alice.

        Args:
            None

        Returns:
            None
        """
        self.current_phase = "PHASE_B"
        n = self.n
        qm = self.owner.timeline.quantum_manager

        if len(self.alice_measurement_results) != n:
            raise RuntimeError(f"{self.name}: expected {n} Alice bits, got {len(self.alice_measurement_results)}")

        # Single circuit for entire Bob phase.
        circ = Circuit(2 * n)
        for i in range(n):
            circ.cx(i, n + i)  # CX(comm_i, data_i)

        for i in range(n):
            if int(self.alice_measurement_results[i]) == 1:
                circ.x(n + i)  # feed-forward X on data_i

        for i in range(n):
            circ.h(i)          # H(comm_i)
            circ.measure(i)    # M(comm_i) in X basis

        keys = self.communication_qubit_keys + self.data_qubit_keys
        # Apply time-based idle decoherence before Phase B consumes comm/data qubits.
        now_ps = int(self.owner.timeline.now())
        qm.apply_idling_decoherence(keys=keys, now_ps=now_ps, t1_sec=self.idle_t1_sec, t2_sec=self.idle_t2_sec)

        rnd = self.owner.get_generator().random()
        results = qm.run_circuit(circ, keys, rnd)
        # Delay Bob's reply by the estimated local circuit processing time.
        finish_t = int(self.owner.timeline.now()) + qm.get_circuit_duration(circ)
        # Mark the comm/data keys busy until the local circuit is considered complete.
        for key in keys:
            qm.last_idle_time_ps_by_key[key] = finish_t

        self.bob_measurement_results = [int(results[key]) for key in self.communication_qubit_keys]
        log.logger.info(f"[{self.name}] Phase B: single-circuit run, x_bits={self.bob_measurement_results}")

        msg = TeleportedCNOTMessage(
            TeleportedCNOTMsgType.BOB_MEASUREMENT,
            protocol_name=self.name,
            bob_measurement_results=self.bob_measurement_results)

        process = Process(self.owner, "send_message", [self.remote_node_name, msg])
        event = Event(finish_t, process, self.owner.timeline.schedule_counter)
        self.owner.timeline.schedule(event)
        self.current_phase = "WAITING_FOR_ALICE"

    def alice_phase_c(self):
        """Execute Alice Phase C and finalize teleported-CNOT execution.

        Args:
            None

        Returns:
            None
        """
        self.current_phase = "PHASE_C"
        n = self.n
        qm = self.owner.timeline.quantum_manager

        if len(self.bob_measurement_results) != n:
            raise RuntimeError(f"{self.name}: expected {n} Bob bits, got {len(self.bob_measurement_results)}")

        # Single circuit for entire Alice correction phase.
        correction_circuit = Circuit(n)
        for i in range(n):
            if int(self.bob_measurement_results[i]) == 1:
                correction_circuit.z(i)

        # Apply time-based idle decoherence before Phase C consumes data qubits.
        now_ps = int(self.owner.timeline.now())
        qm.apply_idling_decoherence(keys=self.data_qubit_keys, now_ps=now_ps, t1_sec=self.idle_t1_sec, t2_sec=self.idle_t2_sec)

        rnd = self.owner.get_generator().random()
        qm.run_circuit(correction_circuit, self.data_qubit_keys, rnd)
        # Delay local completion by the estimated correction-circuit processing time.
        finish_t = int(self.owner.timeline.now()) + qm.get_circuit_duration(correction_circuit)
        # Mark the data block busy until the local correction is considered complete.
        for key in self.data_qubit_keys:
            qm.last_idle_time_ps_by_key[key] = finish_t

        log.logger.info(f"[{self.name}] Phase C: single-circuit Z correction run")
        process = Process(self, "protocol_complete", [])
        event = Event(finish_t, process, self.owner.timeline.schedule_counter)
        self.owner.timeline.schedule(event)

    def protocol_complete(self) -> None:
        """Mark protocol complete and notify local app and remote Bob.

        Args:
            None

        Returns:
            None
        """
        self.end_time = self.owner.timeline.now()
        self.current_phase = 'COMPLETE'

        log.logger.info(f"[{self.name}] Teleported CNOT complete (duration: {self.end_time - self.start_time:,} ps)")

        # Local app callback for this node's link bookkeeping.
        process = Process(self.app, "on_teleported_cnot_complete", [self.remote_node_name])
        event = Event(self.owner.timeline.now() + 1000, process)
        self.owner.timeline.schedule(event)

        # Alice also notifies Bob so Bob can mark completion on its side.
        if self.role == 'alice':
            msg = TeleportedCNOTMessage(TeleportedCNOTMsgType.TCNOT_COMPLETE, protocol_name=self.name)
            self.owner.send_message(self.remote_node_name, msg)
