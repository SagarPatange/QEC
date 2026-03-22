"""
Teleported CNOT Protocol — Transversal logical CNOT via Bell pairs.

Implements the teleportation-based CNOT gate between logical qubits
following PhysRevA.79.032325 Figure 3, Step 1(iii). Works with any
CSS code (parameterized by len(data_qubits)).

Protocol flow:
  Phase A: Alice applies CX(data, comm) + M(comm), sends Z-bits to Bob
  Phase B: Bob applies CX(comm, data) + X_corr + H(comm) + M(comm), sends X-bits to Alice
  Phase C: Alice applies Z corrections based on Bob's X-bits
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
    READY = auto()              # Local side is ready
    READY_ACK = auto()          # Remote acknowledges READY
    START = auto()              # Alice starts protocol execution
    ALICE_MEASUREMENT = auto()  # Alice -> Bob after Phase A
    BOB_MEASUREMENT = auto()    # Bob -> Alice after Phase B
    TCNOT_COMPLETE = auto()     # Alice -> Bob explicit completion notification


class TeleportedCNOTMessage(Message):
    """Classical message exchanged during the teleported CNOT protocol.

        Each message type stores its payload under a dedicated named attribute:

            READY            — No payload. Signals local qubits are allocated.
            READY_ACK        — No payload. Acknowledges peer's READY.
            START            — No payload. Alice signals Phase A is starting.
            ALICE_MEASUREMENT — self.alice_measurement_results (list[int]):
                                Alice's n Z-basis bits from Phase A, sent to Bob.
            BOB_MEASUREMENT  — self.bob_measurement_results (list[int]):
                                Bob's n X-basis bits from Phase B, sent to Alice.
        """

    def __init__(self, msg_type: TeleportedCNOTMsgType, protocol_name: str, **kwargs):
        super().__init__(msg_type, protocol_name)
        self.protocol_name = protocol_name

        if msg_type is TeleportedCNOTMsgType.READY:
            pass
        elif msg_type is TeleportedCNOTMsgType.READY_ACK:
            pass
        elif msg_type is TeleportedCNOTMsgType.START:
            pass
        elif msg_type is TeleportedCNOTMsgType.ALICE_MEASUREMENT:
            self.alice_measurement_results = list(kwargs['alice_measurement_results'])
        elif msg_type is TeleportedCNOTMsgType.BOB_MEASUREMENT:
            self.bob_measurement_results = list(kwargs['bob_measurement_results'])
        elif msg_type is TeleportedCNOTMsgType.TCNOT_COMPLETE:
            pass
        else:
            raise ValueError(f"TeleportedCNOTMessage created with unknown msg_type: {msg_type}")


class TeleportedCNOTProtocol(Protocol):
    """Protocol for performing a teleported CNOT between logical qubits on two nodes."""

    def __init__(self, owner, name: str, role: str, remote_node_name: str, data_qubits, communication_qubits, bob_protocol=None):
        """Initialize a TeleportedCNOT protocol instance.

        Args:
            owner: Quantum router node.
            name: Protocol identifier (TeleportedCNOT_{owner}_to_{remote}).
            role: 'alice' or 'bob'.
            remote_node_name: Partner node name.
            data_qubits: List of n Memory objects (encoded logical qubits).
            communication_qubits: List of n Memory objects (Bell pair qubits).
            bob_protocol: Bob's protocol instance (Alice only, simulation workaround).

        Returns:
            None
        """
        super().__init__(owner, name)

        assert role in ('alice', 'bob'), f"Role must be 'alice' or 'bob', got {role}"

        self.protocol_type = "TeleportedCNOTProtocol"
        self.role = role
        self.remote_node_name = remote_node_name
        self.n = len(data_qubits)
        self.data_qubits = data_qubits
        self.communication_qubits = communication_qubits
        self.bob_protocol = bob_protocol

        assert len(communication_qubits) == self.n, \
            f"Need {self.n} comm qubits, got {len(communication_qubits)}"

        self.current_phase = 'IDLE'
        self.start_time = None
        self.end_time = None
        self.started = False
        self.local_ready = False
        self.remote_ready = False

    def start(self) -> None:
        """Enter handshake mode and wait for both sides to become ready.

        Args:
            None

        Returns:
            None
        """
        if self.started:
            raise RuntimeError(f"{self.name}: start called more than once")

        self.started = True
        self.current_phase = "HANDSHAKE"

        has_data = len(self.data_qubits) == self.n
        has_comm = len(self.communication_qubits) == self.n
        self.local_ready = has_data and has_comm

        if not self.local_ready:
            return

        self._send_message(TeleportedCNOTMsgType.READY)

        if self.role == "bob":
            self.current_phase = "WAITING_START"
            return

        if self.remote_ready:
            self._send_message(TeleportedCNOTMsgType.START)
            process = Process(self, "_alice_phase_a", [])
            event = Event(self.owner.timeline.now() + 50, process)
            self.owner.timeline.schedule(event)
            self.current_phase = "WAITING_FOR_BOB"
        else:
            self.current_phase = "WAITING_READY"

    def is_ready(self) -> bool:
        """Check if the protocol is ready to execute.

        Args:
            None

        Returns:
            bool: Always True.
        """
        return True

    def received_message(self, src: str, msg: TeleportedCNOTMessage) -> None:
        """Handle incoming classical messages.

        Args:
            src: Source node name.
            msg: Incoming teleported-CNOT message.

        Returns:
            None
        """
        if msg.msg_type == TeleportedCNOTMsgType.READY:
            self.remote_ready = True  # peer reports local resources are ready

            # Explicitly acknowledge peer readiness so handshake flow is visible in logs/messages.
            self._send_message(TeleportedCNOTMsgType.READY_ACK)
            return

        if msg.msg_type == TeleportedCNOTMsgType.READY_ACK:
            # Only Alice starts execution after receiving READY_ACK and confirming local readiness.
            if self.role == "alice" and self.started and self.local_ready and self.current_phase in ("HANDSHAKE", "WAITING_READY"):
                self._send_message(TeleportedCNOTMsgType.START)  # tell Bob execution begins now

                # Schedule Phase A on timeline instead of executing inline.
                process = Process(self, "_alice_phase_a", [])
                event = Event(self.owner.timeline.now() + 50, process)
                self.owner.timeline.schedule(event)
                self.current_phase = "WAITING_FOR_BOB"
            return
        
        if msg.msg_type == TeleportedCNOTMsgType.START:
            # START is only valid on Bob side; Alice never consumes START.
            if self.role != "bob":
                return

            # Bob now waits for Alice's Phase-A measurement message.
            self.current_phase = "WAITING_FOR_ALICE_MEASUREMENT"
            return

        if msg.msg_type == TeleportedCNOTMsgType.ALICE_MEASUREMENT:
            if self.role != "bob":
                return

            self.alice_measurement_results = list(msg.alice_measurement_results)
            process = Process(self, "_bob_phase_b", [])
            event = Event(self.owner.timeline.now() + 50, process)
            self.owner.timeline.schedule(event)
            return

        if msg.msg_type == TeleportedCNOTMsgType.BOB_MEASUREMENT:
            # Only Alice handles Bob's measurement message.
            if self.role != "alice":
                return

            # Schedule Alice Phase C after a small protocol delay.
            self.bob_measurement_results = list(msg.bob_measurement_results)
            process = Process(self, "_alice_phase_c", [])
            event = Event(self.owner.timeline.now() + 50, process)
            self.owner.timeline.schedule(event)
            return

        if msg.msg_type == TeleportedCNOTMsgType.TCNOT_COMPLETE:
            # Only Bob consumes explicit completion from Alice.
            if self.role != "bob":
                return

            process = Process(
                self.owner.request_logical_pair_app,
                "_on_teleported_cnot_complete",
                [src],
            )
            event = Event(self.owner.timeline.now() + 50, process)
            self.owner.timeline.schedule(event)
            return

        return

    def _alice_phase_a(self):
        """Alice: transversal CX(data -> comm), measure comm in Z basis, notify Bob.

        Uses the tableau quantum manager: run_circuit() applies gates and
        measures immediately, returning real classical bits sent to Bob.

        Args:
            None

        Returns:
            None
        """
        self.start_time = self.owner.timeline.now()
        self.current_phase = 'PHASE_A'
        if self.bob_protocol is None:
            raise ValueError("bob_protocol not provided to Alice")

        n = self.n
        qm = self.owner.timeline.quantum_manager

        circ = Circuit(2 * n)
        for i in range(n):
            circ.cx(i, n + i)      # CX(data_i, comm_i)
        for i in range(n):
            circ.measure(n + i)    # M(comm_i) — Z basis

        # Key order must match circuit indices: [data_0..data_n-1, comm_0..comm_n-1].
        keys = ([self.data_qubits[i].qstate_key for i in range(n)]
            + [self.communication_qubits[i].qstate_key for i in range(n)])

        rnd = self.owner.get_generator().random()
        results = qm.run_circuit(circ, keys, rnd)

        # Extract Z-basis bits in qubit order from run_circuit results
        self.alice_measurement_results = [results[self.communication_qubits[i].qstate_key] for i in range(n)]
        log.logger.info(f"[{self.name}] Phase A: {n} CX + {n} M, z_bits={self.alice_measurement_results}")

        self._send_message(TeleportedCNOTMsgType.ALICE_MEASUREMENT, alice_measurement_results=self.alice_measurement_results)
        self.current_phase = 'WAITING_FOR_BOB'

    def _bob_phase_b(self):
        """Bob: CX(comm -> data), X correction from Alice's bits, H(comm), measure comm.

        Args:
            None

        Returns:
            None
        """
        self.current_phase = 'PHASE_B'
        n = self.n
        qm = self.owner.timeline.quantum_manager

        # CX(comm, data) + H(comm) + M(comm)
        circ = Circuit(2 * n)
        for i in range(n):
            circ.cx(i, n + i)      # CX(comm_i, data_i)
        for i in range(n):
            circ.h(i)              # H(comm_i)
        for i in range(n):
            circ.measure(i)        # M(comm_i) — X basis

        # Apply X corrections from Alice's Z-basis measurements
        for i in range(n):
            if self.alice_measurement_results[i]:
                corr = Circuit(1)
                corr.x(0)
                qm.run_circuit(corr, [self.data_qubits[i].qstate_key])

        # Key order must match circuit indices: [comm_0..comm_n-1, data_0..data_n-1].
        keys = ([self.communication_qubits[i].qstate_key for i in range(n)]
            + [self.data_qubits[i].qstate_key for i in range(n)])

        rnd = self.owner.get_generator().random()
        results = qm.run_circuit(circ, keys, rnd)

        self.bob_measurement_results = [results[self.communication_qubits[i].qstate_key] for i in range(n)]
        log.logger.info(f"[{self.name}] Phase B: {n} CX + {n} X_corr + {n} H + {n} M, x_bits={self.bob_measurement_results}")

        self._send_message(TeleportedCNOTMsgType.BOB_MEASUREMENT, bob_measurement_results=self.bob_measurement_results)
        self.current_phase = 'WAITING_FOR_ALICE'

    def _alice_phase_c(self):
        """Alice: Z corrections on data qubits conditioned on Bob's X-basis bits.

        Args:
            None

        Returns:
            None
        """
        self.current_phase = 'PHASE_C'
        n = self.n
        qm = self.owner.timeline.quantum_manager

        for i in range(n):
            if self.bob_measurement_results[i]:
                corr = Circuit(1)
                corr.z(0)
                qm.run_circuit(corr, [self.data_qubits[i].qstate_key])

        log.logger.info(f"[{self.name}] Phase C: {n} Z corrections applied")
        self._protocol_complete()

    def _send_message(self, msg_type: TeleportedCNOTMsgType, **kwargs) -> None:
        """Send a classical message to the remote node's protocol instance.

        Args:
            msg_type: Type of message to send.
            **kwargs: Payload fields forwarded to TeleportedCNOTMessage.

        Returns:
            None
        """
        msg = TeleportedCNOTMessage(msg_type, protocol_name=self.name, **kwargs)
        msg.receiver = f"TeleportedCNOT_{self.remote_node_name}_to_{self.owner.name}"
        self.owner.send_message(self.remote_node_name, msg)

    def _protocol_complete(self) -> None:
        """Notify both apps that the teleported CNOT is done.

        Args:
            None

        Returns:
            None
        """
        self.end_time = self.owner.timeline.now()
        self.current_phase = 'COMPLETE'

        log.logger.info(f"[{self.name}] Teleported CNOT complete "
                        f"(duration: {self.end_time - self.start_time:,} ps)")

        # Notify this node's app
        process = Process(self.owner.request_logical_pair_app,'_on_teleported_cnot_complete', [self.remote_node_name])
        event = Event(self.owner.timeline.now() + 1000, process)
        self.owner.timeline.schedule(event)

        # Explicitly notify Bob that this link finished on Alice's side.
        if self.role == 'alice':
            self._send_message(TeleportedCNOTMsgType.TCNOT_COMPLETE)
