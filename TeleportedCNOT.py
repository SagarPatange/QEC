"""
Teleported CNOT Protocol — Transversal logical CNOT via Bell pairs.

Implements the teleportation-based CNOT gate between logical qubits
following PhysRevA.79.032325 Figure 3, Step 1(iii). Works with any
CSS code (parameterized by len(data_qubits)).

Protocol flow:
  Phase A: Alice applies CX(data, comm) + M(comm) — local operations
  Phase B: Classical message Alice -> Bob (measurement notification)
  Phase C: Bob applies CX(comm, data) + X_corr + H(comm) + M(comm)
  Phase D: Classical message Bob -> Alice (measurement notification)
  Phase E: Alice applies Z corrections — local operations
"""

import stim
from enum import Enum, auto
from sequence.protocol import Protocol
from sequence.message import Message
from sequence.utils import log
from sequence.kernel.process import Process
from sequence.kernel.event import Event


# ======================================================================
# Classical message types
# ======================================================================

class TeleportedCNOTMsgType(Enum):
    ALICE_MEASUREMENT = auto()
    BOB_MEASUREMENT = auto()


class TeleportedCNOTMessage(Message):
    """Classical message exchanged during the teleported CNOT protocol."""

    def __init__(self, msg_type: TeleportedCNOTMsgType, protocol_name: str):
        super().__init__(msg_type, protocol_name)
        self.protocol_name = protocol_name


# ======================================================================
# Main protocol
# ======================================================================

class TeleportedCNOTProtocol(Protocol):
    """Teleportation-based transversal CNOT gate across n physical qubits.

    Asymmetric roles:
    - Alice (initiator): Control qubit owner, starts protocol
    - Bob (responder): Target qubit owner, reacts to messages
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, owner, name: str, role: str,
                 remote_node_name: str, data_qubits, communication_qubits,
                 bob_protocol=None):
        """
        Args:
            owner: Quantum router node
            name: Protocol identifier (TeleportedCNOT_{owner}_to_{remote})
            role: 'alice' or 'bob'
            remote_node_name: Partner node name
            data_qubits: List of n Memory objects (encoded logical qubits)
            communication_qubits: List of n Memory objects (Bell pair qubits)
            bob_protocol: Bob's protocol instance (Alice only, simulation workaround)
        """
        super().__init__(owner, name)

        assert role in ('alice', 'bob'), f"Role must be 'alice' or 'bob', got {role}"

        self.protocol_type = "TeleportedCNOTProtocol"
        self.role = role
        self.remote_node_name = remote_node_name
        self.num_physical_qubits = len(data_qubits)
        self.data_qubits = data_qubits
        self.communication_qubits = communication_qubits
        self.bob_protocol = bob_protocol

        assert len(communication_qubits) == self.num_physical_qubits, \
            f"Need {self.num_physical_qubits} comm qubits, got {len(communication_qubits)}"

        self.current_phase = 'IDLE'
        self.start_time = None
        self.end_time = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self):
        """Start the protocol (Protocol interface)."""
        if self.role == 'alice':
            self._alice_phase_a()

    def is_ready(self) -> bool:
        return True

    def received_message(self, src: str, msg: TeleportedCNOTMessage):
        """Handle incoming classical messages.

        Dispatched by SeQUeNCe's node.receive_message() when
        msg.receiver matches this protocol's name.
        """
        if self.role == 'bob':
            assert msg.msg_type == TeleportedCNOTMsgType.ALICE_MEASUREMENT
            process = Process(self, "_bob_phase_c", [])
            event = Event(self.owner.timeline.now() + 50, process)
            self.owner.timeline.schedule(event)
            return True

        elif self.role == 'alice':
            assert msg.msg_type == TeleportedCNOTMsgType.BOB_MEASUREMENT
            process = Process(self, "_alice_phase_e", [])
            event = Event(self.owner.timeline.now() + 50, process)
            self.owner.timeline.schedule(event)
            return True

    # ------------------------------------------------------------------
    # Phase A: Alice's local operations + Z measurements
    # ------------------------------------------------------------------

    def _alice_phase_a(self):
        """Alice: transversal CX(data -> comm), measure comm in Z basis, notify Bob.

        After this phase, Alice's measurement results are encoded in the Stim
        circuit's measurement record. Bob will reference them via target_rec()
        to apply classical feed-forward corrections in Phase C.
        """
        self.start_time = self.owner.timeline.now()
        self.current_phase = 'PHASE_A'

        if self.bob_protocol is None:
            raise ValueError("bob_protocol not provided to Alice")

        # Simulation workaround: Stim requires all entangled qubits in the
        # same stabilizer tableau. group_qubits() merges disjoint circuits
        # into one — purely simulator bookkeeping with no physical analog.
        n = self.num_physical_qubits
        bob = self.bob_protocol

        all_keys = (
            [self.data_qubits[i].qstate_key for i in range(n)]
            + [self.communication_qubits[i].qstate_key for i in range(n)]
            + [bob.communication_qubits[i].qstate_key for i in range(n)]
            + [bob.data_qubits[i].qstate_key for i in range(n)]
        )

        self.owner.timeline.quantum_manager.group_qubits(all_keys)

        # --- LOCC-compliant operations below ---
        circuit = self.owner.timeline.quantum_manager.states[all_keys[0]].circuit

        # Transversal CX: correlate each data qubit with its comm partner
        for i in range(n):
            circuit.append('CX', [self.data_qubits[i].qstate_key,
                                   self.communication_qubits[i].qstate_key])

        # Z-basis measurement of all comm qubits (results enter measurement record)
        for i in range(n):
            circuit.append('M', [self.communication_qubits[i].qstate_key])

        log.logger.info(f"[{self.name}] Phase A: {n} CX + {n} M appended")

        # Notify Bob
        self._send_message(TeleportedCNOTMsgType.ALICE_MEASUREMENT)
        self.current_phase = 'WAITING_FOR_BOB'

    # ------------------------------------------------------------------
    # Phase C: Bob's operations + X measurements
    # ------------------------------------------------------------------

    def _bob_phase_c(self):
        """Bob: CX(comm -> data), X correction, H(comm), measure comm in X basis.

        For each physical qubit i:
          1. CX(comm_i, data_i)  — transfer Bell pair entanglement to data
          2. CX(rec[-n], data_i) — feed-forward X correction using Alice's
             Z-measurement result from Phase A. target_rec(-n) always refers
             to Alice's i-th measurement because we append M one at a time.
          3. H(comm_i)           — rotate comm to X basis
          4. M(comm_i)           — X-basis measurement for Alice's Phase E
        """
        self.current_phase = 'PHASE_C'
        n = self.num_physical_qubits

        circuit = self.owner.timeline.quantum_manager.states[
            self.data_qubits[0].qstate_key].circuit

        for i in range(n):
            data_key = self.data_qubits[i].qstate_key
            comm_key = self.communication_qubits[i].qstate_key

            circuit.append('CX', [comm_key, data_key])
            # Feed-forward: apply X to data conditioned on Alice's measurement
            circuit.append('CX', [stim.target_rec(-n), data_key])
            circuit.append('H', [comm_key])
            circuit.append('M', [comm_key])

        log.logger.info(f"[{self.name}] Phase C: {n} (CX + X_corr + H + M) appended")

        # Notify Alice
        self._send_message(TeleportedCNOTMsgType.BOB_MEASUREMENT)
        self.current_phase = 'WAITING_FOR_ALICE'

    # ------------------------------------------------------------------
    # Phase E: Alice's final Z corrections
    # ------------------------------------------------------------------

    def _alice_phase_e(self):
        """Alice: Z corrections on data qubits conditioned on Bob's X measurements.

        target_rec(i - n) references Bob's i-th measurement from Phase C.
        CZ applies Z to data_i when Bob's result is 1 (Pauli frame update).
        """
        self.current_phase = 'PHASE_E'
        n = self.num_physical_qubits

        circuit = self.owner.timeline.quantum_manager.states[
            self.data_qubits[0].qstate_key].circuit

        # Feed-forward: apply Z to data_i conditioned on Bob's measurement
        for i in range(n):
            circuit.append('CZ', [stim.target_rec(i - n),
                                   self.data_qubits[i].qstate_key])

        log.logger.info(f"[{self.name}] Phase E: {n} CZ corrections appended")
        self._protocol_complete()

    # ------------------------------------------------------------------
    # Private utilities
    # ------------------------------------------------------------------

    def _send_message(self, msg_type: TeleportedCNOTMsgType):
        """Send a classical message to the remote node's protocol instance."""
        msg = TeleportedCNOTMessage(msg_type, protocol_name=self.name)
        msg.receiver = f"TeleportedCNOT_{self.remote_node_name}_to_{self.owner.name}"
        self.owner.send_message(self.remote_node_name, msg)

    def _protocol_complete(self):
        """Notify both apps that the teleported CNOT is done."""
        self.end_time = self.owner.timeline.now()
        self.current_phase = 'COMPLETE'

        log.logger.info(f"[{self.name}] Teleported CNOT complete "
                        f"(duration: {self.end_time - self.start_time:,} ps)")

        # Notify this node's app
        process = Process(self.owner.request_logical_pair_app,
                          '_on_teleported_cnot_complete', [self.remote_node_name])
        event = Event(self.owner.timeline.now() + 1000, process)
        self.owner.timeline.schedule(event)

        # Also notify Bob's app so middle nodes detect "both links done"
        if self.role == 'alice' and self.bob_protocol is not None:
            bob_app = self.bob_protocol.owner.request_logical_pair_app
            if bob_app is not None:
                process_bob = Process(bob_app,
                                      '_on_teleported_cnot_complete', [self.owner.name])
                event_bob = Event(self.owner.timeline.now() + 1001, process_bob)
                self.owner.timeline.schedule(event_bob)