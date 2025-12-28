"""Teleported CNOT Protocol for Encoded Qubits

Implements transversal teleportation-based CNOT gates between logical qubits
following PhysRevA.79.032325 Figure 3, Step 1(iii).

This protocol operates on 7 physical Bell pairs to implement an encoded CNOT gate
between Alice's and Bob's logical qubits, creating an encoded Bell pair.

Protocol Flow:
  Phase A: Alice's local operations and Z measurements
  Phase B: Classical communication Alice → Bob  
  Phase C: Bob's X corrections, operations, and X measurements
  Phase D: Classical communication Bob → Alice
  Phase E: Alice's final Z corrections
"""

from enum import Enum, auto
from typing import TYPE_CHECKING, Dict, List, Optional
from sequence.protocol import Protocol
from sequence.message import Message
from sequence.components.circuit import Circuit
from sequence.utils import log
from sequence.kernel.process import Process
from sequence.kernel.event import Event
import stim

if TYPE_CHECKING:
    from sequence.topology.node import QuantumRouter
    from sequence.components.memory import Memory


class TeleportedCNOTMsgType(Enum):
    """Message types for teleported CNOT protocol."""
    ALICE_MEASUREMENT = auto()  # Alice sends Z measurement result for qubit i
    BOB_MEASUREMENT = auto()     # Bob sends X measurement result for qubit i


class TeleportedCNOTMessage(Message):
    """Classical messages for teleported CNOT protocol.
    
    Attributes:
        msg_type (TeleportedCNOTMsgType): Type of message
        qubit_index (int): Which physical qubit (0-6) this message refers to
        measurement_result (int): 0 or 1 measurement outcome
        protocol_name (str): Name of the protocol instance
    """
    
    def __init__(self, msg_type: TeleportedCNOTMsgType, protocol_name: str, **kwargs):
        super().__init__(msg_type, protocol_name)
        self.protocol_name = protocol_name
        
        if msg_type in [TeleportedCNOTMsgType.ALICE_MEASUREMENT, TeleportedCNOTMsgType.BOB_MEASUREMENT]:
            self.qubit_index = kwargs['qubit_index']
        else:
            raise ValueError(f"Unknown message type: {msg_type}")
    
    def __str__(self):
        if self.msg_type in [TeleportedCNOTMsgType.ALICE_MEASUREMENT, TeleportedCNOTMsgType.BOB_MEASUREMENT]:
            return (f"TeleportedCNOTMessage(type={self.msg_type.name}, "
                f"qubit={self.qubit_index})")
        else:
            return f"TeleportedCNOTMessage(type={self.msg_type.name})"


class TeleportedCNOTProtocol(Protocol):
    """Protocol implementing teleportation-based CNOT gate across 7 physical qubits.
    
    This implements the transversal encoded CNOT gate by performing 7 independent
    teleportation-based CNOT operations, one for each physical qubit in the encoding block.
    
    Asymmetric roles:
    - Alice (initiator): Control qubit owner, starts protocol
    - Bob (responder): Target qubit owner, reacts to messages
    
    Attributes:
        owner (QuantumRouter): The quantum router node
        name (str): Protocol instance name
        role (str): 'alice' or 'bob'
        remote_node_name (str): Name of the partner node
        data_qubits (List[Memory]): 7 encoded logical qubit memories (local)
        communication_qubits (List[Memory]): 7 Bell pair qubit memories (remote entangled partners)
        alice_measurements (Dict[int, int]): Z measurement results (qubit_index -> result)
        bob_measurements (Dict[int, int]): X measurement results
        corrections_applied (Dict[int, bool]): Track which corrections done
        current_phase (str): Current protocol phase
        qubits_processed (int): How many of 7 qubits have been processed
    """
    
    def __init__(self, owner: "QuantumRouter", name: str, role: str,
                 remote_node_name: str, data_qubits: List["Memory"],
                 communication_qubits: List["Memory"], remote_node=None):
        """Initialize teleported CNOT protocol.

        Args:
            owner: Quantum router node
            name: Protocol identifier
            role: 'alice' or 'bob'
            remote_node_name: Partner node name
            data_qubits: List of 7 Memory objects (encoded logical qubits, local)
            communication_qubits: List of 7 Memory objects (Bell pair qubits, remote)
        """
        super().__init__(owner, name)

        assert role in ['alice', 'bob'], f"Role must be 'alice' or 'bob', got {role}"
        assert len(data_qubits) == 7, f"Need exactly 7 data qubits, got {len(data_qubits)}"
        assert len(communication_qubits) == 7, f"Need exactly 7 communication qubits, got {len(communication_qubits)}"

        self.role = role
        self.remote_node_name = remote_node_name
        self.data_qubits = data_qubits
        self.communication_qubits = communication_qubits
        self.remote_node = remote_node 

        # State tracking for each qubit
        self.corrections_applied: Dict[int, bool] = {i: False for i in range(7)}
        
        # Phase tracking
        self.current_phase = 'IDLE'
        self.qubits_processed = 0
        self.alice_qubits_sent = 0
        self.bob_qubits_processed = 0
        
        # Timing
        self.start_time = None
        self.end_time = None
    
    
        
        log.logger.info(f"[{self.name}] TeleportedCNOTProtocol initialized as {role}")
    
    # ==================== PHASE A: Alice's Operations ====================
    
    def alice_start_protocol(self):
        """Alice groups qubits, applies transversal CX + M, notifies Bob."""
        assert self.role == 'alice', "Only Alice can start the protocol"
        
        self.start_time = self.owner.timeline.now()
        self.current_phase = 'PHASE_A'
        
        log.logger.info(f"[T:{self.start_time:,}] [{self.name}] Alice starting teleported CNOT")
        
        # ========================================================================
        # SIMULATION WORKAROUND - NOT LOCC COMPLIANT
        # ========================================================================
        # The following code accesses Bob's quantum state keys through the remote
        # node reference. This is ONLY necessary because Stim requires all entangled
        # qubits to be in the same stabilizer tableau before applying operations.
        #
        # PHYSICAL REALITY:
        # In a real teleported CNOT, Alice operates ONLY on her local qubits:
        # - Her encoded data qubits (control)
        # - Her half of the Bell pairs (communication qubits)
        # She measures, sends classical bits to Bob, and Bob performs corrections.
        # NO access to Bob's quantum state is needed or possible.
        #
        # SIMULATION REQUIREMENT:
        # Stim's group_qubits() needs all qubit keys to merge separate tableaus.
        # This is purely a simulator bookkeeping operation with no physical analog.
        # All subsequent operations (measurements, gates, corrections) are still
        # strictly local to each party and fully LOCC-compliant.
        # ========================================================================
        
        # === Validate Bob's node and protocol ===
        if self.remote_node is None:
            raise ValueError("remote_node not provided")
        
        if not hasattr(self.remote_node, 'request_logical_pair_app'):
            raise ValueError(f"Bob's node {self.remote_node.name} missing request_logical_pair_app")
        
        bob_app = self.remote_node.request_logical_pair_app
        if bob_app.teleported_cnot_protocol is None:
            raise ValueError("Bob's teleported CNOT protocol not initialized")
        
        bob_protocol = bob_app.teleported_cnot_protocol
        
        # === Collect all qubit keys for grouping ===
        all_keys = []
        
        # Alice's data qubits (7)
        for i in range(7):
            all_keys.append(self.data_qubits[i].qstate_key)
        
        # Alice's communication qubits (7)
        for i in range(7):
            all_keys.append(self.communication_qubits[i].qstate_key)
        
        # Bob's communication qubits (7) - NOT LOCC COMPLIANT
        for i in range(7):
            all_keys.append(bob_protocol.communication_qubits[i].qstate_key)
        
        # Bob's data qubits (7) - NOT LOCC COMPLIANT
        for i in range(7):
            all_keys.append(bob_protocol.data_qubits[i].qstate_key)
        
        log.logger.info(f"[T:{self.start_time:,}] [{self.name}] Grouping {len(all_keys)} qubits: "
                        f"Alice data (7) + Alice comm (7) + Bob comm (7) + Bob data (7)")
        
        # Group into shared tableau
        self.owner.timeline.quantum_manager.group_qubits(all_keys)
        
        # === Get shared circuit ===
        circuit = self.owner.timeline.quantum_manager.states[all_keys[0]].circuit
        
        # === Apply transversal CX(data -> comm) ===
        for i in range(7):
            data_key = self.data_qubits[i].qstate_key
            comm_key = self.communication_qubits[i].qstate_key
            circuit.append('CX', [data_key, comm_key])
        
        # === Apply measurements M(comm) ===
        for i in range(7):
            comm_key = self.communication_qubits[i].qstate_key
            circuit.append('M', [comm_key])
        
        log.logger.info(f"[T:{self.owner.timeline.now():,}] [{self.name}] "
                        f"Alice appended 7 CX + 7 M")
        
        # === Send message to Bob ===
        msg = TeleportedCNOTMessage(
            TeleportedCNOTMsgType.ALICE_MEASUREMENT,
            protocol_name=self.name,
            qubit_index=-1
        )
        msg.receiver = f"TeleportedCNOT_{self.remote_node_name}"
        self.owner.send_message(self.remote_node_name, msg)
        
        self.current_phase = 'WAITING_FOR_BOB'
        
    # ==================== PHASE C: Bob's Operations ====================

    def bob_process_all_qubits(self):
        """Bob applies CX, X_correction, H, M for all 7 qubits."""
        self.current_phase = 'PHASE_C'
        
        log.logger.info(f"[T:{self.owner.timeline.now():,}] [{self.name}] "
                        f"Bob processing all qubits")
        
        # Get shared circuit
        circuit = self.owner.timeline.quantum_manager.states[self.data_qubits[0].qstate_key].circuit
        
        for i in range(7):
            data_key = self.data_qubits[i].qstate_key
            comm_key = self.communication_qubits[i].qstate_key
            
            circuit.append('CX', [comm_key, data_key])
            circuit.append('CX', [stim.target_rec(-7), data_key])
            circuit.append('H', [comm_key])
            circuit.append('M', [comm_key])
        
        log.logger.info(f"[T:{self.owner.timeline.now():,}] [{self.name}] "
                        f"Bob appended 7 (CX + X_corr + H + M)")
        
        # Send message to Alice
        msg = TeleportedCNOTMessage(
            TeleportedCNOTMsgType.BOB_MEASUREMENT,
            protocol_name=self.name,
            qubit_index=-1
        )
        msg.receiver = f"TeleportedCNOT_{self.remote_node_name}"
        self.owner.send_message(self.remote_node_name, msg)
        
        self.current_phase = 'WAITING_FOR_ALICE'


    # ==================== PHASE E: Alice's Final Corrections ====================

    def alice_apply_z_corrections(self):
        """Alice applies Z corrections for all 7 qubits."""
        self.current_phase = 'PHASE_E'
        
        log.logger.info(f"[T:{self.owner.timeline.now():,}] [{self.name}] "
                        f"Alice applying Z corrections")
        
        # Get shared circuit
        circuit = self.owner.timeline.quantum_manager.states[self.data_qubits[0].qstate_key].circuit
        
        # Bob's measurement for qubit i is at index 7+i
        # target_rec(i - 7) references measurement at index (current_count) + (i - 7)
        for i in range(7):
            data_key = self.data_qubits[i].qstate_key
            circuit.append('CZ', [stim.target_rec(i - 7), data_key])
        
        log.logger.info(f"[T:{self.owner.timeline.now():,}] [{self.name}] "
                        f"Alice appended 7 CZ corrections")
        
        self.protocol_complete()


    # ==================== MESSAGE HANDLING ====================

    def received_message(self, src: str, msg: TeleportedCNOTMessage):
        """Route incoming messages and schedule appropriate processing.
        
        Args:
            src: Source node name
            msg: Incoming message
        
        Returns:
            bool: True if message was handled
        """
        if self.role == 'bob':
            assert msg.msg_type == TeleportedCNOTMsgType.ALICE_MEASUREMENT, \
                f"[{self.name}] Bob received wrong message type: {msg.msg_type}"
            
            log.logger.debug(f"[T:{self.owner.timeline.now():,}] [{self.name}] "
                            f"Bob received Alice's notification")
            
            process = Process(self, "bob_process_all_qubits", [])
            event = Event(self.owner.timeline.now() + 50, process)
            self.owner.timeline.schedule(event)
            return True
            
        elif self.role == 'alice':
            assert msg.msg_type == TeleportedCNOTMsgType.BOB_MEASUREMENT, \
                f"[{self.name}] Alice received wrong message type: {msg.msg_type}"
            
            log.logger.debug(f"[T:{self.owner.timeline.now():,}] [{self.name}] "
                            f"Alice received Bob's notification")
            
            process = Process(self, "alice_apply_z_corrections", [])
            event = Event(self.owner.timeline.now() + 50, process)
            self.owner.timeline.schedule(event)
            return True
        
        else:
            raise ValueError(f"[{self.name}] Invalid role: {self.role}")


    # ==================== COMPLETION ====================

    def protocol_complete(self):
        """Called when all 7 qubits processed."""
        self.end_time = self.owner.timeline.now()
        self.current_phase = 'COMPLETE'
        
        duration = self.end_time - self.start_time
        
        log.logger.info(f"[T:{self.end_time:,}] [{self.name}] Teleported CNOT protocol COMPLETE")
        log.logger.info(f"[{self.name}] Protocol duration: {duration:,} ps")
        
        # Notify app
        process = Process(self.owner.request_logical_pair_app, '_on_teleported_cnot_complete', [])
        event = Event(self.owner.timeline.now() + 1000, process)
        self.owner.timeline.schedule(event)
    
    
    def is_ready(self) -> bool:
        """Check if protocol is ready to start."""
        return True
 
    
    def start(self):
        """Start the protocol (for compatibility with Protocol interface)."""
        if self.role == 'alice':
            self.alice_start_protocol()
  
    
    def _get_measurement_count(self, circuit: stim.Circuit) -> int:
        """Count total measurements in circuit so far.
        
        Args:
            circuit: The stim.Circuit to count measurements in
            
        Returns:
            Total number of measurements appended to circuit
        """
        count = 0
        for instruction in circuit:
            if instruction.name == 'M':
                count += len(instruction.targets_copy())
        return count