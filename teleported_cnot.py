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
    
    # ==================== ALICE'S PHASE A METHODS ====================
    
    def alice_start_protocol(self):
        """Alice initiates the teleported CNOT protocol.
        
        Called by the app when all 7 Bell pairs are ready and encoding is complete.
        Groups all qubits (Alice's data, Alice's comm, Bob's comm, Bob's data) 
        into shared circuit, then begins serial processing.
        """
        assert self.role == 'alice', "Only Alice can start the protocol"
        
        self.start_time = self.owner.timeline.now()
        self.current_phase = 'ALICE_MEASURING'
        
        log.logger.info(f"[T:{self.start_time:,}] [{self.name}] Alice starting teleported CNOT protocol")
        
        # Collect all qubit keys that need to be grouped
        all_keys = []
        
        # Alice's data qubits (7 qubits)
        for i in range(7):
            all_keys.append(self.data_qubits[i].qstate_key)
        
        # Alice's communication qubits (7 Bell pair halves)
        for i in range(7):
            all_keys.append(self.communication_qubits[i].qstate_key)
        
        # Get Bob's node
        if self.remote_node is None:
            raise ValueError("remote_node not provided - cannot access Bob's data qubits for grouping")

        bob_node = self.remote_node

        # Get Bob's TeleportedCNOT protocol instance via the app reference
        if not hasattr(bob_node, 'request_logical_pair_app'):
            raise ValueError(f"Bob's node {bob_node.name} does not have request_logical_pair_app attribute")

        bob_app = bob_node.request_logical_pair_app
        if bob_app.teleported_cnot_protocol is None:
            raise ValueError(f"Bob's teleported CNOT protocol not initialized")

        bob_protocol = bob_app.teleported_cnot_protocol

        # Bob's data qubits (7 qubits)
        for i in range(7):
            all_keys.append(bob_protocol.data_qubits[i].qstate_key)
        
        log.logger.info(f"[T:{self.start_time:,}] [{self.name}] "
                       f"Grouping {len(all_keys)} qubits: "
                       f"Alice data + Alice comm + Bob data")
        
        # Group all qubits into shared circuit
        self.owner.timeline.quantum_manager.group_qubits(all_keys)
        
        log.logger.info(f"[T:{self.start_time:,}] [{self.name}] "
                       f"All qubits grouped into shared circuit")
        
        # Start processing first qubit
        self.alice_process_qubit(0)
        
  
    def alice_process_qubit(self, qubit_index: int):
        """Alice appends CNOT and measurement to circuit, then notifies Bob (deferred execution).

        Process:
        1. Append local CNOT(data[i] -> comm[i]) to circuit
        2. Append measurement M(comm[i]) to circuit
        3. Send notification message to Bob

        Args:
            qubit_index: Index of qubit to process (0-6)
        """
        if qubit_index >= 7:
            # All qubits processed by Alice
            self.current_phase = 'WAITING_FOR_BOB'
            log.logger.info(f"[T:{self.owner.timeline.now():,}] [{self.name}] "
                        f"Alice completed all measurements, waiting for Bob")
            return

        log.logger.debug(f"[T:{self.owner.timeline.now():,}] [{self.name}] "
                        f"Alice processing qubit {qubit_index}")

        data_key = self.data_qubits[qubit_index].qstate_key
        comm_key = self.communication_qubits[qubit_index].qstate_key

        # Get the shared circuit
        data_state = self.owner.timeline.quantum_manager.states[data_key]
        circuit = data_state.circuit

        # Append CNOT(data -> comm) - NO EXECUTION
        circuit.append('CX', [data_key, comm_key])

        # Append measurement of comm - NO EXECUTION
        circuit.append('M', [comm_key])

        self.alice_qubits_sent += 1

        log.logger.info(f"[T:{self.owner.timeline.now():,}] [{self.name}] "
                    f"Alice appended CNOT+M for qubit {qubit_index}")

        # Send notification message to Bob
        msg = TeleportedCNOTMessage(
            TeleportedCNOTMsgType.ALICE_MEASUREMENT,
            protocol_name=self.name,
            qubit_index=qubit_index
        )

        bob_protocol_name = f"TeleportedCNOT_{self.remote_node_name}"
        msg.receiver = bob_protocol_name
        self.owner.send_message(self.remote_node_name, msg)

        log.logger.info(f"[T:{self.owner.timeline.now():,}] [{self.name}] "
                        f"Alice sent notification for qubit {qubit_index} to {self.remote_node_name}")
    
    # ==================== BOB'S PHASE C METHODS ====================
    
    def bob_receive_alice_measurement(self, msg: TeleportedCNOTMessage):
        """Bob receives notification that Alice processed qubit i and schedules processing.
        
        Process:
        1. Extract qubit index from message
        2. Schedule processing of this qubit with local processing delay
        
        Args:
            msg: Message from Alice
        """
        assert self.role == 'bob', "Only Bob receives Alice's measurements"
        
        qubit_index = msg.qubit_index
        
        log.logger.debug(f"[T:{self.owner.timeline.now():,}] [{self.name}] "
                        f"Bob received notification for qubit {qubit_index}")
        
        # Schedule Bob's processing with local delay
        delay = 50  # Bob's local processing delay (50 ps)
        
        process = Process(self, "bob_process_qubit", [qubit_index])
        event = Event(self.owner.timeline.now() + delay, process)
        self.owner.timeline.schedule(event)
        
        log.logger.debug(f"[T:{self.owner.timeline.now():,}] [{self.name}] "
                    f"Scheduled Bob's processing of qubit {qubit_index} at T={self.owner.timeline.now() + delay:,}")
    
    
    def bob_process_qubit(self, qubit_index: int):
        """Bob appends operations and notifies Alice using target_rec(-1) (deferred execution).

        Process:
        1. Append CNOT(comm -> data)
        2. Append X correction controlled by Alice's measurement (target_rec(-1))
        3. Append Hadamard to comm
        4. Append measurement M(comm) in X basis
        5. Send notification message to Alice

        Args:
            qubit_index: Index of qubit (0-6)
        """
        log.logger.debug(f"[T:{self.owner.timeline.now():,}] [{self.name}] "
                        f"Bob processing qubit {qubit_index}")

        data_key = self.data_qubits[qubit_index].qstate_key
        comm_key = self.communication_qubits[qubit_index].qstate_key

        # Get the shared circuit
        data_state = self.owner.timeline.quantum_manager.states[data_key]
        circuit = data_state.circuit

        # Step 1: CNOT(comm -> data)
        circuit.append('CX', [comm_key, data_key])
        
        # Step 2: X correction controlled by Alice's Z measurement
        # Since we process serially, Alice's measurement is always the most recent (target_rec(-1))
        circuit.append('CX', [stim.target_rec(-1), data_key])
        
        # Step 3: Hadamard on comm (converts Z to X basis)
        circuit.append('H', [comm_key])
        
        # Step 4: Measure comm in Z basis (which is X after Hadamard)
        circuit.append('M', [comm_key])
        
        log.logger.info(f"[T:{self.owner.timeline.now():,}] [{self.name}] "
                    f"Bob appended operations for qubit {qubit_index}")
        
        # Send notification message to Alice
        msg = TeleportedCNOTMessage(
            TeleportedCNOTMsgType.BOB_MEASUREMENT,
            protocol_name=self.name,
            qubit_index=qubit_index
        )

        alice_protocol_name = f"TeleportedCNOT_{self.remote_node_name}"
        msg.receiver = alice_protocol_name
        self.owner.send_message(self.remote_node_name, msg)

        log.logger.info(f"[T:{self.owner.timeline.now():,}] [{self.name}] "
                        f"Bob sent notification for qubit {qubit_index} to {self.remote_node_name}")
        
        self.bob_qubits_processed += 1
        
        # Check if all qubits processed
        if self.bob_qubits_processed == 7:
            self.current_phase = 'WAITING_FOR_ALICE'
            log.logger.info(f"[T:{self.owner.timeline.now():,}] [{self.name}] "
                        f"Bob completed all operations, waiting for Alice")  
            
    # ==================== ALICE'S PHASE E METHODS ====================
    
    def alice_receive_bob_measurement(self, msg: TeleportedCNOTMessage):
        """Alice receives notification that Bob processed qubit i, applies Z correction, and continues.
        
        Process:
        1. Apply Z correction using target_rec(-1)
        2. Mark qubit as complete
        3. Schedule next qubit processing or complete protocol
        
        Args:
            msg: Message from Bob
        """
        assert self.role == 'alice', "Only Alice receives Bob's measurements"
        
        qubit_index = msg.qubit_index
        
        log.logger.debug(f"[T:{self.owner.timeline.now():,}] [{self.name}] "
                        f"Alice received notification for qubit {qubit_index}")
        
        # Apply final Z correction
        data_key = self.data_qubits[qubit_index].qstate_key
        data_state = self.owner.timeline.quantum_manager.states[data_key]
        circuit = data_state.circuit
        
        log.logger.info(f"[T:{self.owner.timeline.now():,}] [{self.name}] "
                    f"Alice applying Z correction for qubit {qubit_index}")
        
        # Z correction controlled by Bob's X measurement
        # Since we process serially, Bob's measurement is always the most recent (target_rec(-1))
        circuit.append('CZ', [stim.target_rec(-1), data_key])
        
        log.logger.info(f"[T:{self.owner.timeline.now():,}] [{self.name}] "
                    f"Alice appended Z correction for qubit {qubit_index}")
        
        # Mark as complete
        self.corrections_applied[qubit_index] = True
        self.qubits_processed += 1
        
        # Check if all qubits processed
        if self.qubits_processed == 7:
            self.protocol_complete()
        else:
            # Schedule processing of next qubit after a small delay
            next_qubit = qubit_index + 1
            delay = 100  # Small delay between qubits (100 ps)
            
            process = Process(self, "alice_process_qubit", [next_qubit])
            event = Event(self.owner.timeline.now() + delay, process)
            self.owner.timeline.schedule(event)
            
            log.logger.debug(f"[T:{self.owner.timeline.now():,}] [{self.name}] "
                        f"Scheduled qubit {next_qubit} processing at T={self.owner.timeline.now() + delay:,}")
        
    # ==================== COMPLETION ====================
    
    def protocol_complete(self):
        """Called when all 7 qubits processed."""
        self.end_time = self.owner.timeline.now()
        self.current_phase = 'COMPLETE'
        
        duration = self.end_time - self.start_time
        
        log.logger.info(f"[T:{self.end_time:,}] [{self.name}] "
                    f"Teleported CNOT protocol COMPLETE")
        log.logger.info(f"[{self.name}] Protocol duration: {duration:,} ps")
        log.logger.info(f"[{self.name}] All 7 qubits processed successfully")
        
        # Schedule notification to app via timeline
        delay = 1000  # 1 nanosecond for local processing
        process = Process(self.owner.request_logical_pair_app, '_on_teleported_cnot_complete', [])
        event = Event(self.owner.timeline.now() + delay, process)
        self.owner.timeline.schedule(event)
        log.logger.info(f"[{self.name}] Scheduled app completion notification")  
      
    # ==================== MESSAGE HANDLING ====================
    
    def received_message(self, src: str, msg: TeleportedCNOTMessage):
        """Route incoming messages based on type and role.
        
        Args:
            src: Source node name
            msg: Incoming message
        
        Returns:
            bool: True if message was handled
        """
        if self.role == 'bob':
            assert msg.msg_type == TeleportedCNOTMsgType.ALICE_MEASUREMENT, \
                f"[{self.name}] Bob received wrong message type: {msg.msg_type} from {src}"
            self.bob_receive_alice_measurement(msg)
            return True
            
        elif self.role == 'alice':
            assert msg.msg_type == TeleportedCNOTMsgType.BOB_MEASUREMENT, \
                f"[{self.name}] Alice received wrong message type: {msg.msg_type} from {src}"
            self.alice_receive_bob_measurement(msg)
            return True
            
        else:
            assert False, f"[{self.name}] Invalid role: {self.role}"
  
    
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