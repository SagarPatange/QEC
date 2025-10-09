"""QEC_protocol.py
Minimal QEC Protocol implementation following teleportation.py pattern.
Handles the quantum operations for [[7,1,3]] Steane code error correction.
"""

from enum import Enum, auto
from typing import Dict, List
from sequence.message import Message
from sequence.utils import log
from sequence.protocol import Protocol
from sequence.components.circuit import Circuit


class QECMsgType(Enum):
    """Message types for QEC protocol"""
    SYNDROME_REQUEST = auto()  # Request syndrome measurement
    SYNDROME_RESULT = auto()   # Report syndrome results
    CORRECTION_CMD = auto()    # Command to apply correction


class QECMessage(Message):
    """QEC protocol messages
    
    Similar to TeleportMessage but for error correction coordination.
    """
    
    def __init__(self, msg_type: QECMsgType, **kwargs):
        super().__init__(msg_type, 'qec_app')
        
        self.logical_id = kwargs.get('logical_id')
        
        if msg_type == QECMsgType.SYNDROME_RESULT:
            self.bit_syndromes = kwargs.get('bit_syndromes', [0, 0, 0])
            self.phase_syndromes = kwargs.get('phase_syndromes', [0, 0, 0])
            self.string = f'QECMessage(SYNDROME_RESULT, logical={self.logical_id})'
            
        elif msg_type == QECMsgType.CORRECTION_CMD:
            self.qubit_index = kwargs.get('qubit_index')
            self.correction_type = kwargs.get('correction_type')  # 'X', 'Z', 'Y'
            self.string = f'QECMessage(CORRECTION_CMD, logical={self.logical_id}, q={self.qubit_index}, type={self.correction_type})'
            
        elif msg_type == QECMsgType.SYNDROME_REQUEST:
            self.string = f'QECMessage(SYNDROME_REQUEST, logical={self.logical_id})'
        
        else:
            raise Exception(f"QECMessage created with unknown type: {msg_type}")
    
    def __str__(self):
        return self.string


class QECProtocol(Protocol): # TODO: inherit from protocol only when 2 nodes need to talk to each other
    """Core QEC protocol logic
    
    Similar to TeleportProtocol but handles continuous error correction
    instead of one-shot teleportation. Works with both QuantumRouter and
    QuantumRouter2ndGeneration nodes.
    
    When used with QuantumRouter2ndGeneration:
    - Can use data memories for logical qubits
    - Can use dedicated ancilla memories for syndrome measurements
    - Keeps communication memories free for distributed operations
    
    Attributes:
        node: Quantum router node running this protocol
        logical_id: ID of the logical qubit being protected
        physical_indices: List of 7 physical qubit indices (data memories)
        syndrome_interval: Time between syndrome measurements
        memory_array_name: Name of memory array holding the logical qubit
        ancilla_indices: List of 6 ancilla indices for syndrome measurements
        ancilla_array_name: Name of ancilla memory array (if using 2nd gen)
        has_ancillas: Whether dedicated ancillas are available
    """

    # Steane code encoding circuit (7 qubits)
    _encoding_circuit = Circuit(7)
    # H gates on qubits 4, 5, 6
    _encoding_circuit.h(4)
    _encoding_circuit.h(5)
    _encoding_circuit.h(6)
    # CNOT gates for encoding
    _encoding_circuit.cx(0, 1)
    _encoding_circuit.cx(0, 2)
    _encoding_circuit.cx(6, 3)
    _encoding_circuit.cx(6, 1)
    _encoding_circuit.cx(6, 0)
    _encoding_circuit.cx(5, 3)
    _encoding_circuit.cx(5, 2)
    _encoding_circuit.cx(5, 0)
    _encoding_circuit.cx(4, 3)
    _encoding_circuit.cx(4, 2)
    _encoding_circuit.cx(4, 1)

    # Decoding circuit (reverse of encoding)
    _decoding_circuit = Circuit(7)
    _decoding_circuit.cx(4, 1)
    _decoding_circuit.cx(4, 2)
    _decoding_circuit.cx(4, 3)
    _decoding_circuit.cx(5, 0)
    _decoding_circuit.cx(5, 2)
    _decoding_circuit.cx(5, 3)
    _decoding_circuit.cx(6, 0)
    _decoding_circuit.cx(6, 1)
    _decoding_circuit.cx(6, 3)
    _decoding_circuit.cx(0, 2)
    _decoding_circuit.cx(0, 1)
    _decoding_circuit.h(6)
    _decoding_circuit.h(5)
    _decoding_circuit.h(4)

    # Single qubit correction circuits
    _x_correction = Circuit(1)
    _x_correction.x(0)

    _z_correction = Circuit(1)
    _z_correction.z(0)

    def __init__(self, node, logical_id: int, 
                 physical_indices: List[int], syndrome_interval: int,
                 memory_array_name: str = None, ancilla_indices: List[int] = None,
                 ancilla_array_name: str = None):
        
        self.owner = node
        self.name = f"{self.node.name}"
        
        self.logical_id = logical_id
        self.physical_indices = physical_indices
        self.syndrome_interval = syndrome_interval
        self.memory_array_name = memory_array_name or node.memo_arr_name
        
        # For QuantumRouter2ndGeneration - ancilla support
        self.ancilla_indices = ancilla_indices or []
        self.ancilla_array_name = ancilla_array_name
        self.has_ancillas = len(self.ancilla_indices) > 0
        
        # Syndrome lookup tables for [[7,1,3]] Steane code
        self._init_syndrome_tables()
        
        log.logger.debug(f"{self.name}: initialized for logical {logical_id} using {self.memory_array_name}"
                        f", ancillas: {self.has_ancillas}")
    

    def received_message(self, src: str, msg: Message) -> bool:
        """Handle incoming QEC messages
        
        Required by Protocol base class. Similar to how TeleportProtocol
        handles messages but for error correction coordination.
        
        Args:
            src: Source node name
            msg: The incoming message
            
        Returns:
            bool: True if message was handled, False otherwise
        """
        # Check if this is a QEC message
        if not isinstance(msg, QECMessage):
            return False
        
        # Check if message is for this logical qubit
        if msg.logical_id != self.logical_id:
            return False
        
        log.logger.debug(f"{self.name}: received_message from {src}: {msg}")
        
        if msg.msg_type == QECMsgType.SYNDROME_REQUEST:
            # Another node requesting syndrome measurement participation
            # (For distributed QEC - not implemented in minimal version)
            log.logger.debug(f"{self.name}: Received syndrome request (not implemented)")
            return True
            
        elif msg.msg_type == QECMsgType.SYNDROME_RESULT:
            # Received syndrome results from another node
            # (For distributed QEC - not implemented in minimal version)
            log.logger.debug(f"{self.name}: Received syndrome results (not implemented)")
            return True
            
        elif msg.msg_type == QECMsgType.CORRECTION_CMD:
            # Correction command from coordinator node
            self.apply_correction_at_index(msg.qubit_index, msg.correction_type)
            log.logger.debug(f"{self.name}: Applied correction from remote command")
            return True
        
        # Unknown message type
        return False
    

    def _init_syndrome_tables(self):
        """Initialize syndrome decoding tables
        
        Maps 3-bit syndromes to error locations for [[7,1,3]] code.
        """
        # X error syndrome table (from Z stabilizer measurements)
        self.x_error_table = {
            (0, 0, 0): None,  # No error
            (1, 0, 0): 0,     # X on qubit 0
            (0, 1, 0): 1,     # X on qubit 1  
            (1, 1, 0): 2,     # X on qubit 2
            (0, 0, 1): 3,     # X on qubit 3
            (1, 0, 1): 4,     # X on qubit 4
            (0, 1, 1): 5,     # X on qubit 5
            (1, 1, 1): 6,     # X on qubit 6
        }
        
        # Z error syndrome table (from X stabilizer measurements)
        self.z_error_table = {
            (0, 0, 0): None,  # No error
            (1, 0, 0): 0,     # Z on qubit 0
            (0, 1, 0): 1,     # Z on qubit 1
            (1, 1, 0): 2,     # Z on qubit 2
            (0, 0, 1): 3,     # Z on qubit 3
            (1, 0, 1): 4,     # Z on qubit 4
            (0, 1, 1): 5,     # Z on qubit 5
            (1, 1, 1): 6,     # Z on qubit 6
        }
    

    def encode(self, initial_state=None):
        """Encode initial state into [[7,1,3]] logical qubit
        
        Similar to how TeleportProtocol prepares qubits for teleportation.
        
        Args:
            initial_state: Quantum state to encode (default |0⟩)
        """
        qm = self.owner.timeline.quantum_manager
        q = self.physical_indices
        
        # Get the correct memory array
        memory_array = self.owner.get_component_by_name(self.memory_array_name)
        
        # Set initial state on first physical qubit
        if initial_state is not None:
            memory_array[q[0]].update_state(initial_state)
        
        # Get quantum state keys for all physical qubits
        qubit_keys = [memory_array[q[i]].qstate_key for i in range(7)]

        # Apply Steane encoding circuit
        rnd = self.owner.get_generator().random()
        qm.run_circuit(self._encoding_circuit, qubit_keys, rnd)
        
        log.logger.debug(f"{self.name}: Encoded logical qubit")
    

    def decode(self):
        """Decode logical qubit back to single physical qubit
        
        Similar to how TeleportProtocol extracts the final teleported state.
        
        Returns:
            Decoded quantum state
        """
        qm = self.owner.timeline.quantum_manager
        q = self.physical_indices
        
        # Get memory array and quantum state keys
        memory_array = self.owner.get_component_by_name(self.memory_array_name)
        qubit_keys = [memory_array[q[i]].qstate_key for i in range(7)]

        # Apply inverse Steane encoding (decoding circuit)
        rnd = self.owner.get_generator().random()
        qm.run_circuit(self._decoding_circuit, qubit_keys, rnd)
        
        # Get final state from first qubit
        final_state = memory_array[q[0]].quantum_state

        log.logger.debug(f"{self.name}: Decoded logical qubit")
        return final_state
    

    def measure_syndromes(self) -> Dict:
        """Measure error syndromes
        
        For minimal version, returns placeholder syndromes.
        Full version would measure stabilizers with ancilla qubits.
        
        Returns:
            Dict with syndrome data
        """
        # In full implementation, would measure 6 stabilizers
        bit_syndromes = [0, 0, 0]
        phase_syndromes = [0, 0, 0]
        
        if self.has_ancillas:
            # With QuantumRouter2ndGeneration, we can use dedicated ancilla memories
            log.logger.debug(f"{self.name}: Would use ancilla memories {self.ancilla_indices} "
                           f"from {self.ancilla_array_name} for syndrome measurements")
            # TODO: Implement actual stabilizer measurements using ancillas
            # ancilla_array = self.owner.get_component_by_name(self.ancilla_array_name)
            # for i, ancilla_idx in enumerate(self.ancilla_indices[:3]):
            #     bit_syndromes[i] = self._measure_x_stabilizer_with_ancilla(ancilla_idx, ...)
            # for i, ancilla_idx in enumerate(self.ancilla_indices[3:6]):
            #     phase_syndromes[i] = self._measure_z_stabilizer_with_ancilla(ancilla_idx, ...)
        else:
            # Without dedicated ancillas, would need to use spare communication memories
            log.logger.debug(f"{self.name}: No dedicated ancillas - would use spare memories")
        
        return {
            'logical_id': self.logical_id,
            'bit_syndromes': bit_syndromes,
            'phase_syndromes': phase_syndromes
        }
    

    def needs_correction(self, syndrome_data: Dict) -> bool:
        """Check if error correction is needed
        
        Args:
            syndrome_data: Syndrome measurement results
            
        Returns:
            True if any syndrome is non-zero
        """
        bit_error = any(syndrome_data['bit_syndromes'])
        phase_error = any(syndrome_data['phase_syndromes'])
        return bit_error or phase_error
    

    def decode_error(self, syndrome_data: Dict) -> Dict[int, str]:
        """Decode syndromes to find error locations
        
        Args:
            syndrome_data: Syndrome measurement results
            
        Returns:
            Dict mapping qubit_index -> correction_type
        """
        corrections = {}
        
        # Check for X errors (from Z stabilizer syndromes)
        phase_syndrome = tuple(syndrome_data['phase_syndromes'])
        x_error_loc = self.x_error_table.get(phase_syndrome)
        if x_error_loc is not None:
            corrections[x_error_loc] = 'X'
        
        # Check for Z errors (from X stabilizer syndromes)
        bit_syndrome = tuple(syndrome_data['bit_syndromes'])
        z_error_loc = self.z_error_table.get(bit_syndrome)
        if z_error_loc is not None:
            if z_error_loc in corrections:
                # Both X and Z needed -> Y correction
                corrections[z_error_loc] = 'Y'
            else:
                corrections[z_error_loc] = 'Z'
        
        return corrections
    

    def apply_corrections(self, corrections: Dict[int, str]):
        """Apply error corrections
        
        Args:
            corrections: Dict mapping qubit_index -> correction_type
        """
        qm = self.owner.timeline.quantum_manager
        
        memory_array = self.owner.get_component_by_name(self.memory_array_name)
        for qubit_idx, correction_type in corrections.items():
            physical_idx = self.physical_indices[qubit_idx]
            qubit_key = memory_array[physical_idx].qstate_key
            
            rnd = self.owner.get_generator().random()
            if correction_type == 'X':
                qm.run_circuit(self._x_correction, [qubit_key], rnd)
            elif correction_type == 'Z':
                qm.run_circuit(self._z_correction, [qubit_key], rnd)
            elif correction_type == 'Y':
                # Y = XZ, apply both corrections
                qm.run_circuit(self._x_correction, [qubit_key], rnd)
                rnd2 = self.owner.get_generator().random()
                qm.run_circuit(self._z_correction, [qubit_key], rnd2)
            
            log.logger.debug(f"{self.name}: Applied {correction_type} to qubit {qubit_idx}")
   
    
    def apply_correction_at_index(self, qubit_index: int, correction_type: str):
        """Apply single correction (for remote corrections)
        
        Args:
            qubit_index: Which of the 7 qubits (0-6)
            correction_type: 'X', 'Z', or 'Y'
        """
        qm = self.owner.timeline.quantum_manager
        memory_array = self.owner.get_component_by_name(self.memory_array_name)
        physical_idx = self.physical_indices[qubit_index]
        qubit_key = memory_array[physical_idx].qstate_key
        
        rnd = self.owner.get_generator().random()
        if correction_type == 'X':
            qm.run_circuit(self._x_correction, [qubit_key], rnd)
        elif correction_type == 'Z':
            qm.run_circuit(self._z_correction, [qubit_key], rnd)
        elif correction_type == 'Y':
            qm.run_circuit(self._x_correction, [qubit_key], rnd)
            rnd2 = self.owner.get_generator().random()
            qm.run_circuit(self._z_correction, [qubit_key], rnd2)
        log.logger.debug(f"{self.name}: Applied {correction_type} to qubit {qubit_index}")


    def apply_logical_gate(self, gate_type: str):
        """Apply logical gate operation
        
        For CSS codes like Steane, logical X and Z are transversal.
        
        Args:
            gate_type: Type of gate ('X', 'Z', 'Y', 'H')
        """
        qm = self.owner.timeline.quantum_manager
        
        if gate_type in ['X', 'Z', 'Y']:
            # Transversal gates - apply to all physical qubits
            memory_array = self.owner.get_component_by_name(self.memory_array_name)
            for i in range(7):
                qubit_key = memory_array[self.physical_indices[i]].qstate_key
                rnd = self.owner.get_generator().random()
                if gate_type == 'X':
                    qm.run_circuit(self._x_correction, [qubit_key], rnd)
                elif gate_type == 'Z':
                    qm.run_circuit(self._z_correction, [qubit_key], rnd)
                elif gate_type == 'Y':
                    qm.run_circuit(self._x_correction, [qubit_key], rnd)
                    rnd2 = self.owner.get_generator().random()
                    qm.run_circuit(self._z_correction, [qubit_key], rnd2)
            log.logger.debug(f"{self.name}: Applied logical {gate_type}")
        
        elif gate_type == 'H':
            # Logical Hadamard is not transversal for CSS codes
            # Would need special circuit - not implemented in minimal version
            log.logger.warning(f"{self.name}: Logical H not implemented in minimal version")
        
        else:
            log.logger.warning(f"{self.name}: Unknown gate type {gate_type}")