"""
QEC 713 Protocol - [[7,1,3]] Steane Code Implementation

This module provides a simple class for applying the [[7,1,3]] quantum error correction code.
The protocol operates locally on each node and protects distributed entanglement from single-qubit errors.

CIRCUIT-BASED DESIGN:
All methods return Circuit or stim.Circuit objects that can be executed with qm.run_circuit().
This separates circuit construction from execution for better modularity.
"""

from typing import TYPE_CHECKING, List, Dict, Optional, Tuple
import stim
from sequence.components.circuit import Circuit
from sequence.utils import log

if TYPE_CHECKING:
    from sequence.kernel.quantum_manager import QuantumManager


class QEC713:
    """Simple class for [[7,1,3]] quantum error correction operations.
    
    This is NOT a protocol in the EntanglementProtocol sense - it's a utility class
    for applying local QEC operations to protect distributed entanglement.
    
    The [[7,1,3]] code:
    - Encodes 1 logical qubit into 7 physical qubits
    - Can correct 1 arbitrary single-qubit error
    - Uses 6 ancilla qubits for syndrome measurements
    
    USAGE PATTERN:
        # Get a circuit
        encoding_circuit = QEC713.create_encoding_circuit()
        
        # Execute it with quantum manager
        qm.run_circuit(encoding_circuit, qubit_keys, rnd=0.5)
    """
    
    # Syndrome lookup table for [[7,1,3]] Steane code
    # Maps 3-bit syndrome to qubit index (same for both X and Z errors)
    SYNDROME_TABLE = {
        (0, 0, 0): None,  # No error
        (1, 0, 0): 0,     # Error on qubit 0
        (0, 1, 0): 1,     # Error on qubit 1
        (1, 1, 0): 2,     # Error on qubit 2
        (0, 0, 1): 3,     # Error on qubit 3
        (1, 0, 1): 4,     # Error on qubit 4
        (0, 1, 1): 5,     # Error on qubit 5
        (1, 1, 1): 6,     # Error on qubit 6
    }
    
    def __init__(self):
        """Constructor for QEC713 utility class."""
        pass
    
    @staticmethod
    def create_encoding_circuit(num_qubits: int = 7) -> stim.Circuit:
        """Create [[7,1,3]] encoding circuit as a stim.Circuit.
        
        This circuit transforms |ψ⟩|0⟩^6 → |ψ⟩_L (logical encoded state)
        
        Args:
            num_qubits: Number of qubits (must be 7)
            
        Returns:
            stim.Circuit that performs encoding on qubits 0-6
            
        Usage:
            circuit = QEC713.create_encoding_circuit()
            qm.run_circuit(circuit, data_keys, rnd=0.5)
        """
        assert num_qubits == 7, "Need exactly 7 qubits for [[7,1,3]] encoding"
        
        circuit = stim.Circuit()
        
        # circuit.append("R", [0,1,2,3,4,5,6])   # Reset all qubits to |0>
        
        # Apply Hadamards to qubits 4, 5, 6
        circuit.append("H", [4, 5, 6])
        
        # CNOT cascade from qubit 0
        circuit.append("CX", [0, 1])
        circuit.append("CX", [0, 2])
        
        # CNOT cascade from qubit 6
        circuit.append("CX", [6, 3])
        circuit.append("CX", [6, 1])
        circuit.append("CX", [6, 0])
        
        # CNOT cascade from qubit 5
        circuit.append("CX", [5, 3])
        circuit.append("CX", [5, 2])
        circuit.append("CX", [5, 0])
        
        # CNOT cascade from qubit 4
        circuit.append("CX", [4, 3])
        circuit.append("CX", [4, 2])
        circuit.append("CX", [4, 1])
        
        return circuit
    
    @staticmethod
    def create_decoding_circuit(num_qubits: int = 7) -> stim.Circuit:
        """Create [[7,1,3]] decoding circuit as a stim.Circuit.
        
        This is the inverse of the encoding circuit.
        
        Args:
            num_qubits: Number of qubits (must be 7)
            
        Returns:
            stim.Circuit that performs decoding on qubits 0-6
            
        Usage:
            circuit = QEC713.create_decoding_circuit()
            qm.run_circuit(circuit, data_keys, rnd=0.5)
        """
        assert num_qubits == 7, "Need exactly 7 qubits for [[7,1,3]] decoding"
        
        circuit = stim.Circuit()
        
        # Reverse of encoding - apply gates in opposite order
        circuit.append("CX", [4, 1])
        circuit.append("CX", [4, 2])
        circuit.append("CX", [4, 3])
        
        circuit.append("CX", [5, 0])
        circuit.append("CX", [5, 2])
        circuit.append("CX", [5, 3])
        
        circuit.append("CX", [6, 0])
        circuit.append("CX", [6, 1])
        circuit.append("CX", [6, 3])
        
        circuit.append("CX", [0, 2])
        circuit.append("CX", [0, 1])
        
        circuit.append("H", [4, 5, 6])
        
        return circuit
    
    @staticmethod
    def create_hadamard_circuit() -> Circuit:
        """Create a single-qubit Hadamard circuit.
        
        Returns:
            Circuit with H gate
            
        Usage:
            h_circuit = QEC713.create_hadamard_circuit()
            qm.run_circuit(h_circuit, [qubit_key], rnd=0.5)
        """
        circuit = Circuit(1)
        circuit.h(0)
        return circuit
    
    @staticmethod
    def create_x_correction_circuit() -> Circuit:
        """Create a single-qubit X correction circuit.
        
        Returns:
            Circuit with X gate
        """
        circuit = Circuit(1)
        circuit.x(0)
        return circuit
    
    @staticmethod
    def create_z_correction_circuit() -> Circuit:
        """Create a single-qubit Z correction circuit.
        
        Returns:
            Circuit with Z gate
        """
        circuit = Circuit(1)
        circuit.z(0)
        return circuit
    
    @staticmethod
    def create_y_correction_circuit() -> Circuit:
        """Create a single-qubit Y correction circuit.
        
        Returns:
            Circuit with Y gate (implemented as X then Z)
        """
        circuit = Circuit(1)
        circuit.x(0)
        circuit.z(0)
        return circuit
    
    @staticmethod
    def create_logical_x_circuit(num_qubits: int = 7) -> Circuit:
        """Create logical X gate circuit (transversal X on all 7 qubits).
        
        Returns:
            Circuit that applies X to all physical qubits
        """
        circuit = Circuit(num_qubits)
        for i in range(num_qubits):
            circuit.x(i)
        return circuit
    
    @staticmethod
    def create_logical_z_circuit(num_qubits: int = 7) -> Circuit:
        """Create logical Z gate circuit (transversal Z on all 7 qubits).
        
        Returns:
            Circuit that applies Z to all physical qubits
        """
        circuit = Circuit(num_qubits)
        for i in range(num_qubits):
            circuit.z(i)
        return circuit
    
    @staticmethod
    def create_logical_h_circuit(num_qubits: int = 7) -> Circuit:
        """Create logical H gate circuit (transversal H on all 7 qubits).
        
        Returns:
            Circuit that applies H to all physical qubits
        """
        circuit = Circuit(num_qubits)
        for i in range(num_qubits):
            circuit.h(i)
        return circuit
    
    @staticmethod
    def create_logical_cnot_circuit(num_qubits: int = 14) -> Circuit:
        """Create logical CNOT gate circuit (transversal CNOTs between two logical qubits).
        
        Args:
            num_qubits: Total qubits (must be 14 for two logical qubits)
            
        Returns:
            Circuit that applies CNOT between corresponding physical qubits
            Control qubits are 0-6, target qubits are 7-13
        """
        assert num_qubits == 14, "Need 14 qubits for logical CNOT (7 control + 7 target)"
        
        circuit = Circuit(num_qubits)
        for i in range(7):
            circuit.cx(i, i + 7)
        return circuit
    
    @staticmethod
    def create_stabilizer_measurement_circuit(num_data_qubits: int = 7, 
                                             num_ancilla_qubits: int = 6) -> stim.Circuit:
        """Create stabilizer measurement circuit for [[7,1,3]] code.
        
        Args:
            num_data_qubits: Number of data qubits (must be 7)
            num_ancilla_qubits: Number of ancilla qubits (must be 6)
            
        Returns:
            stim.Circuit that measures all 6 stabilizers
            Data qubits are 0-6, ancilla qubits are 7-12
            
        Usage:
            circuit = QEC713.create_stabilizer_measurement_circuit()
            qm.run_circuit(circuit, data_keys + ancilla_keys, rnd=0.5)
        """
        assert num_data_qubits == 7, "Need 7 data qubits"
        assert num_ancilla_qubits == 6, "Need 6 ancilla qubits"
        
        circuit = stim.Circuit()
        
        # Data qubits are 0-6, ancilla qubits are 7-12
        d = list(range(7))
        a = list(range(7, 13))
        
        # Bit-flip syndrome measurements (X-type stabilizers)
        circuit.append("CX", [d[0], a[0]])
        circuit.append("CX", [d[2], a[0]])
        circuit.append("CX", [d[4], a[0]])
        circuit.append("CX", [d[6], a[0]])
        
        circuit.append("CX", [d[1], a[1]])
        circuit.append("CX", [d[2], a[1]])
        circuit.append("CX", [d[5], a[1]])
        circuit.append("CX", [d[6], a[1]])
        
        circuit.append("CX", [d[3], a[2]])
        circuit.append("CX", [d[4], a[2]])
        circuit.append("CX", [d[5], a[2]])
        circuit.append("CX", [d[6], a[2]])
        
        # Phase-flip syndrome measurements (Z-type stabilizers)
        circuit.append("H", [a[3], a[4], a[5]])
        
        circuit.append("CX", [a[3], d[0]])
        circuit.append("CX", [a[3], d[2]])
        circuit.append("CX", [a[3], d[4]])
        circuit.append("CX", [a[3], d[6]])
        
        circuit.append("CX", [a[4], d[1]])
        circuit.append("CX", [a[4], d[2]])
        circuit.append("CX", [a[4], d[5]])
        circuit.append("CX", [a[4], d[6]])
        
        circuit.append("CX", [a[5], d[3]])
        circuit.append("CX", [a[5], d[4]])
        circuit.append("CX", [a[5], d[5]])
        circuit.append("CX", [a[5], d[6]])
        
        circuit.append("H", [a[3], a[4], a[5]])
        
        # Measure ancillas
        circuit.append("M", a)
        
        return circuit
    
    @staticmethod
    def decode_syndrome(bit_syndromes: List[int], phase_syndromes: List[int]) -> Dict[int, str]:
        """Decode syndromes to find error locations and types.
        
        Args:
            bit_syndromes: List of 3 bit-flip syndrome measurements
            phase_syndromes: List of 3 phase-flip syndrome measurements
            
        Returns:
            Dict mapping qubit_index -> correction_type ('X', 'Z', or 'Y')
        """
        corrections = {}
        
        # Check for X errors (from Z stabilizer syndromes)
        phase_syndrome = tuple(phase_syndromes)
        x_error_loc = QEC713.SYNDROME_TABLE.get(phase_syndrome)
        if x_error_loc is not None:
            corrections[x_error_loc] = 'X'
        
        # Check for Z errors (from X stabilizer syndromes)
        bit_syndrome = tuple(bit_syndromes)
        z_error_loc = QEC713.SYNDROME_TABLE.get(bit_syndrome)
        if z_error_loc is not None:
            if z_error_loc in corrections:
                # Both X and Z needed -> Y correction
                corrections[z_error_loc] = 'Y'
            else:
                corrections[z_error_loc] = 'Z'
        
        return corrections
    
    # ========================================================================
    # LEGACY METHODS (for backward compatibility - direct QM manipulation)
    # ========================================================================
    
    @staticmethod
    def encode(qm: 'QuantumManager', data_keys: List[int]) -> None:
        """LEGACY: Encode 7 physical qubits into a [[7,1,3]] logical qubit.
        
        Note: Consider using create_encoding_circuit() + qm.run_circuit() instead.
        
        Args:
            qm: Quantum manager instance (must be QuantumManagerStabilizer)
            data_keys: List of 7 qubit keys to encode (order matters!)
        
        Raises:
            AssertionError: If not exactly 7 qubits provided
        """
        assert len(data_keys) == 7, "Need exactly 7 qubits for [[7,1,3]] encoding"
        
        # Use the circuit-based approach
        circuit = QEC713.create_encoding_circuit()
        qm.run_circuit(circuit, data_keys, 0.5)
        
        log.logger.debug(f"Encoded logical qubit using qubits {data_keys}")
    
    @staticmethod
    def decode(qm: 'QuantumManager', data_keys: List[int]) -> None:
        """LEGACY: Decode a [[7,1,3]] logical qubit back to 7 physical qubits.
        
        Note: Consider using create_decoding_circuit() + qm.run_circuit() instead.
        
        Args:
            qm: Quantum manager instance
            data_keys: List of 7 qubit keys to decode (same order as encoding)
        
        Raises:
            AssertionError: If not exactly 7 qubits provided
        """
        assert len(data_keys) == 7, "Need exactly 7 qubits for [[7,1,3]] decoding"
        
        # Use the circuit-based approach
        circuit = QEC713.create_decoding_circuit()
        qm.run_circuit(circuit, data_keys, 0.5)
        
        log.logger.debug(f"Decoded logical qubit from qubits {data_keys}")
    
    @staticmethod
    def apply_logical_gate(qm: 'QuantumManager', data_keys: List[int], gate_type: str) -> None:
        """LEGACY: Apply logical gate operation to the encoded logical qubit.
        
        Note: Consider using create_logical_x_circuit() etc. + qm.run_circuit() instead.
        
        For the [[7,1,3]] Steane code (CSS code), logical X, Z, Y, and H are all transversal.
        
        Args:
            qm: Quantum manager instance
            data_keys: List of 7 data qubit keys
            gate_type: Type of gate ('X', 'Z', 'Y', 'H')
        
        Raises:
            ValueError: If gate_type is not supported
        """
        supported_gates = ['X', 'Z', 'Y', 'H']
        
        if gate_type not in supported_gates:
            raise ValueError(f"Gate type '{gate_type}' not supported. Use one of {supported_gates}")
        
        # Create and apply appropriate circuit
        if gate_type == 'X':
            circuit = QEC713.create_logical_x_circuit()
        elif gate_type == 'Z':
            circuit = QEC713.create_logical_z_circuit()
        elif gate_type == 'H':
            circuit = QEC713.create_logical_h_circuit()
        elif gate_type == 'Y':
            # Y = XZ
            x_circuit = QEC713.create_logical_x_circuit()
            z_circuit = QEC713.create_logical_z_circuit()
            qm.run_circuit(x_circuit, data_keys, 0.5)
            qm.run_circuit(z_circuit, data_keys, 0.5)
            log.logger.debug(f"Applied transversal logical {gate_type} gate to logical qubit")
            return
        
        qm.run_circuit(circuit, data_keys, 0.5)
        log.logger.debug(f"Applied transversal logical {gate_type} gate to logical qubit")
    
    @staticmethod
    def apply_logical_cnot(qm: 'QuantumManager', control_keys: List[int], 
                          target_keys: List[int]) -> None:
        """LEGACY: Apply logical CNOT between two encoded logical qubits.
        
        Note: Consider using create_logical_cnot_circuit() + qm.run_circuit() instead.
        
        For the [[7,1,3]] Steane code, logical CNOT is transversal - 
        apply CNOT between corresponding physical qubits.
        
        Args:
            qm: Quantum manager instance
            control_keys: List of 7 qubit keys for control logical qubit
            target_keys: List of 7 qubit keys for target logical qubit
        
        Raises:
            AssertionError: If wrong number of qubits provided
        """
        assert len(control_keys) == 7, "Control logical qubit needs 7 physical qubits"
        assert len(target_keys) == 7, "Target logical qubit needs 7 physical qubits"
        
        # Use the circuit-based approach
        circuit = QEC713.create_logical_cnot_circuit()
        all_keys = control_keys + target_keys
        qm.run_circuit(circuit, all_keys, 0.5)
        
        log.logger.debug(f"Applied transversal logical CNOT between logical qubits")