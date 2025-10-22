"""
QEC 713 Protocol - [[7,1,3]] Steane Code Implementation

This module provides a simple class for applying the [[7,1,3]] quantum error correction code.
The protocol operates locally on each node and protects distributed entanglement from single-qubit errors.
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
    
    # Pre-compiled correction circuits for efficiency
    _x_correction = Circuit(1)
    _x_correction.x(0)
    
    _z_correction = Circuit(1)
    _z_correction.z(0)
    
    def __init__(self):
        """Constructor for QEC713 utility class."""
        pass
    
    @staticmethod
    def encode(qm: 'QuantumManager', data_keys: List[int]) -> None:
        """Encode 7 physical qubits into a [[7,1,3]] logical qubit.

        Args:
            qm: Quantum manager instance (must be QuantumManagerStabilizer)
            data_keys: List of 7 qubit keys to encode (order matters!)
        
        Raises:
            AssertionError: If not exactly 7 qubits provided
        """
        assert len(data_keys) == 7, "Need exactly 7 qubits for [[7,1,3]] encoding"

        # Get the state (all qubits should be in same grouped state)
        state = qm.states[data_keys[0]]

        # Apply encoding circuit directly to Stim circuit
        q = data_keys
        state.circuit.append("H", [q[4], q[5], q[6]])

        state.circuit.append("CX", [q[0], q[1]])
        state.circuit.append("CX", [q[0], q[2]])

        state.circuit.append("CX", [q[6], q[3]])
        state.circuit.append("CX", [q[6], q[1]])
        state.circuit.append("CX", [q[6], q[0]])

        state.circuit.append("CX", [q[5], q[3]])
        state.circuit.append("CX", [q[5], q[2]])
        state.circuit.append("CX", [q[5], q[0]])

        state.circuit.append("CX", [q[4], q[3]])
        state.circuit.append("CX", [q[4], q[2]])
        state.circuit.append("CX", [q[4], q[1]])

        # Invalidate cached tableau so it gets recomputed on next access
        state._tableau = None
        
        log.logger.debug(f"Encoded logical qubit using qubits {data_keys}")
    
    @staticmethod
    def decode(qm: 'QuantumManager', data_keys: List[int]) -> None:
        """Decode a [[7,1,3]] logical qubit back to 7 physical qubits.

        Args:
            qm: Quantum manager instance
            data_keys: List of 7 qubit keys to decode (same order as encoding)
        
        Raises:
            AssertionError: If not exactly 7 qubits provided
        """
        assert len(data_keys) == 7, "Need exactly 7 qubits for [[7,1,3]] decoding"

        state = qm.states[data_keys[0]]
        q = data_keys

        # Apply decoding circuit (reverse of encoding)
        state.circuit.append("CX", [q[4], q[1]])
        state.circuit.append("CX", [q[4], q[2]])
        state.circuit.append("CX", [q[4], q[3]])

        state.circuit.append("CX", [q[5], q[0]])
        state.circuit.append("CX", [q[5], q[2]])
        state.circuit.append("CX", [q[5], q[3]])

        state.circuit.append("CX", [q[6], q[0]])
        state.circuit.append("CX", [q[6], q[1]])
        state.circuit.append("CX", [q[6], q[3]])

        state.circuit.append("CX", [q[0], q[2]])
        state.circuit.append("CX", [q[0], q[1]])

        state.circuit.append("H", [q[4], q[5], q[6]])

        # Invalidate cached tableau
        state._tableau = None
        
        log.logger.debug(f"Decoded logical qubit from qubits {data_keys}")
    
    @staticmethod
    def measure_stabilizers(qm: 'QuantumManager', data_keys: List[int], 
                           ancilla_keys: List[int]) -> Tuple[List[int], List[int]]:
        """Measure stabilizers to detect errors.
        
        Args:
            qm: Quantum manager instance
            data_keys: List of 7 data qubit keys
            ancilla_keys: List of 6 ancilla qubit keys
            
        Returns:
            Tuple of (bit_flip_syndromes, phase_flip_syndromes)
        
        Raises:
            AssertionError: If wrong number of qubits provided
        """
        assert len(data_keys) == 7, "Need 7 data qubits"
        assert len(ancilla_keys) == 6, "Need 6 ancilla qubits"
        
        # Group all qubits together if not already
        all_keys = data_keys + ancilla_keys
        qm.group_qubits(all_keys)
        state = qm.states[data_keys[0]]
        
        d = data_keys
        a = ancilla_keys
        
        # Bit-flip syndrome measurements (X-type stabilizers)
        state.circuit.append("CX", [d[0], a[0]])
        state.circuit.append("CX", [d[2], a[0]])
        state.circuit.append("CX", [d[4], a[0]])
        state.circuit.append("CX", [d[6], a[0]])
        
        state.circuit.append("CX", [d[1], a[1]])
        state.circuit.append("CX", [d[2], a[1]])
        state.circuit.append("CX", [d[5], a[1]])
        state.circuit.append("CX", [d[6], a[1]])
        
        state.circuit.append("CX", [d[3], a[2]])
        state.circuit.append("CX", [d[4], a[2]])
        state.circuit.append("CX", [d[5], a[2]])
        state.circuit.append("CX", [d[6], a[2]])
        
        # Phase-flip syndrome measurements (Z-type stabilizers)
        state.circuit.append("H", [a[3], a[4], a[5]])
        
        state.circuit.append("CX", [a[3], d[0]])
        state.circuit.append("CX", [a[3], d[2]])
        state.circuit.append("CX", [a[3], d[4]])
        state.circuit.append("CX", [a[3], d[6]])
        
        state.circuit.append("CX", [a[4], d[1]])
        state.circuit.append("CX", [a[4], d[2]])
        state.circuit.append("CX", [a[4], d[5]])
        state.circuit.append("CX", [a[4], d[6]])
        
        state.circuit.append("CX", [a[5], d[3]])
        state.circuit.append("CX", [a[5], d[4]])
        state.circuit.append("CX", [a[5], d[5]])
        state.circuit.append("CX", [a[5], d[6]])
        
        state.circuit.append("H", [a[3], a[4], a[5]])

        # Measure ancillas
        state.circuit.append("M", a)
        
        # Sample to get syndrome values
        sampler = state.circuit.compile_sampler()
        sample = sampler.sample(shots=1)[0]
        
        # Extract the last 6 measurements
        bit_syndromes = [int(sample[-6+i]) for i in range(3)]
        phase_syndromes = [int(sample[-3+i]) for i in range(3)]

        # Invalidate cached tableau
        state._tableau = None
        
        log.logger.debug(f"Measured syndromes - Bit: {bit_syndromes}, Phase: {phase_syndromes}")

        return bit_syndromes, phase_syndromes
    
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
    
    @staticmethod
    def apply_corrections(qm: 'QuantumManager', data_keys: List[int], 
                         corrections: Dict[int, str]) -> None:
        """Apply error corrections to the logical qubit.
        
        Args:
            qm: Quantum manager instance
            data_keys: List of 7 data qubit keys
            corrections: Dict mapping qubit_index -> correction_type
        """
        if not corrections:
            return
            
        for qubit_idx, correction_type in corrections.items():
            if qubit_idx >= len(data_keys):
                log.logger.warning(f"Invalid qubit index {qubit_idx} for correction")
                continue
                
            qubit_key = data_keys[qubit_idx]
            
            if hasattr(qm, 'run_circuit'):
                # Use Circuit interface if available
                rnd = 0.5  # Default random number
                if correction_type == 'X':
                    qm.run_circuit(QEC713._x_correction, [qubit_key], rnd)
                elif correction_type == 'Z':
                    qm.run_circuit(QEC713._z_correction, [qubit_key], rnd)
                elif correction_type == 'Y':
                    # Y = XZ
                    qm.run_circuit(QEC713._x_correction, [qubit_key], rnd)
                    qm.run_circuit(QEC713._z_correction, [qubit_key], rnd)
            else:
                # Use Stim directly
                state = qm.states[qubit_key]
                if correction_type == 'X':
                    state.circuit.append("X", [qubit_key])
                elif correction_type == 'Z':
                    state.circuit.append("Z", [qubit_key])
                elif correction_type == 'Y':
                    state.circuit.append("Y", [qubit_key])
                state._tableau = None
            
            log.logger.debug(f"Applied {correction_type} correction to qubit {qubit_idx} (key {qubit_key})")
    
    @staticmethod
    def apply_logical_gate(qm: 'QuantumManager', data_keys: List[int], gate_type: str) -> None:
        """Apply logical gate operation to the encoded logical qubit.
        
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
        
        # All these gates are transversal for Steane code - apply to all physical qubits
        for qubit_key in data_keys:
            if hasattr(qm, 'run_circuit'):
                rnd = 0.5
                if gate_type == 'X':
                    qm.run_circuit(QEC713._x_correction, [qubit_key], rnd)
                elif gate_type == 'Z':
                    qm.run_circuit(QEC713._z_correction, [qubit_key], rnd)
                elif gate_type == 'Y':
                    qm.run_circuit(QEC713._x_correction, [qubit_key], rnd)
                    qm.run_circuit(QEC713._z_correction, [qubit_key], rnd)
                elif gate_type == 'H':
                    # Create Hadamard circuit
                    h_circuit = Circuit(1)
                    h_circuit.h(0)
                    qm.run_circuit(h_circuit, [qubit_key], rnd)
            else:
                state = qm.states[qubit_key]
                state.circuit.append(gate_type, [qubit_key])
                state._tableau = None
        
        log.logger.debug(f"Applied transversal logical {gate_type} gate to logical qubit")
    
    @staticmethod
    def apply_logical_cnot(qm: 'QuantumManager', control_keys: List[int], 
                          target_keys: List[int]) -> None:
        """Apply logical CNOT between two encoded logical qubits.
        
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
        
        # Apply CNOT between corresponding pairs of physical qubits
        for i in range(7):
            if hasattr(qm, 'run_circuit'):
                cnot_circuit = Circuit(2)
                cnot_circuit.cx(0, 1)
                rnd = 0.5
                qm.run_circuit(cnot_circuit, [control_keys[i], target_keys[i]], rnd)
            else:
                # Group qubits if needed for Stim
                qm.group_qubits([control_keys[i], target_keys[i]])
                state = qm.states[control_keys[i]]
                state.circuit.append("CX", [control_keys[i], target_keys[i]])
                state._tableau = None
        
        log.logger.debug(f"Applied transversal logical CNOT between logical qubits")
    
    @staticmethod
    def perform_round(qm: 'QuantumManager', data_keys: List[int], 
                     ancilla_keys: List[int]) -> bool:
        """Perform a complete QEC round: measure, decode, and correct.
        
        Args:
            qm: Quantum manager instance
            data_keys: List of 7 data qubit keys
            ancilla_keys: List of 6 ancilla qubit keys
            
        Returns:
            True if corrections were applied, False if no errors detected
        """
        # Measure stabilizers
        bit_syndromes, phase_syndromes = QEC713.measure_stabilizers(qm, data_keys, ancilla_keys)
        
        # Decode syndromes to find errors
        corrections = QEC713.decode_syndrome(bit_syndromes, phase_syndromes)
        
        # Apply corrections if needed
        if corrections:
            QEC713.apply_corrections(qm, data_keys, corrections)
            return True
        
        return False