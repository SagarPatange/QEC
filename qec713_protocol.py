"""
QEC 713 Protocol - [[7,1,3]] Steane Code Implementation

This module provides a simple class for applying the [[7,1,3]] quantum error correction code.
The protocol operates locally on each node and protects distributed entanglement from single-qubit errors.
"""

from typing import TYPE_CHECKING, List
import stim

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
    
    def __init__(self):
        """Constructor for QEC713 utility class."""
        pass
    
    @staticmethod
    def encode(qm: 'QuantumManager', data_keys: List[int]) -> None:
        """Encode 7 physical qubits into a [[7,1,3]] logical qubit.

        Args:
            qm: Quantum manager instance (must be QuantumManagerStabilizer)
            data_keys: List of 7 qubit keys to encode (order matters!)
        """
        assert len(data_keys) == 7, "Need exactly 7 qubits for [[7,1,3]] encoding"

        # Get the state (all qubits should be in same grouped state)
        state = qm.states[data_keys[0]]

        # DO NOT group qubits - this is expensive and unnecessary
        # The stabilizer formalism can handle operations on individual qubits efficiently

        # Apply encoding circuit
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

        # DO NOT compute density matrix here - stabilizer formalism is efficient!
        # The density matrix will be computed lazily only when needed (e.g., for measurement)
    
    @staticmethod
    def decode(qm: 'QuantumManager', data_keys: List[int]) -> None:
        """Decode a [[7,1,3]] logical qubit back to 7 physical qubits.

        Args:
            qm: Quantum manager instance
            data_keys: List of 7 qubit keys to decode (same order as encoding)
        """
        assert len(data_keys) == 7, "Need exactly 7 qubits"

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

        # DO NOT compute density matrix - use lazy evaluation
    
    @staticmethod
    def measure_stabilizers(qm: 'QuantumManager', data_keys: List[int], 
                           ancilla_keys: List[int]) -> tuple:
        """Measure stabilizers to detect errors.
        
        Args:
            qm: Quantum manager instance
            data_keys: List of 7 data qubit keys
            ancilla_keys: List of 6 ancilla qubit keys
            
        Returns:
            Tuple of (bit_flip_syndromes, phase_flip_syndromes)
        """
        assert len(data_keys) == 7, "Need 7 data qubits"
        assert len(ancilla_keys) == 6, "Need 6 ancilla qubits"
        
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

        # Invalidate cached tableau
        state._tableau = None

        # DO NOT compute density matrix - measurements are handled by the quantum manager

        # Return syndromes (would need actual measurement results in real implementation)
        # This is simplified - actual implementation would extract measurement outcomes
        return ([0, 0, 0], [0, 0, 0])  # Placeholder