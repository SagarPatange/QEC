"""
=====================================================================
enhanced_stabilizer_circuit.py
---------------------------------------------------------------------
Extended StabilizerCircuit wrapper with full functionality for
[[7,1,3]] quantum error correction and entanglement purification.
=====================================================================
"""

from sequence.utils import log
import stim
import numpy as np
import itertools
from typing import List, Union, Tuple, Optional, Dict

# Import the original StabilizerCircuit and extend it
from stabalizer_circuit import StabilizerCircuit as BaseStabilizerCircuit


class EnhancedStabilizerCircuit(BaseStabilizerCircuit):
    """
    Extended stabilizer circuit with additional methods for QEC protocols.
    Inherits all functionality from base StabilizerCircuit and adds:
    - Support for conditional operations with rec targets
    - CZ gate
    - S_DAG gate  
    - Identity operation
    - Better measurement tracking
    """
    
    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        self._measurement_count = 0
        
    def i(self, qubit: Union[int, List[int]]) -> None:
        """Apply identity gate (no-op, but useful for initialization)."""
        qubits = [qubit] if isinstance(qubit, int) else qubit
        for q in qubits:
            self._validate_qubit(q)
        self.circuit.append("I", qubits)
        log.logger.debug(f"Applied I on qubits {qubits}")
    
    def s_dag(self, qubit: int) -> None:
        """Apply S† (S dagger) gate."""
        self._validate_qubit(qubit)
        self.circuit.append("S_DAG", [qubit])
        log.logger.debug(f"Applied S_DAG on qubit {qubit}")
    
    def cx_conditional(self, control_record: int, target_qubit: int) -> None:
        """
        Apply CX controlled by a measurement record.
        
        Args:
            control_record: Measurement record offset (negative for recent)
            target_qubit: Target qubit index
        """
        self._validate_qubit(target_qubit)
        self.circuit.append("CX", [stim.target_rec(control_record), target_qubit])
        log.logger.debug(f"Applied conditional CX with rec({control_record}) on qubit {target_qubit}")
    
    def cz_conditional(self, control_record: int, target_qubit: int) -> None:
        """
        Apply CZ controlled by a measurement record.
        
        Args:
            control_record: Measurement record offset (negative for recent)
            target_qubit: Target qubit index
        """
        self._validate_qubit(target_qubit)
        self.circuit.append("CZ", [stim.target_rec(control_record), target_qubit])
        log.logger.debug(f"Applied conditional CZ with rec({control_record}) on qubit {target_qubit}")
    
    def measure_batch(self, qubits: List[int], basis: str = 'Z') -> None:
        """
        Measure multiple qubits at once in specified basis.
        
        Args:
            qubits: List of qubit indices to measure
            basis: Measurement basis ('Z', 'X', or 'Y')
        """
        for q in qubits:
            self._validate_qubit(q)
        
        if basis.upper() == 'X':
            for q in qubits:
                self.circuit.append("H", [q])
            log.logger.debug(f"Rotated qubits {qubits} to X basis")
        elif basis.upper() == 'Y':
            for q in qubits:
                self.circuit.append("S_DAG", [q])
                self.circuit.append("H", [q])
            log.logger.debug(f"Rotated qubits {qubits} to Y basis")
        
        self.circuit.append("M", qubits)
        self._measured_qubits.extend(qubits)
        self._measurement_count += len(qubits)
        log.logger.debug(f"Measured qubits {qubits} in {basis.upper()} basis")
    
    def get_measurement_count(self) -> int:
        """Return total number of measurements in the circuit."""
        return self._measurement_count
    
    def sample(self, shots: int = 1) -> np.ndarray:
        """
        Sample measurement outcomes from the circuit.
        
        Args:
            shots: Number of samples to take
            
        Returns:
            Array of shape (shots, num_measurements) with measurement outcomes
        """
        sampler = self.circuit.compile_sampler()
        return sampler.sample(shots=shots)
    
    def copy(self) -> 'EnhancedStabilizerCircuit':
        """Create a deep copy of the circuit."""
        new_circuit = EnhancedStabilizerCircuit(self.num_qubits)
        new_circuit.circuit = self.circuit.copy()
        new_circuit._measured_qubits = list(self._measured_qubits)
        new_circuit._measurement_count = self._measurement_count
        new_circuit._key2q = dict(self._key2q)
        return new_circuit
    
    def append_circuit(self, other: 'EnhancedStabilizerCircuit') -> None:
        """Append another circuit to this one."""
        self.circuit += other.circuit
        self._measured_qubits.extend(other._measured_qubits)
        self._measurement_count += other._measurement_count
    
    def reset(self) -> None:
        """Reset the circuit to empty state."""
        self.circuit = stim.Circuit()
        self._measured_qubits = []
        self._measurement_count = 0
        self._key2q = {}