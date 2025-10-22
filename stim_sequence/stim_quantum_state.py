"""
=====================================================================
stim_quantum_state.py
---------------------------------------------------------------------
Wrapper around StabilizerCircuit with key-to-qubit mapping.
=====================================================================
"""

from __future__ import annotations
from typing import Dict, List, Optional, Union, Tuple
import logging

from drafts.stabalizer_circuit import StabilizerCircuit


class StimQuantumState:
    """
    Wraps a StabilizerCircuit and provides key-based access.
    Keys (memory slots) are dynamically mapped to circuit qubits.
    """
    
    def __init__(self, num_qubits: int = 100):
        """
        Initialize quantum state with a stabilizer circuit.
        
        Args:
            num_qubits: Total qubits in the circuit (should be >= num_memories)
        """
        self.circuit = StabilizerCircuit(num_qubits)
        self._key_to_qubit: Dict[Union[int, str], int] = {}
        self._qubit_to_key: Dict[int, Union[int, str]] = {}
        self._next_free_qubit = 0
        self.num_qubits = num_qubits
        
    def bind_key(self, key: Union[int, str], qubit_index: Optional[int] = None) -> int:
        """
        Associate a memory key with a circuit qubit.
        
        Args:
            key: Memory key/identifier
            qubit_index: Specific qubit to use (optional, auto-assigns if None)
            
        Returns:
            The qubit index bound to this key
        """
        if key in self._key_to_qubit:
            return self._key_to_qubit[key]
        
        if qubit_index is None:
            # Auto-assign next available qubit
            qubit_index = self._next_free_qubit
            self._next_free_qubit += 1
            if self._next_free_qubit > self.num_qubits:
                raise ValueError(f"Ran out of qubits (max {self.num_qubits})")
        
        # Bind both directions
        self._key_to_qubit[key] = qubit_index
        self._qubit_to_key[qubit_index] = key
        
        # Also update the circuit's internal mapping
        self.circuit.bind_key(key, qubit_index)
        
        logging.debug(f"Bound key {key} to qubit {qubit_index}")
        return qubit_index
    
    def unbind_key(self, key: Union[int, str]) -> None:
        """Remove the key-to-qubit mapping."""
        if key in self._key_to_qubit:
            qubit = self._key_to_qubit.pop(key)
            self._qubit_to_key.pop(qubit, None)
            self.circuit.unbind_key(key)
            logging.debug(f"Unbound key {key} from qubit {qubit}")
    
    def qubit_for(self, key: Union[int, str]) -> int:
        """Get the qubit index for a given key."""
        if key not in self._key_to_qubit:
            # Auto-bind if not yet bound
            return self.bind_key(key)
        return self._key_to_qubit[key]
    
    def measure_keys(
        self, 
        keys: List[Union[int, str]], 
        basis: str = 'Z'
    ) -> Dict[Union[int, str], int]:
        """
        Measure multiple keys and return results.
        
        Args:
            keys: List of memory keys to measure
            basis: Measurement basis ('Z' or 'X')
            
        Returns:
            Dictionary mapping keys to measurement results (0 or 1)
        """
        results = {}
        for key in keys:
            qubit = self.qubit_for(key)
            self.circuit.measure(qubit, basis)
            # For MVP, simulate deterministic outcomes
            # In real implementation, would sample from circuit
            results[key] = 0  # Placeholder
        
        logging.debug(f"Measured keys {keys} in {basis} basis")
        return results
    
    def apply_pauli_channel_on_keys(
        self,
        keys: List[Union[int, str]],
        px: float = 0.0,
        pz: float = 0.0
    ) -> None:
        """
        Apply Pauli noise to specific keys.
        
        Args:
            keys: Memory keys to apply noise to
            px: X error probability
            pz: Z error probability
        """
        for key in keys:
            qubit = self.qubit_for(key)
            if px > 0 or pz > 0:
                # Use the circuit's pauli_channel method
                # For single qubit: provide (px, py, pz) tuple
                py = 0.0  # No Y errors for simplicity
                self.circuit.pauli_channel(qubit, (px, py, pz))
                logging.debug(f"Applied Pauli channel to key {key}: px={px}, pz={pz}")
    
    # --- Protocol operations ---
    
    def prepare_bell_pair(
        self, 
        key1: Union[int, str], 
        key2: Union[int, str]
    ) -> None:
        """
        Create a Bell pair between two keys.
        Standard |Φ+⟩ = (|00⟩ + |11⟩)/√2
        """
        q1 = self.qubit_for(key1)
        q2 = self.qubit_for(key2)
        
        # |00⟩ -> |0+⟩ -> |Φ+⟩
        self.circuit.h(q1)
        self.circuit.cx(q1, q2)
        
        logging.debug(f"Created Bell pair between keys {key1} and {key2}")
    
    def apply_swap(
        self,
        control_key: Union[int, str],
        target_key: Union[int, str]
    ) -> Tuple[int, int]:
        """
        Apply entanglement swapping measurement (Bell measurement).
        
        Returns:
            Tuple of measurement results (m1, m2)
        """
        q1 = self.qubit_for(control_key)
        q2 = self.qubit_for(target_key)
        
        # Bell measurement circuit
        self.circuit.cx(q1, q2)
        self.circuit.h(q1)
        self.circuit.measure(q1, 'Z')
        self.circuit.measure(q2, 'Z')
        
        # For MVP, return deterministic results
        # Real implementation would sample
        return (0, 0)
    
    def reset_key(self, key: Union[int, str]) -> None:
        """Reset a key's qubit to |0⟩ state (for MVP, just unbind)."""
        self.unbind_key(key)