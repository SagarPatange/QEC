from typing import List, Dict, Any
import numpy as np
import stim
from sequence.kernel.quantum_state import State
# from sequence.stabalizer_circuit import StabilizerCircuit
from sequence.kernel.quantum_manager import QuantumManager
from drafts.stabalizer_circuit import StabilizerCircuit

class StimStabilizerState(State):
    """
    Wrap a StabilizerCircuit with explicit qubit key labels.
    Keys length must equal circuit.num_qubits.
    """
    def __init__(self, keys: List[int], circuit: StabilizerCircuit):
        if len(keys) != circuit.num_qubits:
            raise ValueError(f"keys length {len(keys)} != circuit.num_qubits {circuit.num_qubits}")
        self.keys = keys
        self.circuit = circuit

    def __repr__(self) -> str:
        return f"<StimStabilizerState keys={self.keys} gates={len(self.circuit.circuit)}>"

    def state_vector(self) -> np.ndarray:
        """Simulate full statevector on demand via Stim."""
        return self.circuit.state_vector()

class QuantumManagerStabilizer(QuantumManager):
    """
    Stores and evolves StimStabilizerState objects under a shared key mapping.
    """
    def __init__(self, keys: List[int], base_circuit: StabilizerCircuit):
        super().__init__(formalism="stim_stabilizer")
        if len(keys) != base_circuit.num_qubits:
            raise ValueError("keys length must match base_circuit.num_qubits")
        self.keys = keys
        self.base_circuit = base_circuit

    def new(self) -> int:
        """Create a new state by cloning the base circuit."""
        idx = self._least_available
        self._least_available += 1
        circ_copy = StabilizerCircuit(self.base_circuit.num_qubits)
        circ_copy.circuit = self.base_circuit.circuit.copy()
        self.states[idx] = StimStabilizerState(self.keys, circ_copy)
        return idx

    def run_circuit(self, circuit: StabilizerCircuit, indices: List[int], meas_samp: Any = None) -> Dict:
        """Append operations to each state's circuit."""
        if circuit.num_qubits != len(self.keys):
            raise ValueError("Circuit qubits != keys length")
        for i in indices:
            self.states[i].circuit.circuit += circuit.circuit
        return {}

    def set(self, indices: List[int], circuit: StabilizerCircuit) -> None:
        """Replace each state's circuit entirely."""
        if circuit.num_qubits != len(self.keys):
            raise ValueError("Circuit qubits != keys length")
        for i in indices:
            self.states[i].circuit = circuit

    def get_state(self, idx: int) -> StimStabilizerState:
        """Direct access to a stored state."""
        return self.states[idx]

    def state_vector(self, idx: int) -> np.ndarray:
        """Simulate and return the statevector of a stored state."""
        return self.get_state(idx).state_vector()

    def remove(self, idx: int) -> None:
        """Delete the stored state."""
        del self.states[idx]
