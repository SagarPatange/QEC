from typing import Any, Dict, List, Optional
import numpy as np
from stabalizer_circuit import StabilizerCircuit

class StabilizerManager:
    """
    A thin wrapper around StabilizerCircuit that mimics the QuantumManager API.

    Attributes:
      num_qubits (int): Total number of qubits.
      sc (StabilizerCircuit): Underlying Stim-backed stabilizer circuit.
      nodes (Dict[Any,List[int]]): Logical node→qubit mapping.
    """

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.sc = StabilizerCircuit(num_qubits)
        self.nodes: Dict[Any, List[int]] = {}

    def add_node(self, node_id: Any, qubits: List[int]) -> None:
        for q in qubits:
            self.sc._validate_qubit(q)
        self.nodes[node_id] = qubits

    @property
    def circuit(self):
        return self.sc.circuit

    @property
    def measured(self) -> List[int]:
        return self.sc.measured

    def sample(self, shots: int = 1) -> np.ndarray:
        """
        Append a Stim circuit (gates, noise, measurements) and sample it.

        Returns an array of shape (shots, num_measurements) of raw bits.
        """
        # compile & sample
        sampler = self.sc.circuit.compile_sampler()
        return sampler.sample(shots=shots)

    def reset(self) -> None:
        """
        Clear out the current circuit and measurement history,
        but keep the same num_qubits and nodes mapping.
        """
        self.sc = StabilizerCircuit(self.num_qubits)

    def statevector(self) -> np.ndarray:
        """Extract the exact stabilizer statevector (ignores measurements)."""
        return self.sc.state_vector()

    def sample_density_matrix( self, qubits: Optional[List[int]] = None, shots: int = 1_000) -> np.ndarray:
        """
        Reconstruct the reduced density matrix via Pauli tomography
        on a specified subset of qubits.

        Args:
          qubits: List of qubit indices to reconstruct. If None,
                  defaults to all qubits [0..num_qubits-1].
          shots:   Number of samples per Pauli setting.

        Returns:
          A (2^k x 2^k) complex numpy array for the specified qubits.
        """
        if qubits is None:
            qubits = list(range(self.num_qubits))
        # delegate to the StabilizerCircuit helper
        return self.sc.tomography_dm(qubits, shots=shots)
