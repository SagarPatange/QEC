# Add below existing FORMALISM constants in quantum_manager.py
STABILIZER_FORMALISM = "stabilizer"

from stabalizer_circuit import StabilizerCircuit
from sequence.kernel.quantum_state import StabilizerState
from sequence.kernel.quantum_manager import QuantumManager
from sequence.kernel.quantum_state import State

class StabilizerState(State):
    """
    Class to represent an n-qubit stabilizer state via a Stim-based circuit.

    Attributes:
        state (StabilizerCircuit):
            Underlying circuit that encodes this stabilizer state.
        keys (list[int]):
            Labels for each qubit (length == num_qubits).
        num_qubits (int):
            Number of qubits in the state.
    """

    def __init__(self, num_qubits: int, keys: list[int] | None = None):
        """
        Args:
            num_qubits (int): number of qubits to allocate.
            keys (list[int], optional): labels for each qubit; defaults to [0,1,…,num_qubits-1].
        """
        super().__init__()
        self.num_qubits = num_qubits
        self.keys = list(keys) if keys is not None else list(range(num_qubits))

        # The Stim-based circuit storing all Clifford gates, measurements, noise, etc.
        self.state = StabilizerCircuit(num_qubits)


class QuantumManagerStabilizer(QuantumManager):
    """Class to track and manage states with the stabilizer formalism."""

    def __init__(self):
        super().__init__(STABILIZER_FORMALISM)

    def new(self,
            num_qubits: int,
            keys: list[int] | None = None) -> int:
        """
        Allocate a fresh stabilizer state on `num_qubits` qubits.

        Args:
            num_qubits: number of qubits for the new state.
            keys: optional labels for each qubit; defaults to [key].
        Returns:
            key handle for the created state.
        """
        key = self._least_available
        self._least_available += 1
        lbls = list(keys) if keys is not None else [key]
        state = StabilizerState(num_qubits, lbls)
        self.states[key] = state
        return key

    def run_circuit(self,
                    circuit: StabilizerCircuit,
                    keys: list[int],
                    meas_samp=None) -> dict[int, bool]:
        """
        Apply a Stim-based stabilizer circuit to the state(s) referenced by `keys`.

        Returns a mapping from measured qubit label to outcome.
        """
        super().run_circuit(circuit, keys, meas_samp)
        # Retrieve the StabilizerState and attach the circuit
        state_obj = self.states[keys[0]]
        state_obj.state = circuit
        # Update all key mappings to this updated state
        for k in keys:
            self.states[k] = state_obj
        # Simulate and return measurement results
        return circuit.simulate()

    def set(self,
            keys: list[int],
            circuit: StabilizerCircuit) -> None:
        """
        Replace the state at `keys` with the given Stim-based circuit.
        """
        super().set(keys, circuit)
        st = StabilizerState(circuit.num_qubits, keys)
        st.state = circuit
        for k in keys:
            self.states[k] = st

    def set_to_zero(self, key: int):
        """Reset qubit `key` to |0>."""
        st = StabilizerState(1, [key])
        self.states[key] = st

    def set_to_one(self, key: int):
        """Reset qubit `key` to |1>."""
        st = StabilizerState(1, [key])
        st.state.x(0)
        self.states[key] = st

    def _measure(self,
                 state: StabilizerCircuit,
                 keys: list[int],
                 all_keys: list[int],
                 meas_samp: float) -> dict[int, bool]:
        """
        Hook for measurement; extract outcomes from Stim-based circuit.
        """
        return state.simulate()
