from sequence.utils import log
import stim
import numpy as np
import itertools
from typing import List, Union, Tuple, Optional


class StabilizerCircuitError(Exception):
    """Raised when an invalid operation is attempted on a StabilizerCircuit."""
    pass


class StabilizerCircuit:
    """
    Stim-backed stabilizer circuit abstraction.

    Implements:
      - Clifford gates with optional Pauli error channels
      - Basis-specific measurements (Z and X)
      - Measurement-conditioned operations
      - Single- or two-qubit Pauli and depolarizing channels
      - Full Pauli tomography for density matrix reconstruction
      - Idle-depolarization based on T1/T2 idling

    Attributes:
      num_qubits (int): Number of qubits in the circuit.
      circuit (stim.Circuit): Underlying Stim circuit instance.
      _measured_qubits (List[int]): Recorded measured qubit indices.

    Public methods:
      x, y, z, h, cx, cz,
      measure, x_if, y_if, z_if,
      pauli_channel, depolarize, idle,
      conditional, rec,
      tomography_dm, state_vector
    """

    def __init__(self, num_qubits: int):
        """
        Initialize a StabilizerCircuit.

        Args:
          num_qubits: Total number of qubits (must be ≥0).
        Returns:
          None
        """
        if num_qubits <= 0:
            log.logger.warning(f"Initializing with non-positive qubit count: {num_qubits}")
        self.num_qubits = num_qubits
        self.circuit = stim.Circuit()
        self._measured_qubits: List[int] = []
        log.logger.debug(f"Initialized StabilizerCircuit with {self.num_qubits} qubits")

    def _validate_qubit(self, qubit: Union[int, stim.GateTarget]) -> None:
        if isinstance(qubit, stim.GateTarget):
            return
        if not (isinstance(qubit, int) and 0 <= qubit < self.num_qubits):
            raise StabilizerCircuitError(f"Qubit index {qubit} out of range [0,{self.num_qubits})")

    def _validate_probability(self, p: float) -> None:
        if not isinstance(p, (int, float)):
            raise StabilizerCircuitError(f"Probability {p!r} is not numeric")
        if not (0 <= p <= 1):
            raise StabilizerCircuitError(f"Probability {p} out of [0,1]")

    def rec(self, offset: int = -1) -> stim.GateTarget:
        """Return a measurement-record target for conditional gates."""
        return stim.target_rec(offset)

    def conditional(self, gate: str, rec_offset: int, *targets: Union[int, stim.GateTarget]) -> None:
        """Append a gate conditioned on a prior measurement outcome."""
        for q in targets:
            self._validate_qubit(q)
        self.circuit.append(gate, [stim.target_rec(rec_offset), *targets])
        log.logger.debug(f"Applied conditional {gate} with rec_offset {rec_offset} on {targets}")

    def x(self, qubit: int, error_prob: float = 0.0) -> None:
        """Apply Pauli-X; optionally inject X_ERROR channel."""
        self._validate_qubit(qubit)
        self._validate_probability(error_prob)
        self.circuit.append("X", [qubit])
        log.logger.debug(f"Applied X on qubit {qubit}")
        if error_prob:
            self.circuit.append("X_ERROR", [qubit], arg_values=[error_prob])
            log.logger.debug(f"Injected X_ERROR (p={error_prob}) on qubit {qubit}")

    def y(self, qubit: int, error_prob: float = 0.0) -> None:
        """Apply Pauli-Y; optionally inject Y_ERROR channel."""
        self._validate_qubit(qubit)
        self._validate_probability(error_prob)
        self.circuit.append("Y", [qubit])
        log.logger.debug(f"Applied Y on qubit {qubit}")
        if error_prob:
            self.circuit.append("Y_ERROR", [qubit], arg_values=[error_prob])
            log.logger.debug(f"Injected Y_ERROR (p={error_prob}) on qubit {qubit}")

    def z(self, qubit: int, error_prob: float = 0.0) -> None:
        """Apply Pauli-Z; optionally inject Z_ERROR channel."""
        self._validate_qubit(qubit)
        self._validate_probability(error_prob)
        self.circuit.append("Z", [qubit])
        log.logger.debug(f"Applied Z on qubit {qubit}")
        if error_prob:
            self.circuit.append("Z_ERROR", [qubit], arg_values=[error_prob])
            log.logger.debug(f"Injected Z_ERROR (p={error_prob}) on qubit {qubit}")

    def h(self, qubit: int) -> None:
        """Apply Hadamard on a qubit."""
        self._validate_qubit(qubit)
        self.circuit.append("H", [qubit])
        log.logger.debug(f"Applied H on qubit {qubit}")

    def cx(self, control: int, target: int) -> None:
        """Apply CNOT from control to target."""
        self._validate_qubit(control)
        self._validate_qubit(target)
        self.circuit.append("CNOT", [control, target])
        log.logger.debug(f"Applied CNOT {control}->{target}")

    def cz(self, control: int, target: int) -> None:
        """Apply CZ from control to target."""
        self._validate_qubit(control)
        self._validate_qubit(target)
        self.circuit.append("CZ", [control, target])
        log.logger.debug(f"Applied CZ {control}->{target}")

    def x_if(self, rec_offset: int, qubit: int) -> None:
        """Conditionally apply X based on a prior measurement."""
        self._validate_qubit(qubit)
        self.conditional("X", rec_offset, qubit)

    def y_if(self, rec_offset: int, qubit: int) -> None:
        """Conditionally apply Y based on a prior measurement."""
        self._validate_qubit(qubit)
        self.conditional("Y", rec_offset, qubit)

    def z_if(self, rec_offset: int, qubit: int) -> None:
        """Conditionally apply Z based on a prior measurement."""
        self._validate_qubit(qubit)
        self.conditional("Z", rec_offset, qubit)

    def measure(self, qubit: int, basis: str = 'Z') -> None:
        """
        Measure a qubit in the specified basis and record the result.

        Implements:
          - Z basis: direct 'M'
          - X basis: 'H' then 'M'

        Args:
          qubit: Qubit index to measure.
          basis: 'Z' or 'X'.
        Returns:
          None
        """
        self._validate_qubit(qubit)
        if basis.upper() == 'X':
            self.circuit.append("H", [qubit])
            log.logger.debug(f"Rotated qubit {qubit} to X basis for measurement")
        self.circuit.append("M", [qubit])
        self._measured_qubits.append(qubit)
        log.logger.debug(f"Measured qubit {qubit} in {basis.upper()} basis")

    @property
    def measured(self) -> List[int]:
        """Access measured qubit indices in order."""
        return list(self._measured_qubits)

    def depolarize(self, qubits: Union[int, List[int], Tuple[int, int]], p: float) -> None:
        """
        Apply a depolarizing channel on one or two qubits.

        Implements:
          - DEPOLARIZE1 for a single qubit
          - DEPOLARIZE2 for two qubits

        Args:
          qubits: int or iterable of 1 or 2 qubit indices.
          p: Depolarization probability in [0,1].
        Raises:
          StabilizerCircuitError: If count of qubits not 1 or 2 or p invalid.
        Returns:
          None
        """
        self._validate_probability(p)
        qs = [qubits] if isinstance(qubits, int) else list(qubits)
        if len(qs) not in (1, 2):
            raise StabilizerCircuitError(f"depolarize supports only 1 or 2 qubits, got {len(qs)}")
        for q in qs:
            self._validate_qubit(q)
        gate = "DEPOLARIZE1" if len(qs) == 1 else "DEPOLARIZE2"
        self.circuit.append(gate, qs, [p])
        log.logger.debug(f"Applied {gate} on qubits {qs} with p={p}")

    def idle(self, qubits: Union[int, List[int], Tuple[int, int]], time: float, T1: Optional[float] = None, T2: Optional[float] = None) -> None:
        """
        Depolarize idling-induced decoherence over a time interval.

        Implements:
          - For pure dephasing (T2): p_dep = (4/3)*(1 - exp(-time/T2))
          - For amplitude damping (T1): p_dep = (2 + γ - 2*sqrt(1-γ))/3, γ = 1 - exp(-time/T1)

        Args:
          qubits: int or iterable of 1 or 2 qubit indices.
          time: Idle duration.
          T1: Relaxation time constant (optional).
          T2: Dephasing time constant (optional).
        Raises:
          StabilizerCircuitError: If neither T1 nor T2 provided, or invalid params.
        Returns:
          None
        """
        if T2 is not None:
            p_deph = 1 - np.exp(-time / T2)
            p_dep = (4/3) * p_deph
        elif T1 is not None:
            gamma = 1 - np.exp(-time / T1)
            p_dep = (2 + gamma - 2 * np.sqrt(1 - gamma)) / 3
        else:
            raise StabilizerCircuitError("Provide at least T1 or T2 for idle depolarization.")
        self.depolarize(qubits, p_dep)
    
    def tomography_dm(self, qubits: Union[List[int], Tuple[int, ...]], shots: int = 10_000, use_direct_pauli_meas: bool = True) -> np.ndarray:
        """
        Reconstruct the reduced density matrix via full Pauli tomography.

        Args:
          qubits: Iterable of qubit indices to reconstruct (length k).
          shots: Number of samples per Pauli setting.
          use_direct_pauli_meas:
            True  → use MX/MY/M instructions
            False → rotate with S_DAG/H then M (Z).

        Returns:
          np.ndarray: (2^k × 2^k) complex density matrix.
        """
        PAULIS = ("I", "X", "Y", "Z")
        SIGMA = {
            "I": np.eye(2, dtype=complex),
            "X": np.array([[0, 1], [1, 0]], dtype=complex),
            "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
            "Z": np.array([[1, 0], [0, -1]], dtype=complex),
        }

        def kron_all(mats):
            out = mats[0]
            for m in mats[1:]:
                out = np.kron(out, m)
            return out

        qs = tuple(qubits)
        k = len(qs)
        base_meas = self.circuit.num_measurements
        exp_vals = {}

        for setting in itertools.product(PAULIS, repeat=k):
            if all(p == "I" for p in setting):
                exp_vals[setting] = 1.0
                continue

            c = self.circuit.copy()
            measured = 0

            if use_direct_pauli_meas:
                for q, p in zip(qs, setting):
                    if p == "I": continue
                    gate = {"X": "MX", "Y": "MY", "Z": "M"}[p]
                    c.append(gate, [q])
                    measured += 1
            else:
                for q, p in zip(qs, setting):
                    if p == "I": continue
                    if p == "X":
                        c.append("H", [q])
                    elif p == "Y":
                        c.append("S_DAG", [q])
                        c.append("H", [q])
                    c.append("M", [q])
                    measured += 1

            sampler = c.compile_sampler()
            bits = sampler.sample(shots=shots)
            tomo = bits[:, base_meas : base_meas + measured]
            eig = 1 - 2 * tomo  # 0→+1, 1→-1
            exp_vals[setting] = (
                float(np.mean(np.prod(eig, axis=1))) if measured > 1 else float(np.mean(eig))
            )

        dim = 2**k
        rho = np.zeros((dim, dim), dtype=complex)
        for setting, val in exp_vals.items():
            mats = [SIGMA[p] for p in setting]
            rho += val * kron_all(mats)
        rho /= dim

        return rho

    def state_vector(self) -> np.ndarray:
        """Compute the exact stabilizer state vector for the circuit."""
        sim = stim.TableauSimulator()
        sim.do(self.circuit)
        vec = sim.state_vector()
        return np.asarray(vec, dtype=complex)

    def pauli_channel(self, qubits, probs):
        qs = [qubits] if isinstance(qubits, int) else list(qubits)
        if len(qs) not in (1, 2):
            raise StabilizerCircuitError(f"pauli_channel supports 1 or 2 qubits, got {len(qs)}")
        for q in qs:
            self._validate_qubit(q)
        gate = "PAULI_CHANNEL_1" if len(qs) == 1 else "PAULI_CHANNEL_2"
        # <-- pass the probability tuple as the `arg` parameter:
        self.circuit.append(gate, qs, tuple(probs))
        log.logger.debug(f"Applied {gate} on qubits {qs} with probs {probs}")