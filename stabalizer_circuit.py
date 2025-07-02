from sequence.utils import log
from typing import List, Dict, Union
import stim
import numpy as np


class StabilizerCircuitError(Exception):
    """Exception raised for errors in the StabilizerCircuit operations."""
    pass


class StabilizerCircuit: ## TODO: add monte carlon representation of density matrix and also add a tableu representation 
    """
    A stabilizer-formalism quantum circuit using Stim, with support for
    Clifford gates, measurements, high-performance sampling, and
    in-built Pauli noise channels.

    Attributes:
        num_qubits: Number of qubits in the circuit.
        circuit: Underlying stim.Circuit object.
        _measured_qubits: Ordered list of qubit indices measured.
    """

    def __init__(self, num_qubits: int):
        if num_qubits <= 0:
            raise StabilizerCircuitError(
                f"num_qubits must be positive, got {num_qubits}"
            )
        self.num_qubits: int = num_qubits
        self.circuit: stim.Circuit = stim.Circuit()
        self._measured_qubits: List[int] = []
        log.logger.debug(f"Initialized StabilizerCircuit with {self.num_qubits} qubits")

    def _validate_qubit(self, qubit: Union[int, stim.GateTarget]) -> None:
        """
        Accept either a raw qubit index or a stim.GateTarget (e.g. a rec-based control).
        """
        if isinstance(qubit, stim.GateTarget):
            return
        if not (isinstance(qubit, int) and 0 <= qubit < self.num_qubits):
            raise StabilizerCircuitError(
                f"Qubit index {qubit} out of range [0, {self.num_qubits})"
            )

    def _validate_probability(self, p: float) -> None:
        """
        Ensure p is a numeric probability in [0, 1].
        """
        if not isinstance(p, (int, float)):
            raise StabilizerCircuitError(f"Probability {p!r} is not a number")
        if not (0 <= p <= 1):
            raise StabilizerCircuitError(f"Probability {p} out of range [0, 1]")

    def rec(self, offset: int = -1) -> stim.GateTarget:
        """
        Get a GateTarget referring to a past measurement result.
        offset = -1 → most recent, -2 → one before, etc.
        """
        return stim.target_rec(offset)

    def conditional(self, gate: str, rec_offset: int, *targets: Union[int, stim.GateTarget]) -> None:
        """
        Apply any Clifford `gate` controlled by a measurement record.
        E.g. conditional('CNOT', -1, 3) is an X on qubit 3 if meas[-1] == 1.
        """
        for q in targets:
            self._validate_qubit(q)
        self.circuit.append(gate, [stim.target_rec(rec_offset), *targets])
        log.logger.debug(f"Applied conditional {gate} with rec_offset {rec_offset} on targets {targets}")

    def x_if(self, rec_offset: int, target: int) -> None:
        """Alias for measurement-controlled X on `target`."""
        self.conditional('CNOT', rec_offset, target)

    def z_if(self, rec_offset: int, target: int) -> None:
        """Alias for measurement-controlled Z on `target`."""
        self.conditional('CZ', rec_offset, target)

    cx_if = x_if  # backward compatibility alias

    def density_matrix(self, qubits: List[int]) -> np.ndarray:
        """
        Compute the reduced density matrix rho for the specified qubits,
        including qubits that may already have been measured.

        Args:
            qubits (List[int]): Indices of the qubits whose joint density matrix
                                is desired, in the exact order given.

        Returns:
            np.ndarray: A (2^k x 2^k) complex array rho, where k = len(qubits).

        Raises:
            StabilizerCircuitError: If any qubit index is invalid.
        """
        # 1. Validate and log measured qubits
        for q in qubits:
            self._validate_qubit(q)
            if q in self._measured_qubits:
                log.logger.info(f"Qubit {q} was already measured; including its post-measurement state.")

        # 2. Strip out all measurement and record-based instructions
        lines = [
            line for line in str(self.circuit).splitlines()
            if not line.startswith("M ") and "rec" not in line
        ]
        clean_circuit = stim.Circuit("\n".join(lines))

        # 3. Run the stabilizer tableau to get the full state vector |ψ⟩
        sim = stim.TableauSimulator()
        sim.do_circuit(clean_circuit)
        psi = sim.state_vector()                  # shape (2**n,)

        # 4. Reshape into an n-dimensional tensor of shape (2,2,...,2)
        n = self.num_qubits
        psi_tensor = psi.reshape([2] * n)

        # 5. Permute axes so that our qubits-of-interest come first
        keep = qubits
        trace_out = [i for i in range(n) if i not in keep]
        perm = keep + trace_out
        psi_perm = np.transpose(psi_tensor, perm)

        # 6. Flatten into a (2^k) × (2^(n-k)) matrix
        k = len(keep)
        dim_keep = 2 ** k
        dim_trace = 2 ** (n - k)
        psi_mat = psi_perm.reshape((dim_keep, dim_trace))

        # 7. Form ρ = |ψ⟩⟨ψ| on the kept subsystem
        dm = psi_mat @ psi_mat.conj().T

        log.logger.debug(
            f"Computed density matrix for qubits {qubits} "
            f"of shape {dm.shape}"
        )
        return dm

    def state_vector(self) -> np.ndarray:
        """
        Simulate and return the full pure state vector for all qubits.

        Returns:
            np.ndarray: Complex state vector of shape (2**num_qubits,),
                        representing the quantum state |ψ⟩.

        Raises:
            StabilizerCircuitError: If the simulator encounters an error.
        """
        simulator = stim.TableauSimulator()
        simulator.do_circuit(self.circuit)  ## TODO: check the API
        vec = simulator.state_vector()
        log.logger.debug(f"Obtained full state vector of length {vec.shape[0]}")
        return vec

    # Single-qubit Clifford and identity gates
    def i(self, qubit: int) -> None:
        """
        Apply an identity (I) gate to a single qubit.

        Args:
            qubit (int): Index of the qubit to which the I gate is applied.

        Raises:
            StabilizerCircuitError: If `qubit` is out of range.
        """
        self._validate_qubit(qubit)
        self.circuit.append("I", [qubit])
        log.logger.debug(f"Applied I (identity) on qubit {qubit}")

    def h(self, qubit: int) -> None:
        """
        Apply a Hadamard (H) gate to a single qubit.

        Args:
            qubit (int): Index of the qubit to which the H gate is applied.

        Raises:
            StabilizerCircuitError: If `qubit` is out of range.
        """
        self._validate_qubit(qubit)
        self.circuit.append("H", [qubit])
        log.logger.debug(f"Applied H (Hadamard) on qubit {qubit}")

    def x(self, qubit: int) -> None:
        """
        Apply a Pauli-X (NOT) gate to a single qubit.

        Args:
            qubit (int): Index of the qubit to which the X gate is applied.

        Raises:
            StabilizerCircuitError: If `qubit` is out of range.
        """
        self._validate_qubit(qubit)
        self.circuit.append("X", [qubit])
        log.logger.debug(f"Applied X (Pauli-X) on qubit {qubit}")

    def y(self, qubit: int) -> None:
        """
        Apply a Pauli-Y gate to a single qubit.

        Args:
            qubit (int): Index of the qubit to which the Y gate is applied.

        Raises:
            StabilizerCircuitError: If `qubit` is out of range.
        """
        self._validate_qubit(qubit)
        self.circuit.append("Y", [qubit])
        log.logger.debug(f"Applied Y (Pauli-Y) on qubit {qubit}")

    def z(self, qubit: int) -> None:
        """
        Apply a Pauli-Z (phase-flip) gate to a single qubit.

        Args:
            qubit (int): Index of the qubit to which the Z gate is applied.

        Raises:
            StabilizerCircuitError: If `qubit` is out of range.
        """
        self._validate_qubit(qubit)
        self.circuit.append("Z", [qubit])
        log.logger.debug(f"Applied Z (Pauli-Z) on qubit {qubit}")

    def s(self, qubit: int) -> None:
        """
        Apply the S (phase) gate to a single qubit.

        Args:
            qubit (int): Index of the qubit to which the S gate is applied.

        Raises:
            StabilizerCircuitError: If `qubit` is out of range.
        """
        self._validate_qubit(qubit)
        self.circuit.append("S", [qubit])
        log.logger.debug(f"Applied S (phase) on qubit {qubit}")

    def sdg(self, qubit: int) -> None:
        """
        Apply the S† (phase-dagger) gate to a single qubit.
        Since Stim doesn’t expose SD or -S, we do S³ = S⁻¹.
        """
        self._validate_qubit(qubit)
        # inverse of S is S^3
        self.circuit.append("S", [qubit])
        self.circuit.append("S", [qubit])
        self.circuit.append("S", [qubit])
    # ──────────────────────────────────────────────────────────────────────────
    # Two-qubit Clifford gates
    def cx(self, control: int, target: int) -> None:
        """
        Apply a controlled-NOT (CNOT) gate.

        Args:
            control (int): Index of the control qubit.
            target (int): Index of the target qubit.

        Raises:
            StabilizerCircuitError: If any qubit index is invalid.
        """
        self._validate_qubit(control)
        self._validate_qubit(target)
        self.circuit.append("CNOT", [control, target])
        log.logger.debug(f"Applied CNOT from qubit {control} to qubit {target}")

    def cz(self, qubit1: int, qubit2: int) -> None:
        """
        Apply a controlled-Z (CZ) gate.

        Args:
            qubit1 (int): Index of the first qubit.
            qubit2 (int): Index of the second qubit.

        Raises:
            StabilizerCircuitError: If any qubit index is invalid.
        """
        self._validate_qubit(qubit1)
        self._validate_qubit(qubit2)
        self.circuit.append("CZ", [qubit1, qubit2])
        log.logger.debug(f"Applied CZ between qubit {qubit1} and qubit {qubit2}")

    def cy(self, qubit1: int, qubit2: int) -> None:
        """
        Apply a controlled-Y (CY) gate.

        Args:
            qubit1 (int): Index of the first qubit.
            qubit2 (int): Index of the second qubit.

        Raises:
            StabilizerCircuitError: If any qubit index is invalid.
        """
        self._validate_qubit(qubit1)
        self._validate_qubit(qubit2)
        self.circuit.append("CY", [qubit1, qubit2])
        log.logger.debug(f"Applied CY between qubit {qubit1} and qubit {qubit2}")

    def swap(self, qubit1: int, qubit2: int) -> None:
        """
        Swap the states of two qubits.

        Args:
            qubit1 (int): Index of the first qubit.
            qubit2 (int): Index of the second qubit.

        Raises:
            StabilizerCircuitError: If any qubit index is invalid.
        """
        self._validate_qubit(qubit1)
        self._validate_qubit(qubit2)
        self.circuit.append("SWAP", [qubit1, qubit2])
        log.logger.debug(f"Applied SWAP between qubit {qubit1} and qubit {qubit2}")

    # ──────────────────────────────────────────────────────────────────────────
    # Measurement
    def measure(self, qubit: int) -> None:
        """
        Measure a single qubit in the Z basis and record its outcome.

        Args:
            qubit (int): Index of the qubit to measure.

        Raises:
            StabilizerCircuitError: If qubit index is invalid.
        """
        self._validate_qubit(qubit)
        self.circuit.append("M", [qubit])
        self._measured_qubits.append(qubit)
        log.logger.debug(f"Measured qubit {qubit}")

    @property
    def measurement_qubits(self) -> List[int]:
        """
        Get the ordered list of qubits that have been measured.

        Returns:
            List[int]: Measured qubit indices in measurement order.
        """
        return list(self._measured_qubits)

    # ──────────────────────────────────────────────────────────────────────────
    # Pauli error / noise channels
    def x_error(self, qubit: int, p: float) -> None:
        """
        Apply a Pauli-X error channel on the specified qubit with probability p.
        """
        self._validate_qubit(qubit)
        self._validate_probability(p)
        self.circuit.append("X_ERROR", [qubit], p)
        log.logger.debug(f"Applied X_ERROR on qubit {qubit} with p={p}")

    def y_error(self, qubit: int, p: float) -> None:
        """
        Apply a Pauli-Y error channel on the specified qubit with probability p.
        """
        self._validate_qubit(qubit)
        self._validate_probability(p)
        self.circuit.append("Y_ERROR", [qubit], p)
        log.logger.debug(f"Applied Y_ERROR on qubit {qubit} with p={p}")

    def z_error(self, qubit: int, p: float) -> None:
        """
        Apply a Pauli-Z error channel on the specified qubit with probability p.
        """
        self._validate_qubit(qubit)
        self._validate_probability(p)
        self.circuit.append("Z_ERROR", [qubit], p)
        log.logger.debug(f"Applied Z_ERROR on qubit {qubit} with p={p}")

    def pauli_channel1(
        self, qubit: int, p_x: float, p_y: float, p_z: float
    ) -> None:
        """
        Apply a single-qubit Pauli channel.

        Args:
            qubit (int): Index of the qubit.
            p_x (float): Probability of X error.
            p_y (float): Probability of Y error.
            p_z (float): Probability of Z error.

        Raises:
            StabilizerCircuitError: On invalid qubit, p out of range, or sum(p_x,p_y,p_z)>1.
        """
        self._validate_qubit(qubit)
        for val in (p_x, p_y, p_z):
            self._validate_probability(val)
        if (p_x + p_y + p_z) > 1:
            raise StabilizerCircuitError(
                f"Sum of Pauli error probabilities > 1: {p_x+p_y+p_z}"  
            )
        self.circuit.append(
            "PAULI_CHANNEL_1", [qubit], arg_values=[p_x, p_y, p_z]
        )
        log.logger.debug(
            f"Applied PAULI_CHANNEL_1 on qubit {qubit} with p_x={p_x}, p_y={p_y}, p_z={p_z}"
        )

    def depolarize1(self, qubit: int, p: float) -> None:
        """
        Apply a single-qubit depolarizing channel to the specified qubit.

        Args:
            qubit (int): Index of the qubit to depolarize.
            p (float): Depolarizing probability in [0, 1].

        Raises:
            StabilizerCircuitError: If `qubit` is out of range or `p` is not in [0, 1].
        """
        self._validate_qubit(qubit)
        self._validate_probability(p)
        # pass p as the 'arg' not 'arg_values'
        self.circuit.append("DEPOLARIZE1", [qubit], p)
        log.logger.debug(f"Applied DEPOLARIZE1 on qubit {qubit} with p={p}")

    def depolarize2(self, qubit1: int, qubit2: int, p: float) -> None:
        """
        Apply a two-qubit depolarizing channel between the two specified qubits.

        Args:
            qubit1 (int): Index of the first qubit.
            qubit2 (int): Index of the second qubit.
            p (float): Depolarizing probability in [0, 1].

        Raises:
            StabilizerCircuitError: If either qubit index is out of range or `p` is not in [0, 1].
        """
        self._validate_qubit(qubit1)
        self._validate_qubit(qubit2)
        self._validate_probability(p)
        self.circuit.append("DEPOLARIZE2", [qubit1, qubit2], p)
        log.logger.debug(f"Applied DEPOLARIZE2 on qubits {qubit1},{qubit2} with p={p}")


    # ────────────────────────────────────────────────────────────────────────────
    # Reset (to |0> state)
    def reset(self, qubit: int) -> None:
        """Reset the specified qubit to |0> by measurement and conditional X."""
        self.measure(qubit)
        self.x_if(-1, qubit)

    # ────────────────────────────────────────────────────────────────────────────
    # Sampling

    def sample(self, shots: int = 1) -> np.ndarray:
        """
        Draw `shots` samples of all measured qubits (0/1) via Stim 1.15.0.
        """
        sampler = self.circuit.compile_sampler()
        return sampler.sample(shots=shots).astype(int)

    def tomography_density_matrix(self, qubit: int, shots: int = 1000) -> np.ndarray:
        """
        Estimate the 1-qubit density matrix on `qubit` by sampling in Z, X, and Y:
         - clone the circuit
         - append basis-change gates + measure
         - compile & sample via Stim

        Returns a 2x2 numpy array rho.
        """
        # Pauli matrices
        I = np.array([[1, 0], [0, 1]], dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)

        # map 0/1 bits → expectation value (+1/−1)
        def exp_val(bits: np.ndarray) -> float:
            return 1 - 2 * bits.mean()

        # helper: clone + prep_ops + measure + sample
        def sample_basis(prep_ops: list[str]) -> np.ndarray:
            sc = StabilizerCircuit(self.num_qubits)
            sc.circuit = self.circuit.copy()
            for op in prep_ops:
                getattr(sc, op)(qubit)
            sc.measure(qubit)
            sampler = sc.circuit.compile_sampler()
            return sampler.sample(shots=shots)[:, 0]

        # collect expectation values
        ez = exp_val(sample_basis([]))            # Z basis
        ex = exp_val(sample_basis(['h']))         # X = H then M
        ey = exp_val(sample_basis(['sdg', 'h']))  # Y = S†, H then M

        # reconstruct ρ = ½ (I + xX + yY + zZ)
        rho = 0.5 * (I + ex * X + ey * Y + ez * Z)
        return rho
    


    def tomography_density_matrix_1(
        self,
        qubit: int,
        shots: int = 1000,
    ) -> tuple[np.ndarray, dict[str, int]]:
        """
        Estimate the 1-qubit density matrix on `qubit` by sampling in Z, X, Y.
        Also return the measurement result of a *single* shot in each basis.

        Returns
        -------
        rho : np.ndarray
            2x2 density matrix.
        single_shot : dict[str, int]
            Raw outcomes { "Z": 0/1, "X": 0/1, "Y": 0/1 } taken from one shot.
        """
        # Pauli matrices
        I = np.array([[1, 0], [0, 1]], dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)

        # helper to compute ⟨σ⟩ and also grab one raw bit
        def sample_basis(prep_ops: list[str]):
            sc = StabilizerCircuit(self.num_qubits)
            sc.circuit = self.circuit.copy()

            for op in prep_ops:
                getattr(sc, op)(qubit)           # apply H / Sdg etc.

            sc.measure(qubit)
            sampler = sc.circuit.compile_sampler()

            # sample many shots for expectation value …
            all_bits = sampler.sample(shots=shots)[:, 0]
            exp_val = 1 - 2 * all_bits.mean()

            # … and one extra shot for the raw outcome
            single_bit = int(sampler.sample(shots=1)[0, 0])

            return exp_val, single_bit

        # Z basis
        ez, bit_z = sample_basis([])

        # X basis:    |±>  = H|0/1>
        ex, bit_x = sample_basis(['h'])

        # Y basis:    |±_y> = HS†|0/1>   (Sdg then H)
        ey, bit_y = sample_basis(['sdg', 'h'])

        # reconstruct ρ = ½ ( I + x X + y Y + z Z )
        rho = 0.5 * (I + ex * X + ey * Y + ez * Z)

        single_shot = {"Z": bit_z, "X": bit_x, "Y": bit_y}
        return rho, single_shot


    # ────────────────────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        """Return the number of operations (gates + measurements + noise)."""
        return len(self.circuit)

    def __repr__(self) -> str:
        return (
            f"<StabilizerCircuit qubits={self.num_qubits} "
            f"ops={len(self)} measures={len(self._measured_qubits)}>"
        )

    def __str__(self) -> str:
        """Return the textual .stim file format of the circuit."""
        return str(self.circuit)

    @classmethod
    def from_stim_circuit(
        cls, stim_circuit: stim.Circuit
    ) -> "StabilizerCircuit":
        """
        Construct a StabilizerCircuit from an existing stim.Circuit.

        Args:
            stim_circuit: An existing Stim Circuit instance.

        Returns:
            A new StabilizerCircuit instance with the same content.
        """
        instance = cls(stim_circuit.num_qubits)
        instance.circuit = stim_circuit.copy()
        return instance
