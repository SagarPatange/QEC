"""ABCD system setup using stim to demonstrate Steane-code-based teleportation.

This module implements a quantum error correction protocol using the Steane [[7,1,3]]
code to distribute entanglement across four nodes (A, B, C, D) via teleported CNOTs.
The system uses the stim library's tableau simulator for efficient stabilizer circuit
simulation.
"""

from typing import Iterable, Sequence

import numpy as np
import stim

# Steane code parity-check matrix (H_74)
# Columns correspond to physical qubits, rows are the 3 independent checks of
# the [[7, 1, 3]] Steane code. The same matrix captures X- and Z-type checks.
# This 3×7 matrix defines the stabilizer generators for both X-type and Z-type
# stabilizers, enabling syndrome extraction and error correction.
H_74 = np.array(
    [
        [0, 0, 0, 1, 1, 1, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [1, 0, 1, 0, 1, 0, 1],
    ],
    dtype=np.uint8,
)


def _segment(start: int, length: int) -> np.ndarray:
    """Helper function to create a contiguous qubit index array."""
    return np.arange(start, start + length, dtype=np.int64)


# Data registers (Steane blocks at nodes A, B_left/right, C_left/right, D)
# Each register contains 7 qubits representing a logical qubit in the Steane code.
# Nodes B and C have two registers each (left and right) to facilitate teleportation.
Q_DATA_A = _segment(0, 7)
Q_DATA_B_LEFT = _segment(7, 7)
Q_DATA_B_RIGHT = _segment(14, 7)
Q_DATA_C_LEFT = _segment(21, 7)
Q_DATA_C_RIGHT = _segment(28, 7)
Q_DATA_D = _segment(35, 7)

# Ancilla registers used for Steane syndrome extraction at the ends
# Each endpoint (A and D) requires 6 ancilla qubits: 3 for X-stabilizers and 3 for Z-stabilizers.
Q_ANC_A = _segment(42, 6)
Q_ANC_D = _segment(48, 6)

# Communication registers mediating entanglement across the chain
# These 7-qubit Bell pairs serve as the quantum communication resource for teleportation.
# Pairs (q_comm_A ↔ q_comm_B_left), (q_comm_B_right ↔ q_comm_C_left), (q_comm_C_right ↔ q_comm_D)
# connect adjacent nodes in the ABCD chain.
Q_COMM_A = _segment(54, 7)
Q_COMM_B_LEFT = _segment(61, 7)
Q_COMM_B_RIGHT = _segment(68, 7)
Q_COMM_C_LEFT = _segment(75, 7)
Q_COMM_C_RIGHT = _segment(82, 7)
Q_COMM_D = _segment(89, 7)


DATA_ATTRS = (
    "q_data_A",
    "q_data_B_left",
    "q_data_B_right",
    "q_data_C_left",
    "q_data_C_right",
    "q_data_D",
)

ANC_ATTRS = (
    "q_anc_A",
    "q_anc_D",
)

COMM_ATTRS = (
    "q_comm_A",
    "q_comm_B_left",
    "q_comm_B_right",
    "q_comm_C_left",
    "q_comm_C_right",
    "q_comm_D",
)

# Subset of communication qubits where noise is applied in noise models
COMM_NOISE_ATTRS = ("q_comm_A", "q_comm_B_right", "q_comm_C_right")

# All communication qubits measured during the teleportation protocol
COMM_MEAS_ATTRS = (
    "q_comm_A",
    "q_comm_B_left",
    "q_comm_B_right",
    "q_comm_C_left",
    "q_comm_C_right",
    "q_comm_D",
)

# Dictionary mapping attribute names to default qubit arrays for easy initialization

REGISTER_DEFAULTS = {
    "q_data_A": Q_DATA_A,
    "q_data_B_left": Q_DATA_B_LEFT,
    "q_data_B_right": Q_DATA_B_RIGHT,
    "q_data_C_left": Q_DATA_C_LEFT,
    "q_data_C_right": Q_DATA_C_RIGHT,
    "q_data_D": Q_DATA_D,
    "q_anc_A": Q_ANC_A,
    "q_anc_D": Q_ANC_D,
    "q_comm_A": Q_COMM_A,
    "q_comm_B_left": Q_COMM_B_LEFT,
    "q_comm_B_right": Q_COMM_B_RIGHT,
    "q_comm_C_left": Q_COMM_C_LEFT,
    "q_comm_C_right": Q_COMM_C_RIGHT,
    "q_comm_D": Q_COMM_D,
}

# Pauli operator labels and mappings
PAULI_LABELS = ("I", "X", "Y", "Z")
PAULI_GATE_MAP = {"X": "x", "Y": "y", "Z": "z"}  # Map from uppercase labels to stim gate names

# All 16 two-qubit Pauli products (I⊗I, I⊗X, ..., Z⊗Z) for two-qubit noise models
TWO_Q_PAULI_PRODUCTS = tuple(
    (PAULI_LABELS[i], PAULI_LABELS[j]) for i in range(4) for j in range(4)
)


class ABCD_System:
    """Encapsulates the ABCD layout and tableau simulator for the Steane code.
    
    This class manages the full quantum circuit for distributing entanglement across
    four nodes using teleported CNOTs with Steane [[7,1,3]] error correction. It maintains:
    - All qubit registers (data, ancilla, communication)
    - A stim TableauSimulator for efficient stabilizer simulation
    - Operation history for debugging
    - BSM (Bell-State Measurement) results and error-corrected bits
    """

    def __init__(self) -> None:
        # Store references to register attribute names for iteration
        self._data_attrs = DATA_ATTRS
        self._anc_attrs = ANC_ATTRS
        self._comm_attrs = COMM_ATTRS
        self._register_attrs = self._data_attrs + self._anc_attrs + self._comm_attrs

        # Initialize all qubit registers as instance attributes
        for attr in self._register_attrs:
            setattr(self, attr, REGISTER_DEFAULTS[attr].copy())

        # Store a copy of the Steane parity-check matrix
        self.H = H_74.copy()
        
        # Random number generator for noise injection
        self._rng = np.random.default_rng()

        # Collect all unique qubit IDs and determine total register size
        self._all_qubits = self._collect_all_qubits()
        self.qubits_number = len(self._all_qubits)
        self._register_size = max(self._all_qubits) + 1

        # Initialize the tableau simulator and operation log
        self.TableauSimulator = stim.TableauSimulator()
        self.operations_applied = []

        # Storage for Bell-state measurement results
        # These will be populated during BSM operations at nodes B and C
        self.z_bits_BSM_B = np.array([], dtype=np.uint8)
        self.x_bits_BSM_B = np.array([], dtype=np.uint8)
        self.z_bits_BSM_C = np.array([], dtype=np.uint8)
        self.x_bits_BSM_C = np.array([], dtype=np.uint8)
        
        # Error-corrected BSM bits (aggregated from B and C)
        self.x_bit_BSM_corrected = np.array([], dtype=np.uint8)
        self.z_bit_BSM_corrected = np.array([], dtype=np.uint8)

        # Prepare the initial quantum state: all qubits in |0⟩ and create entanglement resources
        self._initialize_all_qubits()
        self._prepare_maximally_entangled_resource()

    # ------------------------------------------------------------------
    # Public summary helpers
    # ------------------------------------------------------------------
    def summary_qubit_layout(self) -> None:
        """Prints a summary of the qubit layout across all registers."""

        self._emit_register_group("Qubit layout (data):", self._data_attrs)
        self._emit_register_group("Qubit layout (ancilla):", self._anc_attrs)
        self._emit_register_group("Qubit layout (communication):", self._comm_attrs)
        print(f"Total unique qubits tracked: {self.qubits_number}")

    def summary_stabilizers(self) -> None:
        """Prints canonical stabilizers from the tableau simulator."""

        tableau = self._get_current_tableau()
        num_qubits = self._tableau_size(tableau)
        label_width = len(f"S{max(num_qubits - 1, 0)}")

        for stabilizer_index in range(num_qubits):
            pauli_string = tableau.z_output(stabilizer_index)
            sign = '+' if getattr(pauli_string, 'sign', 1) == 1 else '-'
            body = self._pauli_string_to_text(pauli_string)
            label = f"S{stabilizer_index}".ljust(label_width)
            print(f"{label}: {sign}{body}")

    def summary_BSM(self) -> None:
        """Prints stored Bell-state measurement (BSM) results."""

        print("BSM results at node B:")
        print(f"  z_bits_BSM_B        : {self.z_bits_BSM_B}")
        print(f"  x_bits_BSM_B        : {self.x_bits_BSM_B}")

        print("BSM results at node C:")
        print(f"  z_bits_BSM_C        : {self.z_bits_BSM_C}")
        print(f"  x_bits_BSM_C        : {self.x_bits_BSM_C}")

        print("Corrected BSM aggregates (B + C):")
        print(f"  z_bit_BSM_corrected : {self.z_bit_BSM_corrected}")
        print(f"  x_bit_BSM_corrected : {self.x_bit_BSM_corrected}")

    def print_operations_history(self) -> None:
        """Prints the recorded history of operations applied to the system."""

        if not self.operations_applied:
            print("No operations have been applied yet.")
            return

        for index, entry in enumerate(self.operations_applied, start=1):
            op_name = entry.get('op', 'UNKNOWN')
            targets = entry.get('targets', [])
            metadata = [
                f"{key}={value}"
                for key, value in entry.items()
                if key not in {'op', 'targets'}
            ]
            metadata_str = f" ({', '.join(metadata)})" if metadata else ""
            print(f"{index:04d}: {op_name} targets={targets}{metadata_str}")

    # ------------------------------------------------------------------
    # Noise initialization & teleportation routines
    # ------------------------------------------------------------------
    def initial_q_comm_noise_v1(self, f: float) -> None:
        """Applies symmetric Pauli noise to communication qubits.
        
        Args:
            f: Fidelity parameter in [0, 1]. The noise model uses probability
               p_I=f, p_X=p_Y=p_Z=(1-f)/3 for each communication qubit.
        """

        f = float(f)
        if not 0.0 <= f <= 1.0:
            raise ValueError("f must be in [0, 1].")

        p_v = self._prepare_prob_vector([f, (1.0 - f) / 3.0, (1.0 - f) / 3.0, (1.0 - f) / 3.0], "p_v")
        for attr in COMM_NOISE_ATTRS:
            for site in getattr(self, attr):
                self._pauli_noise_1qubit(int(site), p_v, label=f"{attr}_noise_v1")

    def initial_q_comm_noise_v2(self, f: float) -> None:
        """Applies asymmetric (3:1:2) Pauli noise to communication qubits.
        
        Args:
            f: Fidelity parameter in [0, 1]. The noise model uses probability
               p_I=f, with the remaining error budget (1-f) split in a 3:1:2
               ratio among X, Y, Z errors respectively.
        """

        f = float(f)
        if not 0.0 <= f <= 1.0:
            raise ValueError("f must be in [0, 1].")

        remainder = 1.0 - f
        weights = np.array([3.0, 1.0, 2.0], dtype=float)
        error_probs = remainder * (weights / weights.sum())
        p_v = self._prepare_prob_vector([f, *error_probs], "p_v")
        for attr in COMM_NOISE_ATTRS:
            for site in getattr(self, attr):
                self._pauli_noise_1qubit(int(site), p_v, label=f"{attr}_noise_v2")

    def q_data_comm_tele_cnot(
        self,
        gate_missing_rate: float = 0.0,
        gate_fidelity: float = 1.0,
    ) -> None:
        """Implements teleportation-based CNOT distribution as specified in the prompt.
        
        This method executes the full teleportation sequence to distribute CNOT gates
        across adjacent data blocks (A↔B, B↔C, C↔D) using the prepared communication
        entanglement resources. The protocol includes:
        1. Entangling data qubits with communication qubits via CNOTs
        2. Basis changes (Hadamards) on selected communication qubits
        3. Pre-measurement Pauli noise
        4. Bell-basis measurements of communication qubits
        5. Feed-forward Pauli corrections to data qubits based on measurement outcomes
        
        Args:
            gate_missing_rate: Probability in [0, 1] that a teleportation round is skipped
            gate_fidelity: Fidelity parameter for gate noise; 1.0 means perfect gates
        """

        gate_missing_rate = float(gate_missing_rate)
        gate_fidelity = float(gate_fidelity)
        if not 0.0 <= gate_missing_rate <= 1.0:
            raise ValueError("gate_missing_rate must be in [0, 1].")
        if not 0.0 <= gate_fidelity <= 1.0:
            raise ValueError("gate_fidelity must be in [0, 1].")

        p_v_1 = self._prepare_prob_vector(
            [gate_fidelity, (1.0 - gate_fidelity) / 3.0, (1.0 - gate_fidelity) / 3.0, (1.0 - gate_fidelity) / 3.0],
            "p_v_1",
        )
        p_v_2 = self._prepare_prob_vector(
            [gate_fidelity] + [(1.0 - gate_fidelity) / 15.0] * 15,
            "p_v_2",
        )

        for idx in range(len(self.q_data_A)):
            if self._rng.random() < gate_missing_rate:
                self._log_operation(
                    'TELE_CNOT_SKIP',
                    [int(self.q_data_A[idx])],
                    label=f'iteration_{idx}',
                    note='gate_missing',
                )
                continue

            # (1) Entangle A↔B_left chain.
            self._apply_cnot_with_noise(
                self.q_data_A[idx],
                self.q_comm_A[idx],
                p_v_2,
                label=f'q_data_A_to_comm_A[{idx}]',
            )
            self._apply_cnot_with_noise(
                self.q_comm_B_left[idx],
                self.q_data_B_left[idx],
                p_v_2,
                label=f'comm_B_left_to_data_B_left[{idx}]',
            )

            # (2) Entangle B_right↔C_left chain.
            self._apply_cnot_with_noise(
                self.q_data_B_right[idx],
                self.q_comm_B_right[idx],
                p_v_2,
                label=f'data_B_right_to_comm_B_right[{idx}]',
            )
            self._apply_cnot_with_noise(
                self.q_comm_C_left[idx],
                self.q_data_C_left[idx],
                p_v_2,
                label=f'comm_C_left_to_data_C_left[{idx}]',
            )

            # (3) Entangle C_right↔D chain.
            self._apply_cnot_with_noise(
                self.q_data_C_right[idx],
                self.q_comm_C_right[idx],
                p_v_2,
                label=f'data_C_right_to_comm_C_right[{idx}]',
            )
            self._apply_cnot_with_noise(
                self.q_comm_D[idx],
                self.q_data_D[idx],
                p_v_2,
                label=f'comm_D_to_data_D[{idx}]',
            )

            # (4) Hadamards and pre-measurement noise on specified comm qubits.
            for attr in ("q_comm_B_left", "q_comm_C_left", "q_comm_D"):
                qubit = int(getattr(self, attr)[idx])
                self._apply_single_qubit_gate('h', qubit, label=f'{attr}[{idx}]_basis_change')
                self._pauli_noise_1qubit(qubit, p_v_1, label=f'{attr}[{idx}]_basis_noise')

            # (5) Pre-measurement noise on all communication qubits.
            for attr in COMM_MEAS_ATTRS:
                qubit = int(getattr(self, attr)[idx])
                self._pauli_noise_1qubit(qubit, p_v_1, label=f'{attr}[{idx}]_meas_noise')

            # (6)-(8) Bell-basis measurements on communication qubits.
            x_bit_B_left = self._measure_qubit(int(self.q_comm_A[idx]), label=f'q_comm_A[{idx}]_meas')
            z_bit_A = self._measure_qubit(int(self.q_comm_B_left[idx]), label=f'q_comm_B_left[{idx}]_meas')

            x_bit_C_left = self._measure_qubit(int(self.q_comm_B_right[idx]), label=f'q_comm_B_right[{idx}]_meas')
            z_bit_B_right = self._measure_qubit(int(self.q_comm_C_left[idx]), label=f'q_comm_C_left[{idx}]_meas')

            x_bit_D = self._measure_qubit(int(self.q_comm_C_right[idx]), label=f'q_comm_C_right[{idx}]_meas')
            z_bit_C_right = self._measure_qubit(int(self.q_comm_D[idx]), label=f'q_comm_D[{idx}]_meas')

            # (9) Feed-forward Pauli corrections with accompanying single-qubit noise.
            if z_bit_A:
                target = int(self.q_data_A[idx])
                self._apply_single_qubit_gate('z', target, label=f'q_data_A[{idx}]_feedforward_Z')
                self._pauli_noise_1qubit(target, p_v_1, label=f'q_data_A[{idx}]_feedforward_Z_noise')

            if x_bit_B_left:
                target = int(self.q_data_B_left[idx])
                self._apply_single_qubit_gate('x', target, label=f'q_data_B_left[{idx}]_feedforward_X')
                self._pauli_noise_1qubit(target, p_v_1, label=f'q_data_B_left[{idx}]_feedforward_X_noise')

            if z_bit_B_right:
                target = int(self.q_data_B_right[idx])
                self._apply_single_qubit_gate('z', target, label=f'q_data_B_right[{idx}]_feedforward_Z')
                self._pauli_noise_1qubit(target, p_v_1, label=f'q_data_B_right[{idx}]_feedforward_Z_noise')

            if x_bit_C_left:
                target = int(self.q_data_C_left[idx])
                self._apply_single_qubit_gate('x', target, label=f'q_data_C_left[{idx}]_feedforward_X')
                self._pauli_noise_1qubit(target, p_v_1, label=f'q_data_C_left[{idx}]_feedforward_X_noise')

            if z_bit_C_right:
                target = int(self.q_data_C_right[idx])
                self._apply_single_qubit_gate('z', target, label=f'q_data_C_right[{idx}]_feedforward_Z')
                self._pauli_noise_1qubit(target, p_v_1, label=f'q_data_C_right[{idx}]_feedforward_Z_noise')

            if x_bit_D:
                target = int(self.q_data_D[idx])
                self._apply_single_qubit_gate('x', target, label=f'q_data_D[{idx}]_feedforward_X')
                self._pauli_noise_1qubit(target, p_v_1, label=f'q_data_D[{idx}]_feedforward_X_noise')

    def q_data_B_local_BSM(self, BSM_noise: float = 0.0) -> None:
        """Executes Bell-state measurements on node B with noise."""

        self.z_bits_BSM_B, self.x_bits_BSM_B = self._perform_local_BSM(
            left_attr="q_data_B_left",
            right_attr="q_data_B_right",
            base_label="q_data_B",
            noise=BSM_noise,
        )

    def q_data_C_local_BSM(self, BSM_noise: float = 0.0) -> None:
        """Executes Bell-state measurements on node C with noise."""

        self.z_bits_BSM_C, self.x_bits_BSM_C = self._perform_local_BSM(
            left_attr="q_data_C_left",
            right_attr="q_data_C_right",
            base_label="q_data_C",
            noise=BSM_noise,
        )

    def error_correction_on_BSM_bits(self) -> tuple[np.ndarray, np.ndarray]:
        """Performs Hamming code (7, 4) decoding on the collected BSM bits.
        
        This method takes the raw BSM bits from nodes B and C, computes their syndromes
        using the parity-check matrix, decodes to find errors, and aggregates
        the error-corrected bits. The corrected bits represent the logical Bell state
        information needed for Pauli-frame correction at node A.
        
        Returns:
            Tuple of (z_error_BSM, x_error_BSM): The decoded error patterns
        """

        x_bit_B = self._ensure_binary_array(self.x_bits_BSM_B, 'x_bits_BSM_B')
        z_bit_B = self._ensure_binary_array(self.z_bits_BSM_B, 'z_bits_BSM_B')
        x_bit_C = self._ensure_binary_array(self.x_bits_BSM_C, 'x_bits_BSM_C')
        z_bit_C = self._ensure_binary_array(self.z_bits_BSM_C, 'z_bits_BSM_C')

        z_error_B = self._syndrome_to_error_H74(self._message_to_syndrome_H74(x_bit_B))
        z_error_C = self._syndrome_to_error_H74(self._message_to_syndrome_H74(x_bit_C))
        z_error_BSM = (z_error_B + z_error_C) % 2

        x_error_B = self._syndrome_to_error_H74(self._message_to_syndrome_H74(z_bit_B))
        x_error_C = self._syndrome_to_error_H74(self._message_to_syndrome_H74(z_bit_C))
        x_error_BSM = (x_error_B + x_error_C) % 2

        x_bit_total = (x_bit_B + x_bit_C) % 2
        z_bit_total = (z_bit_B + z_bit_C) % 2

        self.x_bit_BSM_corrected = (x_bit_total + z_error_BSM) % 2
        self.z_bit_BSM_corrected = (z_bit_total + x_error_BSM) % 2

        return z_error_BSM, x_error_BSM

    def correct_pauli_frame(self) -> None:
        """Applies Pauli-frame corrections to node A using decoded BSM bits.
        
        After error correction on the BSM bits, this method applies Z gates (for each '1'
        in z_bit_BSM_corrected) and X gates (for each '1' in x_bit_BSM_corrected) to the
        corresponding data qubits at node A. This restores the target Bell state |Φ⁺⟩
        between nodes A and D.
        """

        z_corrections = self._ensure_binary_array(self.z_bit_BSM_corrected, 'z_bit_BSM_corrected')
        x_corrections = self._ensure_binary_array(self.x_bit_BSM_corrected, 'x_bit_BSM_corrected')

        for idx, needs_z in enumerate(z_corrections):
            if needs_z:
                target = int(self.q_data_A[idx])
                self._apply_single_qubit_gate('z', target, label=f'q_data_A[{idx}]_pauli_frame_Z')

        for idx, needs_x in enumerate(x_corrections):
            if needs_x:
                target = int(self.q_data_A[idx])
                self._apply_single_qubit_gate('x', target, label=f'q_data_A[{idx}]_pauli_frame_X')

    def syndrome_measurement_two_ends(self, gate_fidelity: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Runs noisy Steane stabilizer measurements on end nodes.
        
        Performs syndrome extraction at nodes A and D using 6 ancilla qubits per node:
        - 3 ancillas for X-stabilizers (via CNOT gates)
        - 3 ancillas for Z-stabilizers (via CZ gates)
        Each stabilizer measurement involves Hadamards, controlled gates, and measurements
        with optional noise. The syndromes are then decoded to infer physical error patterns.
        
        Args:
            gate_fidelity: Fidelity parameter for gates and measurements; 1.0 means noiseless
            
        Returns:
            Tuple of (synd_X_two_ends, synd_Z_two_ends, z_error_two_ends, x_error_two_ends):
            - synd_X_two_ends: Combined X-type syndrome (3 bits)
            - synd_Z_two_ends: Combined Z-type syndrome (3 bits)  
            - z_error_two_ends: Decoded Z-error pattern (7 bits)
            - x_error_two_ends: Decoded X-error pattern (7 bits)
        """

        gate_fidelity = float(gate_fidelity)
        if not 0.0 <= gate_fidelity <= 1.0:
            raise ValueError("gate_fidelity must be in [0, 1].")


        p_v_1 = self._prepare_prob_vector(
            [gate_fidelity, (1.0 - gate_fidelity) / 3.0, (1.0 - gate_fidelity) / 3.0, (1.0 - gate_fidelity) / 3.0],
            "p_v_1_two_ends",
        )
        p_v_2 = self._prepare_prob_vector(
            [gate_fidelity] + [(1.0 - gate_fidelity) / 15.0] * 15,
            "p_v_2_two_ends",
        )


        for attr in (self.q_anc_A, self.q_anc_D):
            for qubit in attr:
                self._apply_single_qubit_gate('h', int(qubit), label='anc_pre_H')
                if gate_fidelity < 1.0:
                    self._pauli_noise_1qubit(int(qubit), p_v_1, label='anc_pre_H_noise')

        for row_idx, pattern in enumerate(self.H):
            control_A = int(self.q_anc_A[row_idx])
            control_D = int(self.q_anc_D[row_idx])
            for col_idx, value in enumerate(pattern):
                if not value:
                    continue
                data_A = int(self.q_data_A[col_idx])
                data_D = int(self.q_data_D[col_idx])
                self._apply_cnot_with_noise(control_A, data_A, p_v_2, label=f'anc_A_CNOT_row{row_idx}_col{col_idx}')
                self._apply_cnot_with_noise(control_D, data_D, p_v_2, label=f'anc_D_CNOT_row{row_idx}_col{col_idx}')

        for row_offset, pattern in enumerate(self.H, start=3):
            control_A = int(self.q_anc_A[row_offset])
            control_D = int(self.q_anc_D[row_offset])
            for col_idx, value in enumerate(pattern):
                if not value:
                    continue
                data_A = int(self.q_data_A[col_idx])
                data_D = int(self.q_data_D[col_idx])
                self._apply_two_qubit_gate('cz', control_A, data_A, label=f'anc_A_CZ_row{row_offset}_col{col_idx}')
                if gate_fidelity < 1.0:
                    self._pauli_noise_2qubit([control_A, data_A], p_v_2, label=f'anc_A_CZ_row{row_offset}_col{col_idx}_noise')
                self._apply_two_qubit_gate('cz', control_D, data_D, label=f'anc_D_CZ_row{row_offset}_col{col_idx}')
                if gate_fidelity < 1.0:
                    self._pauli_noise_2qubit([control_D, data_D], p_v_2, label=f'anc_D_CZ_row{row_offset}_col{col_idx}_noise')

        for attr in (self.q_anc_A, self.q_anc_D):
            for qubit in attr:
                self._apply_single_qubit_gate('h', int(qubit), label='anc_post_H')
                if gate_fidelity < 1.0:
                    self._pauli_noise_1qubit(int(qubit), p_v_1, label='anc_post_H_noise')

        m_A_X = np.zeros(3, dtype=np.uint8)
        m_A_Z = np.zeros(3, dtype=np.uint8)
        m_D_X = np.zeros(3, dtype=np.uint8)
        m_D_Z = np.zeros(3, dtype=np.uint8)

        for idx in range(3):
            qubit_A = int(self.q_anc_A[idx])
            qubit_D = int(self.q_anc_D[idx])
            if gate_fidelity < 1.0:
                self._pauli_noise_1qubit(qubit_A, p_v_1, label=f'anc_A_X[{idx}]_meas_noise')
                self._pauli_noise_1qubit(qubit_D, p_v_1, label=f'anc_D_X[{idx}]_meas_noise')
            m_A_X[idx] = self._measure_qubit(qubit_A, label=f'anc_A_X[{idx}]_meas')
            m_D_X[idx] = self._measure_qubit(qubit_D, label=f'anc_D_X[{idx}]_meas')

        for idx in range(3, 6):
            qubit_A = int(self.q_anc_A[idx])
            qubit_D = int(self.q_anc_D[idx])
            if gate_fidelity < 1.0:
                self._pauli_noise_1qubit(qubit_A, p_v_1, label=f'anc_A_Z[{idx - 3}]_meas_noise')
                self._pauli_noise_1qubit(qubit_D, p_v_1, label=f'anc_D_Z[{idx - 3}]_meas_noise')
            m_A_Z[idx - 3] = self._measure_qubit(qubit_A, label=f'anc_A_Z[{idx - 3}]_meas')
            m_D_Z[idx - 3] = self._measure_qubit(qubit_D, label=f'anc_D_Z[{idx - 3}]_meas')

        synd_X_two_ends = (m_A_X + m_D_X) % 2
        synd_Z_two_ends = (m_A_Z + m_D_Z) % 2

        z_error_two_ends = self._syndrome_to_error_H74(synd_X_two_ends)
        x_error_two_ends = self._syndrome_to_error_H74(synd_Z_two_ends)

        return synd_X_two_ends, synd_Z_two_ends, z_error_two_ends, x_error_two_ends

    def recovery_at_two_ends(
        self,
        gate_fidelity: float = 1.0,
        x_error_two_ends: Sequence[int] | np.ndarray | None = None,
        z_error_two_ends: Sequence[int] | np.ndarray | None = None,
    ) -> None:
        """Applies recovery operations on node A guided by decoded errors.
        
        Based on the decoded error patterns from syndrome_measurement_two_ends,
        this method applies corrective Pauli gates to the data qubits at node A.
        For each '1' in z_error_two_ends, a Z gate is applied; for each '1' in
        x_error_two_ends, an X gate is applied. Optional gate noise can be included.
        
        Args:
            gate_fidelity: Fidelity parameter for recovery gates; 1.0 means noiseless
            x_error_two_ends: Decoded X-error pattern (7 bits)
            z_error_two_ends: Decoded Z-error pattern (7 bits)
        """

        gate_fidelity = float(gate_fidelity)
        if not 0.0 <= gate_fidelity <= 1.0:
            raise ValueError("gate_fidelity must be in [0, 1].")

        if x_error_two_ends is None or z_error_two_ends is None:
            raise ValueError("Both x_error_two_ends and z_error_two_ends must be provided.")

        x_error = self._ensure_binary_array(x_error_two_ends, 'x_error_two_ends')
        z_error = self._ensure_binary_array(z_error_two_ends, 'z_error_two_ends')

        p_v_1 = self._prepare_prob_vector(
            [gate_fidelity, (1.0 - gate_fidelity) / 3.0, (1.0 - gate_fidelity) / 3.0, (1.0 - gate_fidelity) / 3.0],
            "p_v_1_recovery",
        )

        for idx, value in enumerate(z_error):
            if value:
                target = int(self.q_data_A[idx])
                self._apply_single_qubit_gate('z', target, label=f'q_data_A[{idx}]_recovery_Z')
                if gate_fidelity < 1.0:
                    self._pauli_noise_1qubit(target, p_v_1, label=f'q_data_A[{idx}]_recovery_Z_noise')

        for idx, value in enumerate(x_error):
            if value:
                target = int(self.q_data_A[idx])
                self._apply_single_qubit_gate('x', target, label=f'q_data_A[{idx}]_recovery_X')
                if gate_fidelity < 1.0:
                    self._pauli_noise_1qubit(target, p_v_1, label=f'q_data_A[{idx}]_recovery_X_noise')

    def check_logical_error(self) -> tuple[bool, bool, bool]:
        """Checks whether the logical Bell state between nodes A and D is in the target |Φ⁺⟩ state.
        
        This method evaluates whether the final logical Bell state between nodes A and D
        matches the target |Φ⁺⟩ state by measuring the logical XX and ZZ operators.
        A logical error is detected if either operator returns an unexpected value.
        
        Returns:
            Tuple of (logical_error, logical_X_error, logical_Z_error):
            - logical_error: True if any logical error occurred
            - logical_X_error: True if logical X-type error detected (ZZ eigenvalue wrong)
            - logical_Z_error: True if logical Z-type error detected (XX eigenvalue wrong)
        """

        tableau = self._get_current_tableau()
        support = np.concatenate((self.q_data_A, self.q_data_D)).astype(int)

        x_string = self._pauli_string_on_support(support, 'X')
        z_string = self._pauli_string_on_support(support, 'Z')

        logical_Z_error = bool(self._tableau_measure_observable(tableau, x_string))
        logical_X_error = bool(self._tableau_measure_observable(tableau, z_string))

        logical_error = logical_Z_error or logical_X_error
        return logical_error, logical_X_error, logical_Z_error

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _emit_register_group(self, heading: str, attr_names: tuple[str, ...]) -> None:
        print(heading)
        width = max(len(name) for name in attr_names)
        for name in attr_names:
            label = name.ljust(width)
            print(f"  {label} : {getattr(self, name)}")

    def _collect_all_qubits(self) -> list[int]:
        """Aggregate every register into a single deduplicated list of qubit IDs.
        
        Returns:
            Sorted list of unique qubit indices used across all registers
        """

        registers = [getattr(self, attr).astype(np.int64, copy=False) for attr in self._register_attrs]
        concatenated = np.concatenate(registers)
        return np.unique(concatenated).astype(int).tolist()

    def _perform_local_BSM(
        self,
        *,
        left_attr: str,
        right_attr: str,
        base_label: str,
        noise: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Shared routine for local Bell-state measurements on internal nodes.
        
        Executes a Bell-state measurement between two 7-qubit Steane blocks by:
        1. Applying a CNOT gate from left to right qubit (with noise)
        2. Applying Hadamard to the left qubit
        3. Measuring both qubits in the computational basis
        
        Args:
            left_attr: Name of the left register attribute (e.g., "q_data_B_left")
            right_attr: Name of the right register attribute (e.g., "q_data_B_right")
            base_label: Base label for operation logging (e.g., "q_data_B")
            noise: Noise parameter for BSM operations
            
        Returns:
            Tuple of (z_bits, x_bits): 7-element arrays of measurement outcomes
        """

        BSM_noise = float(noise)
        if not 0.0 <= BSM_noise <= 1.0:
            raise ValueError("BSM_noise must be in [0, 1].")

        p_v_1 = self._prepare_prob_vector(
            [1.0 - BSM_noise, BSM_noise / 3.0, BSM_noise / 3.0, BSM_noise / 3.0],
            f"p_v_1_{base_label}",
        )
        p_v_2 = self._prepare_prob_vector(
            [1.0 - BSM_noise] + [BSM_noise / 15.0] * 15,
            f"p_v_2_{base_label}",
        )

        z_bits: list[int] = []
        x_bits: list[int] = []
        left_register = getattr(self, left_attr)
        right_register = getattr(self, right_attr)
        left_label = f"{base_label}_left"
        right_label = f"{base_label}_right"
        cnot_label = f"{base_label}_left_to_right"

        for idx, (left, right) in enumerate(zip(left_register, right_register)):
            left_qubit = int(left)
            right_qubit = int(right)

            self._apply_cnot_with_noise(left_qubit, right_qubit, p_v_2, label=f"{cnot_label}[{idx}]")

            self._apply_single_qubit_gate('h', left_qubit, label=f"{left_label}[{idx}]_BSM_prep")
            self._pauli_noise_1qubit(left_qubit, p_v_1, label=f"{left_label}[{idx}]_BSM_noise")

            z_bits.append(self._measure_qubit(left_qubit, label=f"{left_label}[{idx}]_meas"))
            x_bits.append(self._measure_qubit(right_qubit, label=f"{right_label}[{idx}]_meas"))

        return np.array(z_bits, dtype=np.uint8), np.array(x_bits, dtype=np.uint8)

    def _initialize_all_qubits(self) -> None:
        """Ensure the simulator starts in |0...0⟩ and record that fact.
        
        Resets all tracked qubits to the |0⟩ state using stim's reset method
        if available, or logs that the default initial state is acknowledged.
        """

        reset_fn = getattr(self.TableauSimulator, 'reset', None)
        if callable(reset_fn):
            for qubit in self._all_qubits:
                reset_fn(int(qubit))
            self._log_operation('RESET', self._all_qubits, detail='initialize |0> across all registers')
        else:
            self._log_operation(
                'INIT_ZERO',
                self._all_qubits,
                detail='stim.TableauSimulator default |0> state acknowledged (no explicit reset available)',
            )

    def _prepare_maximally_entangled_resource(self) -> None:
        """Create the initial entangled links and logical code states.
        
        This initialization routine:
        1. Prepares communication Bell pairs (|Φ⁺⟩ states) between adjacent nodes
        2. Projects each data block into the logical code space by measuring stabilizers
        3. Prepares certain data blocks in the |+⟩ basis via Hadamards
        
        After this method, the system has valid Steane-encoded logical qubits and
        entanglement resources ready for teleportation.
        """

        for attr in ("q_comm_A", "q_comm_B_right", "q_comm_C_right"):
            for qubit in getattr(self, attr):
                self._apply_single_qubit_gate('h', qubit, label=attr)

        self._entangle_registers('q_comm_A', 'q_comm_B_left', 'comm_A_to_B_left')
        self._entangle_registers('q_comm_B_right', 'q_comm_C_left', 'comm_B_right_to_C_left')
        self._entangle_registers('q_comm_C_right', 'q_comm_D', 'comm_C_right_to_D')

        for axis in ('X', 'Z'):
            for attr in self._data_attrs:
                block = getattr(self, attr)
                self._measure_block_stabilizers(attr, block, axis=axis)

        for attr in ('q_data_A', 'q_data_B_right', 'q_data_C_right'):
            for qubit in getattr(self, attr):
                self._apply_single_qubit_gate('h', qubit, label=attr)

    def _entangle_registers(self, control_attr: str, target_attr: str, label_prefix: str) -> None:
        controls = getattr(self, control_attr)
        targets = getattr(self, target_attr)
        for idx, (control, target) in enumerate(zip(controls, targets)):
            self._apply_two_qubit_gate('cx', control, target, label=f"{label_prefix}[{idx}]")

    def _measure_block_stabilizers(self, block_label: str, block: np.ndarray, axis: str) -> None:
        """Measure and, if needed, fix a Steane stabilizer on a given block.
        
        For each row of the parity-check matrix H, this method measures the corresponding
        stabilizer generator (either X-type or Z-type). If the measurement outcome is -1
        (indicating the stabilizer is violated), it applies a corrective Pauli gate to
        flip the stabilizer sign back to +1, ensuring the block is in the code space.
        
        Args:
            block_label: String identifier for the block (for logging)
            block: Array of 7 qubit indices comprising the Steane block
            axis: 'X' or 'Z' indicating which type of stabilizer to measure
        """

        assert axis in {'X', 'Z'}, "axis must be 'X' or 'Z'"

        for row_index, pattern in enumerate(self.H):
            targets = [int(block[i]) for i, value in enumerate(pattern) if value]
            if not targets:
                continue

            measurement_label = f"{block_label}_{axis}_stab[{row_index}]"
            result = self._measure_pauli_product(targets, axis=axis, label=measurement_label)

            if result:
                correction_axis = 'z' if axis == 'X' else 'x'
                correction_op = 'Z' if axis == 'X' else 'X'
                correction_target = targets[0]
                self._apply_single_qubit_gate(
                    correction_axis,
                    correction_target,
                    label=f"{measurement_label}_correction",
                    note=f"flip stabilizer sign for {axis}-type"
                )
                # Update the last logged operation to flag it as a correction.
                self.operations_applied[-1]['correction_for'] = measurement_label
                self.operations_applied[-1]['op'] = correction_op

    def _measure_pauli_product(self, targets: list[int], axis: str, label: str) -> int:
        """Measure a multi-qubit Pauli product via stim's observable API.
        
        Constructs a Pauli string with the specified axis (X or Z) acting on the target
        qubits and identity elsewhere, then measures it using the tableau simulator.
        
        Args:
            targets: List of qubit indices on which the Pauli operator acts
            axis: 'X' or 'Z' specifying the Pauli type
            label: String label for logging
            
        Returns:
            Measurement outcome: 0 for +1 eigenvalue, 1 for -1 eigenvalue
        """

        pauli_chars = ['I'] * self._register_size
        axis_char = axis.upper()
        for qubit in targets:
            pauli_chars[int(qubit)] = axis_char

        pauli_string = stim.PauliString(''.join(pauli_chars))
        result = self._call_stim('measure_observable', pauli_string)
        self._log_operation(
            f"MEASURE_{axis_char}",
            targets,
            label=label,
            result=int(result),
        )
        return int(result)

    def apply_pauli(self, qubit: int, axis: str, *, label: str | None = None, note: str | None = None) -> None:
        """Apply a single-qubit Pauli (X, Y, or Z) to the specified qubit."""

        axis = axis.upper()
        if axis not in ('X', 'Y', 'Z'):
            raise ValueError("axis must be one of 'X', 'Y', or 'Z'.")

        gate_name = PAULI_GATE_MAP[axis]
        self._apply_single_qubit_gate(gate_name, int(qubit), label=label, note=note)

    def _apply_single_qubit_gate(self, method_name: str, qubit: int, *, label: str | None = None, note: str | None = None) -> None:
        """Apply a single-qubit Clifford gate and log it."""
        self._call_stim(method_name, int(qubit))
        op_name = method_name.upper()
        self._log_operation(op_name, [int(qubit)], label=label, note=note)

    def _apply_two_qubit_gate(self, method_name: str, control: int, target: int, *, label: str | None = None) -> None:
        """Apply a two-qubit Clifford gate (usually CNOT) and log it."""
        self._call_stim(method_name, int(control), int(target))
        op_name = 'CNOT' if method_name.lower() in {'cx', 'cnot'} else method_name.upper()
        self._log_operation(op_name, [int(control), int(target)], label=label)

    def _get_current_tableau(self) -> stim.Tableau:
        """Fetch the simulator's current tableau, handling API variations.
        
        Different versions of stim may provide 'current_tableau()' or 'current_inverse_tableau()'.
        This method attempts both and inverts the inverse tableau if necessary.
        
        Returns:
            The current tableau representing the stabilizer state
        """

        direct_fn = getattr(self.TableauSimulator, 'current_tableau', None)
        if callable(direct_fn):
            return direct_fn()

        inverse_fn = getattr(self.TableauSimulator, 'current_inverse_tableau', None)
        if callable(inverse_fn):
            inverse_tableau = inverse_fn()
            return inverse_tableau.inverse()

        raise AttributeError(
            "stim.TableauSimulator provides neither 'current_tableau' nor 'current_inverse_tableau'."
        )

    def _tableau_size(self, tableau: stim.Tableau) -> int:
        num_qubits_attr = getattr(tableau, 'num_qubits', None)
        if callable(num_qubits_attr):
            return int(num_qubits_attr())
        if isinstance(num_qubits_attr, int):
            return num_qubits_attr
        return self.qubits_number

    def _call_stim(self, method_name: str, *args, **kwargs):
        """Thin wrapper that validates stim method availability."""
        method = getattr(self.TableauSimulator, method_name, None)
        if method is None:
            raise AttributeError(f"stim.TableauSimulator has no method named '{method_name}'.")
        return method(*args, **kwargs)

    def _log_operation(self, op_name: str, targets, **metadata) -> None:
        """Append an entry to operations_applied with optional metadata."""
        entry = {'op': op_name, 'targets': [int(q) for q in targets]}
        entry.update({key: value for key, value in metadata.items() if value is not None})
        self.operations_applied.append(entry)

    @staticmethod
    def _pauli_string_to_text(pauli_string: stim.PauliString) -> str:
        """Convert a Pauli string into a human-readable set of characters."""

        try:
            x_bits, z_bits = pauli_string.to_numpy()
            x_bits = np.array(x_bits, dtype=np.uint8)
            z_bits = np.array(z_bits, dtype=np.uint8)
        except AttributeError:
            string_repr = str(pauli_string)
            offset = 1 if string_repr and string_repr[0] in '+-' else 0
            core = string_repr[offset:]
            return core.replace('I', '_')

        characters: list[str] = []
        for x_bit, z_bit in zip(x_bits, z_bits):
            if x_bit and z_bit:
                characters.append('Y')
            elif x_bit:
                characters.append('X')
            elif z_bit:
                characters.append('Z')
            else:
                characters.append('_')
        return ''.join(characters)

    def _prepare_prob_vector(self, values: Sequence[float], label: str) -> np.ndarray:
        """Validate and normalize a probability vector.
        
        Ensures the input is a valid probability distribution (non-negative, sums to ~1).
        Automatically normalizes if the sum is close to but not exactly 1.
        
        Args:
            values: Sequence of probabilities
            label: Identifier for error messages
            
        Returns:
            Normalized probability array
        """
        arr = np.asarray(values, dtype=float)
        if arr.ndim != 1:
            raise ValueError(f"{label} must be a flat probability vector.")
        if np.any(arr < -1e-12):
            raise ValueError(f"{label} contains negative probabilities: {arr}.")
        total = arr.sum()
        if total <= 0.0:
            raise ValueError(f"{label} must sum to a positive value.")
        if not np.isclose(total, 1.0):
            arr = arr / total
        return arr

    def _pauli_noise_1qubit(self, qubit: int, probabilities: Sequence[float], *, label: str | None = None) -> None:
        """Apply single-qubit Pauli noise by sampling from a probability distribution.
        
        Randomly selects a Pauli operator (I, X, Y, or Z) according to the provided
        probabilities and applies it to the specified qubit.
        
        Args:
            qubit: Target qubit index
            probabilities: 4-element array [p_I, p_X, p_Y, p_Z]
            label: Optional label for logging
        """
        probs = self._prepare_prob_vector(probabilities, 'pauli_noise_1q')
        choice = int(self._rng.choice(len(probs), p=probs))
        pauli = PAULI_LABELS[choice]

        if pauli != 'I':
            gate_name = PAULI_GATE_MAP[pauli]
            self._call_stim(gate_name, qubit)

        self._log_operation(
            'PAULI_NOISE_1Q',
            [qubit],
            label=label,
            pauli=pauli,
            probs=probs.tolist(),
        )

    def _pauli_noise_2qubit(self, qubits: Iterable[int], probabilities: Sequence[float], *, label: str | None = None) -> None:
        """Apply two-qubit Pauli noise by sampling from the 16 Pauli products.
        
        Randomly selects a two-qubit Pauli product (e.g., IX, XY, ZZ, etc.) according to
        the provided probabilities and applies it to the two target qubits.
        
        Args:
            qubits: Iterable of exactly 2 qubit indices
            probabilities: 16-element array for all two-qubit Pauli products
            label: Optional label for logging
        """
        probs = self._prepare_prob_vector(probabilities, 'pauli_noise_2q')
        qubit_list = [int(q) for q in qubits]
        if len(qubit_list) != 2:
            raise ValueError("Two-qubit noise requires exactly two target qubits.")

        choice = int(self._rng.choice(len(probs), p=probs))
        pauli_pair = TWO_Q_PAULI_PRODUCTS[choice]

        for pauli, qubit in zip(pauli_pair, qubit_list):
            if pauli == 'I':
                continue
            gate_name = PAULI_GATE_MAP[pauli]
            self._call_stim(gate_name, qubit)

        self._log_operation(
            'PAULI_NOISE_2Q',
            qubit_list,
            label=label,
            pauli=''.join(pauli_pair),
            probs=probs.tolist(),
        )

    def _apply_cnot_with_noise(self, control: int, target: int, probabilities: Sequence[float], *, label: str | None = None) -> None:
        control = int(control)
        target = int(target)
        self._apply_two_qubit_gate('cx', control, target, label=label)
        self._pauli_noise_2qubit([control, target], probabilities, label=f"{label}_noise" if label else None)

    def _measure_qubit(self, qubit: int, *, label: str | None = None) -> int:
        result = int(self._call_stim('measure', qubit))
        self._log_operation('MEASURE_Z', [qubit], label=label, result=result)
        return result

    def _ensure_binary_array(self, value: Sequence[int] | np.ndarray, name: str) -> np.ndarray:
        """Validate and convert a sequence to a binary (mod 2) array of length 7.
        
        Args:
            value: Input sequence (must have 7 elements)
            name: Variable name for error messages
            
        Returns:
            7-element uint8 array with values reduced mod 2
        """
        arr = np.array(value, dtype=np.uint8).flatten()
        if arr.size == 0:
            raise ValueError(f"{name} is empty; run the requisite measurement routine first.")
        if arr.size != 7:
            raise ValueError(f"{name} must have length 7; received length {arr.size}.")
        return arr % 2

    def _message_to_syndrome_H74(self, message: Sequence[int] | np.ndarray) -> np.ndarray:
        """Compute the Hamming code (7, 4) syndrome from a 7-bit message using H @ message mod 2.
        
        Args:
            message: 7-element binary array (the codeword or error pattern)
            
        Returns:
            3-element syndrome array
        """
        msg = self._ensure_binary_array(message, 'message')
        synd = (self.H @ msg) % 2
        return synd.astype(np.uint8)

    def _syndrome_to_error_H74(self, syndrome: Sequence[int] | np.ndarray) -> np.ndarray:
        """Decode a Hamming code (7, 4) syndrome to a 7-bit error pattern.
        
        Performs simple syndrome decoding: if the syndrome matches a column of H,
        returns a single-bit error at that position; otherwise returns all zeros.
        
        Args:
            syndrome: 3-element syndrome array
            
        Returns:
            7-element error pattern (one-hot or all-zero)
        """
        synd = np.array(syndrome, dtype=np.uint8).flatten() % 2
        if synd.size != 3:
            raise ValueError(f"syndrome must have length 3; received length {synd.size}.")
        if np.all(synd == 0):
            return np.zeros(7, dtype=np.uint8)
        for idx in range(self.H.shape[1]):
            if np.array_equal(self.H[:, idx], synd):
                out = np.zeros(7, dtype=np.uint8)
                out[idx] = 1
                return out
        return np.zeros(7, dtype=np.uint8)

    def _pauli_string_on_support(self, support: Iterable[int], axis: str) -> stim.PauliString:
        """Build a Pauli string that acts with the given axis on the support qubits.
        
        Creates a full-register Pauli string with identity everywhere except on the
        specified support qubits, where the given Pauli type (X or Z) is applied.
        
        Args:
            support: Iterable of qubit indices
            axis: 'X' or 'Z' specifying the Pauli type
            
        Returns:
            A stim.PauliString object
        """
        axis = axis.upper()
        if axis not in {'X', 'Z'}:
            raise ValueError("axis must be 'X' or 'Z'.")
        chars = ['I'] * self._register_size
        for qubit in support:
            chars[int(qubit)] = axis
        return stim.PauliString(''.join(chars))

    def _tableau_measure_observable(self, tableau: stim.Tableau, observable: stim.PauliString) -> int:
        measure_fn = getattr(tableau, 'measure_observable', None)
        if callable(measure_fn):
            return int(measure_fn(observable))
        clone = self._clone_simulator()
        return int(clone.measure_observable(observable))

    def _clone_simulator(self) -> stim.TableauSimulator:
        copy_fn = getattr(self.TableauSimulator, 'copy', None)
        if callable(copy_fn):
            return copy_fn()
        raise AttributeError("stim.TableauSimulator does not support copying; logical checks unavailable.")


