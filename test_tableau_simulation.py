"""5-node Steane tableau simulation with global noise parameters."""

from __future__ import annotations

import statistics
import stim


# Fixed physical qubit layout for the 5-node chain:
# Data blocks (56 qubits) + communication blocks for teleported CNOT links (56 qubits).
# Total = 112 qubits.
BLOCK_QUBITS = {
    # Data blocks.
    "A": list(range(0, 7)),
    "B_L": list(range(7, 14)),
    "B_R": list(range(14, 21)),
    "C_L": list(range(21, 28)),
    "C_R": list(range(28, 35)),
    "D_L": list(range(35, 42)),
    "D_R": list(range(42, 49)),
    "E": list(range(49, 56)),

    # Communication blocks for each inter-node link direction.
    "A_to_BL": list(range(56, 63)),
    "BL_to_A": list(range(63, 70)),
    "BR_to_CL": list(range(70, 77)),
    "CL_to_BR": list(range(77, 84)),
    "CR_to_DL": list(range(84, 91)),
    "DL_to_CR": list(range(91, 98)),
    "DR_to_E": list(range(98, 105)),
    "E_to_DR": list(range(105, 112)),
}

# Fixed block order for ancilla assignment.
LOGICAL_BLOCK_ORDER = [
    "A", "B_L", "B_R", "C_L", "C_R", "D_L", "D_R", "E",
    "A_to_BL", "BL_to_A", "BR_to_CL", "CL_to_BR",
    "CR_to_DL", "DL_to_CR", "DR_to_E", "E_to_DR",
]

# Four FT ancillas per logical block (enough for standard FT checks).
# Minimal FT uses just the first ancilla in each list.
FT_ANCILLA_QUBITS = {
    name: list(range(112 + 4 * i, 112 + 4 * i + 4))
    for i, name in enumerate(LOGICAL_BLOCK_ORDER)
}

TOTAL_QUBITS = 112 + 4 * len(LOGICAL_BLOCK_ORDER)  # 112 + 64 = 176

# Steane parity-check equations for bit order [q0..q6].
CHECK_ROWS = [
    [3, 4, 5, 6],
    [1, 2, 5, 6],
    [0, 2, 4, 6],
]

# Syndrome -> single-qubit correction index.
DECODE_TABLE = {
    (0, 0, 0): None,
    (0, 0, 1): 0,
    (0, 1, 0): 1,
    (0, 1, 1): 2,
    (1, 0, 0): 3,
    (1, 0, 1): 4,
    (1, 1, 0): 5,
    (1, 1, 1): 6,
}

# Z-type stabilizers used for FT |0_L> verification.
Z_STABILIZERS_STANDARD = [
    [3, 4, 5, 6],  # IIIZZZZ
    [0, 5, 6],     # ZIIIIZZ
    [1, 4, 6],     # IZIIZIZ
    [2, 4, 5],     # IIZIZZI
]
Z_STABILIZER_MINIMAL = [0, 5, 6]  # ZIIIIZZ

# Stim PAULI_CHANNEL_2 probability order.
PAULI2_ORDER = [
    "IX", "IY", "IZ",
    "XI", "XX", "XY", "XZ",
    "YI", "YX", "YY", "YZ",
    "ZI", "ZX", "ZY", "ZZ",
]

# Default biased weights for 2q Pauli noise (phase-heavy).
# These are weights, not probabilities. They are normalized to sum to p_2q.
DEFAULT_P2_WEIGHTS = {
    "IX": 1.0,
    "IY": 0.2,
    "IZ": 4.0,
    "XI": 1.0,
    "XX": 0.2,
    "XY": 0.1,
    "XZ": 1.2,
    "YI": 0.2,
    "YX": 0.1,
    "YY": 0.05,
    "YZ": 0.3,
    "ZI": 4.0,
    "ZX": 1.2,
    "ZY": 0.3,
    "ZZ": 6.0,
}


class SteaneTableauSimulation:
    """One 5-node simulation instance with fixed global noise parameters.

    Args:
        p_1q: Probability used in DEPOLARIZE1 after each 1-qubit gate.
        p_2q: Total probability mass used in PAULI_CHANNEL_2 after each 2-qubit gate.
        p2_weights: Optional custom weights for the 15 two-qubit Pauli error terms.
        p_meas: Measurement bit-flip probability (modeled as pre-measure X_ERROR).
        seed: Optional simulator seed for reproducibility.

    Returns:
        None.
    """

    def __init__(self, p_1q: float, p_2q: float, p2_weights: dict[str, float] | None = None, p_meas: float = 0.0, seed: int | None = None) -> None:
        if not (0.0 <= p_1q <= 1.0):
            raise ValueError("p_1q must be in [0, 1].")
        if not (0.0 <= p_2q <= 1.0):
            raise ValueError("p_2q must be in [0, 1].")
        if not (0.0 <= p_meas <= 1.0):
            raise ValueError("p_meas must be in [0, 1].")

        self.p_1q = p_1q
        self.p_2q = p_2q
        self.p_meas = p_meas
        self.seed = seed

        # Build ordered PAULI_CHANNEL_2 probabilities from weight map.
        weight_map = DEFAULT_P2_WEIGHTS.copy()
        if p2_weights is not None:
            for key, value in p2_weights.items():
                if key not in weight_map:
                    raise ValueError(f"Unknown 2q Pauli key '{key}'.")
                if value < 0:
                    raise ValueError(f"Weight for '{key}' must be non-negative.")
                weight_map[key] = float(value)

        weight_total = sum(weight_map[k] for k in PAULI2_ORDER)
        if weight_total <= 0 and p_2q > 0:
            raise ValueError("At least one 2q weight must be > 0 when p_2q > 0.")

        if p_2q == 0 or weight_total == 0:
            self.p2_probs = [0.0] * len(PAULI2_ORDER)
        else:
            self.p2_probs = [p_2q * (weight_map[k] / weight_total) for k in PAULI2_ORDER]

        # Prepare a simulator for all data/communication qubits plus FT ancillas.
        self.sim = stim.TableauSimulator(seed=seed)
        self.sim.set_num_qubits(TOTAL_QUBITS)

    def _h(self, q: int) -> None:
        """Apply H plus 1q depolarizing noise.

        Args:
            q: Target qubit index.

        Returns:
            None.
        """
        self.sim.h(q)
        if self.p_1q > 0:
            self.sim.do(stim.Circuit(f"DEPOLARIZE1({self.p_1q}) {q}"))

    def _measure_z_stabilizer(self, block_name: str, support: list[int], ancilla_q: int) -> int:
        """Measure one Z stabilizer on a block using one ancilla qubit.

        Args:
            block_name: Logical block name in BLOCK_QUBITS.
            support: Data-qubit positions (0..6) where Z acts.
            ancilla_q: Physical ancilla qubit index.

        Returns:
            int: Syndrome bit (0/1).
        """
        if block_name not in BLOCK_QUBITS:
            raise ValueError(f"Unknown block_name '{block_name}'.")

        data = BLOCK_QUBITS[block_name]
        self.sim.reset_z(ancilla_q)
        for i in support:
            self._cx(data[i], ancilla_q)
        return self._measure_z(ancilla_q)

    def _cx(self, c: int, t: int) -> None:
        """Apply CX plus biased 2q Pauli channel noise.

        Args:
            c: Control qubit index.
            t: Target qubit index.

        Returns:
            None.
        """
        self.sim.cx(c, t)
        if self.p_2q > 0:
            ptxt = ", ".join(f"{p:.12g}" for p in self.p2_probs)
            self.sim.do(stim.Circuit(f"PAULI_CHANNEL_2({ptxt}) {c} {t}"))

    def encode_steane_block(self, block_name: str, logical_state: str, ft_style: str = "none") -> bool:
        """Encode one named Steane block into |0_L> or |+_L>.

        Args:
            block_name: One of BLOCK_QUBITS keys (e.g., "A", "B_L", ...).
            logical_state: Target logical basis state label ("0" or "+").
            ft_style: Fault-tolerant preparation mode ("none", "minimal", "standard").

        Returns:
            bool: Whether this block was accepted by preparation/postselection.
        """
        if block_name not in BLOCK_QUBITS:
            raise ValueError(f"Unknown block_name '{block_name}'.")
        if ft_style not in {"none", "minimal", "standard"}:
            raise ValueError("ft_style must be 'none', 'minimal', or 'standard'.")

        data = BLOCK_QUBITS[block_name]
        q0, q1, q2, q3, q4, q5, q6 = data

        # Build |0_L> with the same Steane encoder wiring used in prior files.
        self._h(q4)
        self._h(q5)
        self._h(q6)

        self._cx(q0, q1)
        self._cx(q0, q2)
        self._cx(q6, q3)
        self._cx(q6, q1)
        self._cx(q6, q0)
        self._cx(q5, q3)
        self._cx(q5, q2)
        self._cx(q5, q0)
        self._cx(q4, q3)
        self._cx(q4, q2)
        self._cx(q4, q1)

        # FT postselection checks on prepared |0_L>.
        accepted = True
        anc = FT_ANCILLA_QUBITS[block_name]

        if ft_style == "minimal":
            s = self._measure_z_stabilizer(block_name, Z_STABILIZER_MINIMAL, anc[0])
            accepted = (s == 0)
        elif ft_style == "standard":
            syndromes = [
                self._measure_z_stabilizer(block_name, Z_STABILIZERS_STANDARD[0], anc[0]),
                self._measure_z_stabilizer(block_name, Z_STABILIZERS_STANDARD[1], anc[1]),
                self._measure_z_stabilizer(block_name, Z_STABILIZERS_STANDARD[2], anc[2]),
                self._measure_z_stabilizer(block_name, Z_STABILIZERS_STANDARD[3], anc[3]),
            ]
            accepted = all(bit == 0 for bit in syndromes)

        if not accepted:
            return False

        # Convert accepted |0_L> to |+_L> when requested.
        if logical_state == "+":
            for q in data:
                self._h(q)
        elif logical_state != "0":
            raise ValueError("logical_state must be '0' or '+'.")

        return True

    def transerse_cnot(self, control_block: str, target_block: str) -> None:
        """Apply a transversal CNOT from one logical block to another.

        Args:
            control_block: Name of the 7-qubit control block.
            target_block: Name of the 7-qubit target block.

        Returns:
            None.
        """
        if control_block not in BLOCK_QUBITS:
            raise ValueError(f"Unknown control_block '{control_block}'.")
        if target_block not in BLOCK_QUBITS:
            raise ValueError(f"Unknown target_block '{target_block}'.")

        control_qubits = BLOCK_QUBITS[control_block]
        target_qubits = BLOCK_QUBITS[target_block]

        # Pairwise physical CNOTs implement the logical transversal CNOT.
        for c, t in zip(control_qubits, target_qubits):
            self._cx(c, t)

    def teleported_cnot_block(
        self,
        control_data_block: str,
        control_comm_block: str,
        target_comm_block: str,
        target_data_block: str,
        apply_corrections: bool = True,
    ) -> dict[str, object]:
        """Run one transversal teleported CNOT with pre-shared comm Bell state.

        Args:
            control_data_block: Block containing control data qubits s1.
            control_comm_block: Block containing control-side communication qubits c1.
            target_comm_block: Block containing target-side communication qubits c2.
            target_data_block: Block containing target data qubits s2.
            apply_corrections: If True, apply feedforward corrections to data rails.

        Returns:
            dict[str, object]: Raw measurement bits and correction bits used.
        """
        if control_data_block not in BLOCK_QUBITS:
            raise ValueError(f"Unknown control_data_block '{control_data_block}'.")
        if control_comm_block not in BLOCK_QUBITS:
            raise ValueError(f"Unknown control_comm_block '{control_comm_block}'.")
        if target_comm_block not in BLOCK_QUBITS:
            raise ValueError(f"Unknown target_comm_block '{target_comm_block}'.")
        if target_data_block not in BLOCK_QUBITS:
            raise ValueError(f"Unknown target_data_block '{target_data_block}'.")

        s1 = BLOCK_QUBITS[control_data_block]
        c1 = BLOCK_QUBITS[control_comm_block]
        c2 = BLOCK_QUBITS[target_comm_block]
        s2 = BLOCK_QUBITS[target_data_block]

        # Local unitaries (orange boxes in the nonlocal-CNOT figure), transversally.
        for i in range(7):
            self._cx(s1[i], c1[i])  # CNOT(s1 -> c1)
        for i in range(7):
            self._cx(c2[i], s2[i])  # CNOT(c2 -> s2)

        # Projective measurements (green box): c1 in Z, c2 in X.
        m_z_c1 = [self._measure_z(q) for q in c1]
        m_x_c2 = [self._measure_x(q) for q in c2]

        # Feedforward from measurement outcomes:
        # Z on s1 from X(c2), X on s2 from Z(c1), per rail.
        if apply_corrections:
            for i in range(7):
                if m_x_c2[i] == 1:
                    self.sim.z(s1[i])
                if m_z_c1[i] == 1:
                    self.sim.x(s2[i])

        return {
            "m_z_c1": m_z_c1,
            "m_x_c2": m_x_c2,
            "apply_corrections": apply_corrections,
        }

    def prepare_comm_bell_pair(self, left_comm_block: str, right_comm_block: str) -> None:
        """Prepare physical Bell pairs on two communication blocks.

        Args:
            left_comm_block: Left communication block name.
            right_comm_block: Right communication block name.

        Returns:
            None.
        """
        if left_comm_block not in BLOCK_QUBITS:
            raise ValueError(f"Unknown left_comm_block '{left_comm_block}'.")
        if right_comm_block not in BLOCK_QUBITS:
            raise ValueError(f"Unknown right_comm_block '{right_comm_block}'.")

        left = BLOCK_QUBITS[left_comm_block]
        right = BLOCK_QUBITS[right_comm_block]

        # Initialize 7 independent physical Bell pairs (one per rail).
        for lq, rq in zip(left, right):
            self._h(lq)
            self._cx(lq, rq)

    def _measure_z(self, q: int) -> int:
        """Measure one qubit in Z basis with measurement noise.

        Args:
            q: Qubit index.

        Returns:
            int: Measured bit (0 or 1).
        """
        # Readout noise is modeled as a bit flip right before measurement.
        if self.p_meas > 0:
            self.sim.do(stim.Circuit(f"X_ERROR({self.p_meas}) {q}"))
        return int(self.sim.measure(q))

    def _measure_x(self, q: int) -> int:
        """Measure one qubit in X basis with gate+measurement noise.

        Args:
            q: Qubit index.

        Returns:
            int: Measured bit (0 or 1).
        """
        # X-basis measurement is implemented by H then Z measurement.
        self._h(q)
        return self._measure_z(q)

    def measure_block(self, block_name: str, basis: str) -> list[int]:
        """Measure all 7 qubits of a block in the requested basis.

        Args:
            block_name: One of BLOCK_QUBITS keys.
            basis: Measurement basis ("X" or "Z").

        Returns:
            list[int]: Seven measured bits in physical-index order.
        """
        if block_name not in BLOCK_QUBITS:
            raise ValueError(f"Unknown block_name '{block_name}'.")
        if basis not in {"X", "Z"}:
            raise ValueError("basis must be 'X' or 'Z'.")

        bits: list[int] = []
        for q in BLOCK_QUBITS[block_name]:
            if basis == "X":
                bits.append(self._measure_x(q))
            else:
                bits.append(self._measure_z(q))
        return bits

    def decode_middle_bsm(self, left_x_bits: list[int], right_z_bits: list[int]) -> dict[str, object]:
        """Decode Steane syndromes and correct middle-node BSM bitstrings.

        Args:
            left_x_bits: Seven X-basis bits from left block measurement.
            right_z_bits: Seven Z-basis bits from right block measurement.

        Returns:
            dict[str, object]: Syndrome bits, decoded flips, corrected bits, and raw/corrected 2-bit outputs.
        """
        # s_x = H*x mod 2 using hardcoded Steane check rows.
        s_x = [
            left_x_bits[3] ^ left_x_bits[4] ^ left_x_bits[5] ^ left_x_bits[6],
            left_x_bits[1] ^ left_x_bits[2] ^ left_x_bits[5] ^ left_x_bits[6],
            left_x_bits[0] ^ left_x_bits[2] ^ left_x_bits[4] ^ left_x_bits[6],
        ]

        # s_z = H*z mod 2 using the same check rows.
        s_z = [
            right_z_bits[3] ^ right_z_bits[4] ^ right_z_bits[5] ^ right_z_bits[6],
            right_z_bits[1] ^ right_z_bits[2] ^ right_z_bits[5] ^ right_z_bits[6],
            right_z_bits[0] ^ right_z_bits[2] ^ right_z_bits[4] ^ right_z_bits[6],
        ]

        x_flip_qubit = DECODE_TABLE[tuple(s_x)]
        z_flip_qubit = DECODE_TABLE[tuple(s_z)]

        # Apply classical correction to the measured bitstrings.
        x_corrected = left_x_bits.copy()
        z_corrected = right_z_bits.copy()
        if x_flip_qubit is not None:
            x_corrected[x_flip_qubit] ^= 1
        if z_flip_qubit is not None:
            z_corrected[z_flip_qubit] ^= 1

        # Compute two-bit outputs before and after classical correction.
        b_x_raw = sum(left_x_bits) & 1
        b_z_raw = sum(right_z_bits) & 1
        b_x_corrected = sum(x_corrected) & 1
        b_z_corrected = sum(z_corrected) & 1

        return {
            "s_x": s_x,
            "s_z": s_z,
            "x_flip_qubit": x_flip_qubit,
            "z_flip_qubit": z_flip_qubit,
            "x_corrected": x_corrected,
            "z_corrected": z_corrected,
            "b_x_raw": b_x_raw,
            "b_z_raw": b_z_raw,
            "b_x_corrected": b_x_corrected,
            "b_z_corrected": b_z_corrected,
        }

    def bell_measure_middle_node(self, left_block: str, right_block: str) -> dict[str, object]:
        """Run logical Bell measurement + classical decode at one middle node.

        Args:
            left_block: Left 7-qubit block of the node.
            right_block: Right 7-qubit block of the node.

        Returns:
            dict[str, object]: Raw BSM bits and decoded/corrected metadata.
        """
        # Entangle local left/right logical halves at this middle node.
        self.transerse_cnot(control_block=left_block, target_block=right_block)

        # Then measure left in X basis and right in Z basis.
        left_x_bits = self.measure_block(left_block, basis="X")
        right_z_bits = self.measure_block(right_block, basis="Z")
        decoded = self.decode_middle_bsm(left_x_bits=left_x_bits, right_z_bits=right_z_bits)
        return {
            "left_x_bits": left_x_bits,
            "right_z_bits": right_z_bits,
            **decoded,
        }

    def apply_logical_pauli_frame(self, block_name: str, b_x: int, b_z: int) -> None:
        """Apply a logical Pauli frame update to one 7-qubit block.

        Args:
            block_name: Name of the logical block to update.
            b_x: If 1, apply logical X (transversal X on all 7 qubits).
            b_z: If 1, apply logical Z (transversal Z on all 7 qubits).

        Returns:
            None.
        """
        if block_name not in BLOCK_QUBITS:
            raise ValueError(f"Unknown block_name '{block_name}'.")

        # Use transversal Paulis to represent a logical frame update.
        for q in BLOCK_QUBITS[block_name]:
            if b_x == 1:
                self.sim.x(q)
            if b_z == 1:
                self.sim.z(q)

    def logical_bell_correlators(self, left_block: str, right_block: str) -> float:
        """Compute only the logical Bell fidelity estimate for two blocks.

        Args:
            left_block: Name of first logical block.
            right_block: Name of second logical block.

        Returns:
            float: logical_fidelity for |Phi+> using (1 + <XX> - <YY> + <ZZ>) / 4.
        """
        if left_block not in BLOCK_QUBITS:
            raise ValueError(f"Unknown left_block '{left_block}'.")
        if right_block not in BLOCK_QUBITS:
            raise ValueError(f"Unknown right_block '{right_block}'.")

        def corr(basis: str) -> float:
            obs = stim.PauliString(self.sim.num_qubits)
            for q in BLOCK_QUBITS[left_block]:
                obs[q] = basis
            for q in BLOCK_QUBITS[right_block]:
                obs[q] = basis
            return float(self.sim.peek_observable_expectation(obs))

        xx = corr("X")
        yy = corr("Y")
        zz = corr("Z")
        logical_fidelity = (1.0 + xx - yy + zz) / 4.0
        return logical_fidelity

    def generate_logical_pair(
        self,
        use_syndrome_corrected_frame_bits: bool = True,
        apply_frame: bool = True,
        ft_style: str = "none",
    ) -> dict[str, object]:
        """Run one full 5-node protocol shot and return final logical-pair metrics.

        Args:
            use_syndrome_corrected_frame_bits: If True, use syndrome-corrected BSM bits for frame propagation.
            apply_frame: If True, apply computed frame to end node E before final fidelity readout.
            ft_style: Data-block preparation mode ("none", "minimal", "standard").

        Returns:
            dict[str, object]: Middle-node BSM outputs, frame bits, and final A-E logical fidelities.
        """
        if ft_style not in {"none", "minimal", "standard"}:
            raise ValueError("ft_style must be 'none', 'minimal', or 'standard'.")

        # 1) Prepare data blocks (encoded qubits).
        data_plan = [
            ("A", "+"),
            ("B_L", "0"),
            ("B_R", "+"),
            ("C_L", "0"),
            ("C_R", "+"),
            ("D_L", "0"),
            ("D_R", "+"),
            ("E", "0"),
        ]
        for block, state in data_plan:
            accepted = self.encode_steane_block(block, state, ft_style=ft_style)
            if not accepted:
                return {"accepted": False, "failed_block": block, "ft_style": ft_style}

        # 2) Prepare communication Bell-pair resources (separate from data encoding).
        comm_plan = [
            ("A_to_BL", "BL_to_A"),
            ("BR_to_CL", "CL_to_BR"),
            ("CR_to_DL", "DL_to_CR"),
            ("DR_to_E", "E_to_DR"),
        ]
        for left_c, right_c in comm_plan:
            self.prepare_comm_bell_pair(left_c, right_c)

        # 3) Run teleported CNOTs over each inter-node link.
        tcnot_plan = [
            ("A", "A_to_BL", "BL_to_A", "B_L", "ab"),
            ("B_R", "BR_to_CL", "CL_to_BR", "C_L", "bc"),
            ("C_R", "CR_to_DL", "DL_to_CR", "D_L", "cd"),
            ("D_R", "DR_to_E", "E_to_DR", "E", "de"),
        ]
        tcnot_results: dict[str, dict[str, object]] = {}
        for control, c1, c2, target, tag in tcnot_plan:
            tcnot_results[tag] = self.teleported_cnot_block(
                control_data_block=control,
                control_comm_block=c1,
                target_comm_block=c2,
                target_data_block=target,
                apply_corrections=True,
            )

        # 4) Run middle-node Bell measurements and decode syndrome bits.
        bob_bsm = self.bell_measure_middle_node(left_block="B_L", right_block="B_R")
        charlie_bsm = self.bell_measure_middle_node(left_block="C_L", right_block="C_R")
        david_bsm = self.bell_measure_middle_node(left_block="D_L", right_block="D_R")

        if use_syndrome_corrected_frame_bits:
            bx_key = "b_x_corrected"
            bz_key = "b_z_corrected"
        else:
            bx_key = "b_x_raw"
            bz_key = "b_z_raw"

        # 4) Combine frame bits across swaps (xor along chain).
        # IMPORTANT: For this BSM convention (left in X, right in Z),
        # the end-node Pauli frame uses swapped channels:
        # X-frame comes from Z-side parity bits, and Z-frame comes from X-side parity bits.
        frame_bx = bob_bsm[bz_key] ^ charlie_bsm[bz_key] ^ david_bsm[bz_key]
        frame_bz = bob_bsm[bx_key] ^ charlie_bsm[bx_key] ^ david_bsm[bx_key]

        # 5) Evaluate final A-E logical pair before/after optional frame application.
        logical_fidelity_before = self.logical_bell_correlators(left_block="A", right_block="E")
        if apply_frame:
            self.apply_logical_pauli_frame(block_name="E", b_x=frame_bx, b_z=frame_bz)
        logical_fidelity_after = self.logical_bell_correlators(left_block="A", right_block="E")

        return {
            "accepted": True,
            "failed_block": None,
            "ft_style": ft_style,
            "bob_bsm": bob_bsm,
            "charlie_bsm": charlie_bsm,
            "david_bsm": david_bsm,
            "tcnot_ab": tcnot_results["ab"],
            "tcnot_bc": tcnot_results["bc"],
            "tcnot_cd": tcnot_results["cd"],
            "tcnot_de": tcnot_results["de"],
            "frame_bx": frame_bx,
            "frame_bz": frame_bz,
            "logical_fidelity_before": logical_fidelity_before,
            "logical_fidelity_after": logical_fidelity_after,
            "use_syndrome_corrected_frame_bits": use_syndrome_corrected_frame_bits,
            "apply_frame": apply_frame,
        }

    def run_multi_shot(
        self,
        shots: int,
        master_seed: int,
        use_syndrome_corrected_frame_bits: bool = True,
        apply_frame: bool = True,
        ft_style: str = "none",
    ) -> dict[str, float]:
        """Run many independent shots and return average logical fidelities.

        Args:
            shots: Number of Monte Carlo shots.
            master_seed: Base seed used to derive per-shot seeds.
            use_syndrome_corrected_frame_bits: If True, use corrected BSM bits for frame propagation.
            apply_frame: If True, apply the computed Pauli frame to end node E.
            ft_style: Data-block preparation mode ("none", "minimal", "standard").

        Returns:
            dict[str, float]: Average fidelities before/after frame over all shots.
        """
        f_before: list[float] = []
        f_after: list[float] = []
        accepted_shots = 0

        for shot_id in range(shots):
            shot_seed = master_seed * 100000 + shot_id
            shot_sim = SteaneTableauSimulation(
                p_1q=self.p_1q,
                p_2q=self.p_2q,
                p2_weights=None,
                p_meas=self.p_meas,
                seed=shot_seed,
            )
            out = shot_sim.generate_logical_pair(
                use_syndrome_corrected_frame_bits=use_syndrome_corrected_frame_bits,
                apply_frame=apply_frame,
                ft_style=ft_style,
            )
            if not out.get("accepted", False):
                continue
            accepted_shots += 1
            f_before.append(float(out["logical_fidelity_before"]))
            f_after.append(float(out["logical_fidelity_after"]))

        if accepted_shots == 0:
            return {
                "accepted_shots": 0.0,
                "acceptance_rate": 0.0,
                "avg_fidelity_before_frame": 0.0,
                "avg_fidelity_after_frame": 0.0,
            }

        return {
            "accepted_shots": float(accepted_shots),
            "acceptance_rate": accepted_shots / shots,
            "avg_fidelity_before_frame": statistics.fmean(f_before),
            "avg_fidelity_after_frame": statistics.fmean(f_after),
        }

def main() -> None:
    """Run a multi-shot sweep using per-shot seeds.

    Args:
        None.

    Returns:
        None.
    """
    # -----------------------
    # User-tunable parameters
    # -----------------------
    # Change this to reproduce or vary random trajectories.
    # Same seed -> same shot sequence.
    master_seed = 7

    # Number of Monte Carlo shots.
    # Larger values reduce statistical noise but take longer.
    shots = 1000

    # Physical error rates:
    # p_1q  -> DEPOLARIZE1 probability applied after each modeled 1-qubit gate.
    #          In this circuit model, noisy 1q gates are mainly H
    #          (including X-basis readout implemented as H + MZ).
    # p_2q  -> total probability for PAULI_CHANNEL_2 after each CX.
    # p_meas-> readout bit-flip probability before each measurement.
    p_1q = 0.0001
    p_2q = 0.0001
    p_meas = 0.0001

    # Protocol toggles:
    # ft_style: "none", "minimal", or "standard"
    # use_syndrome_corrected_frame_bits:
    # use syndrome-corrected middle-node bits for frame propagation.
    # apply_frame: apply final Pauli frame correction at the end node
    ft_style = "none"
    use_syndrome_corrected_frame_bits = True
    apply_frame = True

    simulation = SteaneTableauSimulation(
        p_1q=p_1q,
        p_2q=p_2q,
        p2_weights=None,
        p_meas=p_meas,
        seed=master_seed,
    )

    summary = simulation.run_multi_shot(
        shots=shots,
        master_seed=master_seed,
        use_syndrome_corrected_frame_bits=use_syndrome_corrected_frame_bits,
        apply_frame=apply_frame,
        ft_style=ft_style,
    )

    print(
        f"shots={shots}, master_seed={master_seed}, "
        f"p_1q={simulation.p_1q}, p_2q={simulation.p_2q}, p_meas={simulation.p_meas}"
    )
    print("Total qubits allocated:", TOTAL_QUBITS)
    print("FT ancillas per block:", len(next(iter(FT_ANCILLA_QUBITS.values()))))
    print(f"accepted shots: {int(summary['accepted_shots'])}/{shots} ({summary['acceptance_rate']:.6f})")
    print(f"avg logical fidelity before frame: {summary['avg_fidelity_before_frame']:.6f}")
    print(f"avg logical fidelity after frame:  {summary['avg_fidelity_after_frame']:.6f}")


if __name__ == "__main__":
    main()
