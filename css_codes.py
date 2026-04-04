"""CSS code definitions used by RequestLogicalPairApp.

This module keeps a compact API centered on:
1) encoding circuits,
2) fault-tolerant preparation circuits,
3) decode-table metadata.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

import stim


def get_css_code(name: str) -> CSSCode:
    """Look up a CSS code instance by name.

    Args:
        name: Code identifier.

    Returns:
        CSSCode: Registered code instance.
    """
    if name not in CSS_CODE_REGISTRY:
        available = ", ".join(sorted(CSS_CODE_REGISTRY.keys()))
        raise KeyError(f"CSS code '{name}' not found. Available: {available}")
    return CSS_CODE_REGISTRY[name]


def _rows_to_pauli_strings(rows: List[List[int]], pauli: str) -> List[str]:
    """Convert binary matrix rows into Pauli-support strings.

    Args:
        rows: Binary rows where 1 indicates Pauli support.
        pauli: Pauli character ("X" or "Z").

    Returns:
        List[str]: Pauli strings over "I"/pauli.
    """
    return ["".join(pauli if bit else "I" for bit in row) for row in rows]


def _logical_zero_prep_circuit(x_stabilizers: List[str], z_stabilizers: List[str], logical_z: str) -> stim.Circuit:
    """Build a tableau-synthesized |0_L> preparation circuit.

    Args:
        x_stabilizers: X stabilizer strings.
        z_stabilizers: Z stabilizer strings.
        logical_z: Logical-Z operator string.

    Returns:
        stim.Circuit: Circuit preparing the logical zero state.
    """
    generators: List[str] = []
    generators.extend("+" + s.replace("I", "_") for s in x_stabilizers)
    generators.extend("+" + s.replace("I", "_") for s in z_stabilizers)
    generators.append("+" + logical_z.replace("I", "_"))

    tableau = stim.Tableau.from_stabilizers([stim.PauliString(g) for g in generators],allow_redundant=False)
    return tableau.to_circuit()


class CSSCode(ABC):
    """Base class for CSS quantum error-correcting codes.

    Args:
        name: Code identifier, e.g. "[[7,1,3]]".
        n: Number of physical qubits.
        k: Number of logical qubits.
        d: Code distance.
        x_support: Support indices for logical X.
        z_support: Support indices for logical Z.
    """

    def __init__(self, name: str, n: int, k: int, d: int, x_support: List[int], z_support: List[int]) -> None:
        """Initialize core code metadata.

        Args:
            name: Code identifier.
            n: Number of physical qubits.
            k: Number of logical qubits.
            d: Code distance.
            x_support: Logical-X support indices.
            z_support: Logical-Z support indices.

        Returns:
            None
        """
        self.name = name
        self.n = n
        self.k = k
        self.d = d
        self.x_support = list(x_support)
        self.z_support = list(z_support)

    @abstractmethod
    def encode(self, circuit: stim.Circuit) -> None:
        """Append an encoding circuit for this code.

        Args:
            circuit: Output Stim circuit to append to.

        Returns:
            None
        """

    @abstractmethod
    def get_x_stabilizer_strings(self) -> List[str]:
        """Return X-type stabilizer strings.

        Args:
            None.

        Returns:
            List[str]: X-stabilizer Pauli strings.
        """

    @abstractmethod
    def get_z_stabilizer_strings(self) -> List[str]:
        """Return Z-type stabilizer strings.

        Args:
            None.

        Returns:
            List[str]: Z-stabilizer Pauli strings.
        """

    def get_logical_x_string(self) -> str:
        """Return logical X support as a Pauli string.

        Args:
            None.

        Returns:
            str: Logical-X Pauli string.
        """
        chars = ["I"] * self.n
        for qubit in self.x_support:
            chars[qubit] = "X"
        return "".join(chars)

    def get_logical_z_string(self) -> str:
        """Return logical Z support as a Pauli string.

        Args:
            None.

        Returns:
            str: Logical-Z Pauli string.
        """
        chars = ["I"] * self.n
        for qubit in self.z_support:
            chars[qubit] = "Z"
        return "".join(chars)

    def get_ft_prep_circuit(self, mode: str, ancilla_offset: int = 0) -> stim.Circuit:
        """Return a fault-tolerant preparation check circuit for this code.

        Args:
            mode: FT preparation mode.
            ancilla_offset: Ancilla-block qubit offset.

        Returns:
            stim.Circuit: FT prep/check circuit.
        """
        if mode == "none":
            return stim.Circuit()

        supports = self.get_ft_check_supports(mode)

        circuit = stim.Circuit()
        for check_index, support in enumerate(supports):
            anc = ancilla_offset + check_index
            circuit.append("R", [anc])
            for data_index in support:
                circuit.append("CX", [data_index, anc])
            circuit.append("M", [anc])
        return circuit

    def get_ft_check_supports(self, mode: str) -> List[List[int]]:
        """Return Z-check supports used for FT prep.

        Args:
            mode: FT prep mode.

        Returns:
            List[List[int]]: Data-index supports for each check.
        """
        if mode == "none":
            return []

        supports = []
        for stab in self.get_z_stabilizer_strings():
            support = [i for i, p in enumerate(stab) if p in ("Z", "Y")]
            if support:
                supports.append(support)

        if len(supports) == 0:
            raise RuntimeError(f"{self.name}: no Z checks available")
        if mode == "minimal":
            return supports[:1]
        if mode == "standard":
            return supports
        raise RuntimeError(f"{self.name}: unknown ft_prep_mode {mode}")

    def run_encode_ft_prep(self, simulator: stim.TableauSimulator, data_locals: List[int], ancilla_locals: List[int], ft_prep_mode: str, max_ft_prep_shots: int, logical_state: str = "0") -> bool:
        """Run encode plus optional FT prep shots on a tableau simulator.

        Args:
            simulator: Stim tableau simulator.
            data_locals: Local tableau indices for data qubits.
            ancilla_locals: Local tableau indices for ancillas.
            ft_prep_mode: FT prep mode.
            max_ft_prep_shots: Maximum retry shots.
            logical_state: Target encoded logical state ("0" or "+").

        Returns:
            bool: True if accepted, False otherwise.
        """
        if max_ft_prep_shots < 1:
            raise RuntimeError(f"{self.name}: max_ft_prep_shots must be >= 1")
        if logical_state not in {"0", "+"}:
            raise RuntimeError(f"{self.name}: unknown logical_state {logical_state}")

        supports = self.get_ft_check_supports(ft_prep_mode)
        if len(ancilla_locals) < len(supports):
            raise RuntimeError(f"{self.name}: insufficient ancillas for {ft_prep_mode} FT prep")

        enc = stim.Circuit()
        self.encode(enc)

        for _ in range(max_ft_prep_shots):
            # Fresh shot.
            for q_local in data_locals:
                simulator.reset_z(q_local)
            for a_local in ancilla_locals[:len(supports)]:
                simulator.reset_z(a_local)

            simulator.do(enc)

            if ft_prep_mode == "none":
                if logical_state == "+":
                    for q_local in data_locals:
                        simulator.h(q_local)
                return True

            accepted = True
            for check_idx, support in enumerate(supports):
                anc_local = ancilla_locals[check_idx]
                simulator.reset_z(anc_local)
                for i in support:
                    simulator.cx(data_locals[i], anc_local)
                if int(simulator.measure(anc_local)) != 0:
                    accepted = False
                    break

            if accepted:
                if logical_state == "+":
                    for q_local in data_locals:
                        simulator.h(q_local)
                return True

        return False


    def get_decode_table(self) -> Dict[tuple[int, int, int], int | None]:
        """Return a syndrome decode table when available.

        Args:
            None.

        Returns:
            Dict[tuple[int, int, int], int | None]: Syndrome->qubit map.
        """
        return {}

    def decode_middle_bsm(self, left_x_bits: List[int], right_z_bits: List[int], correction_mode: str) -> dict[str, object]:
        """Decode middle-node BSM bitstrings into corrected parity data.

        Args:
            left_x_bits: X-basis measurement bits from the left logical block.
            right_z_bits: Z-basis measurement bits from the right logical block.
            correction_mode: Correction mode controlling whether classical syndrome correction is applied.

        Returns:
            dict[str, object]: Syndrome bits, flip indices, corrected strings, and corrected parity bits.
        """
        raise NotImplementedError(f"{self.name}: middle-node BSM decoding is not implemented")

    def run_qec_round(self, simulator: stim.TableauSimulator, data_locals: List[int], ancilla_locals: List[int], apply_physical_correction: bool = True) -> dict[str, object]:
        """Run one local QEC round on an encoded data block.

        Args:
            simulator: Tableau simulator containing the encoded data block.
            data_locals: Local simulator indices for the data qubits.
            ancilla_locals: Local simulator indices for ancilla qubits.
            apply_physical_correction: Whether to apply the decoded correction to the data block.

        Returns:
            dict[str, object]: Measured syndromes and decoded correction data.
        """
        raise NotImplementedError(f"{self.name}: qec round is not implemented")
    
    def get_ft_required_ancillas(self, mode: str) -> int:
        """Return required ancilla count for FT prep mode.

        Args:
            mode: FT prep mode.

        Returns:
            int: Required ancilla count.
        """
        if mode == "none":
            return 0
        if mode == "minimal":
            return 1
        if mode == "standard":
            return len(self.get_z_stabilizer_strings())
        raise RuntimeError(f"{self.name}: unknown ft_prep_mode {mode}")


    def __repr__(self) -> str:
        """Return compact code representation.

        Args:
            None.

        Returns:
            str: Debug representation.
        """
        return f"CSSCode({self.name}, n={self.n}, k={self.k}, d={self.d})"


class Steane713(CSSCode):
    """[[7,1,3]] Steane code."""

    STEANE_SYNDROME_ROWS: List[List[int]] = [
        [3, 4, 5, 6],
        [1, 2, 5, 6],
        [0, 2, 4, 6],
    ]
    STEANE_FT_CHECK_SUPPORTS_STANDARD: List[List[int]] = [
        [3, 4, 5, 6],
        [0, 5, 6],
        [1, 4, 6],
        [2, 4, 5],
    ]

    def __init__(self) -> None:
        """Initialize Steane [[7,1,3]] metadata.

        Args:
            None.

        Returns:
            None
        """
        super().__init__(name="[[7,1,3]]", n=7, k=1, d=3, x_support=[0, 1, 2, 3, 4, 5, 6], z_support=[0, 1, 2, 3, 4, 5, 6])

    def encode(self, circuit: stim.Circuit) -> None:
        """Append Paetznick-Reichardt style Steane encoder (8 CNOT form).

        Args:
            circuit: Output Stim circuit to append to.

        Returns:
            None
        """
        q0 = 0
        q1 = 1
        q2 = 2
        q3 = 3
        q4 = 4
        q5 = 5
        q6 = 6

        circuit.append("H", [q1, q2, q3])
        circuit.append("CX", [q1, q0])
        circuit.append("CX", [q3, q5])
        circuit.append("CX", [q2, q6])
        circuit.append("CX", [q1, q4])
        circuit.append("CX", [q2, q0])
        circuit.append("CX", [q3, q6])
        circuit.append("CX", [q1, q5])
        circuit.append("CX", [q6, q4])

    def get_x_stabilizer_strings(self) -> List[str]:
        """Return Steane X stabilizers.

        Args:
            None.

        Returns:
            List[str]: X stabilizer strings.
        """
        return ["IIIXXXX", "IXXIIXX", "XIXIXIX"]

    def get_z_stabilizer_strings(self) -> List[str]:
        """Return Steane Z stabilizers.

        Args:
            None.

        Returns:
            List[str]: Z stabilizer strings.
        """
        return ["IIIZZZZ", "IZZIIZZ", "ZIZIZIZ"]

    def get_ft_check_supports(self, mode: str) -> List[List[int]]:
        """Return Steane FT check supports.

        Args:
            mode: FT prep mode.

        Returns:
            List[List[int]]: Check supports for selected mode.
        """
        if mode == "none":
            return []
        if mode == "minimal":
            return [list(self.STEANE_FT_CHECK_SUPPORTS_STANDARD[1])]
        if mode == "standard":
            return [list(support) for support in self.STEANE_FT_CHECK_SUPPORTS_STANDARD]
        raise RuntimeError(f"{self.name}: unknown ft_prep_mode {mode}")

    def get_ft_required_ancillas(self, mode: str) -> int:
        """Return required ancilla count for Steane FT prep mode.

        Args:
            mode: FT prep mode.

        Returns:
            int: Required ancilla count.
        """
        return len(self.get_ft_check_supports(mode))

    def get_decode_table(self) -> Dict[tuple[int, int, int], int | None]:
        """Return Steane 3-bit syndrome decode map.

        Args:
            None.

        Returns:
            Dict[tuple[int, int, int], int | None]: Syndrome->error qubit index.
        """
        return {
            (0, 0, 0): None,
            (0, 0, 1): 0,
            (0, 1, 0): 1,
            (0, 1, 1): 2,
            (1, 0, 0): 3,
            (1, 0, 1): 4,
            (1, 1, 0): 5,
            (1, 1, 1): 6,
        }

    def decode_middle_bsm(self, left_x_bits: List[int], right_z_bits: List[int], correction_mode: str) -> dict[str, object]:
        """Decode Steane syndromes from middle-node Bell-measurement bitstrings.

        Args:
            left_x_bits: Seven X-basis bits from the left logical block.
            right_z_bits: Seven Z-basis bits from the right logical block.
            correction_mode: Correction mode controlling whether classical syndrome correction is applied.

        Returns:
            dict[str, object]: Syndrome bits, flip indices, corrected strings, and corrected parity bits.
        """
        x_corrected = list(left_x_bits)
        z_corrected = list(right_z_bits)
        s_x = [sum(left_x_bits[index] for index in row) & 1 for row in self.STEANE_SYNDROME_ROWS]
        s_z = [sum(right_z_bits[index] for index in row) & 1 for row in self.STEANE_SYNDROME_ROWS]
        decode_table = self.get_decode_table()
        x_flip_qubit = decode_table[tuple(s_x)]
        z_flip_qubit = decode_table[tuple(s_z)]
        apply_classical_correction = correction_mode in {"cec", "qec+cec"}
        if apply_classical_correction:
            if x_flip_qubit is not None:
                x_corrected[x_flip_qubit] ^= 1
            if z_flip_qubit is not None:
                z_corrected[z_flip_qubit] ^= 1

        b_x_corrected = sum(x_corrected) & 1
        b_z_corrected = sum(z_corrected) & 1

        return {
            "s_x": s_x,
            "s_z": s_z,
            "x_flip_qubit": x_flip_qubit,
            "z_flip_qubit": z_flip_qubit,
            "x_corrected": x_corrected,
            "z_corrected": z_corrected,
            "b_x_corrected": b_x_corrected,
            "b_z_corrected": b_z_corrected,
        }

    def run_qec_round(self, simulator: stim.TableauSimulator, data_locals: List[int], ancilla_locals: List[int], apply_physical_correction: bool = True) -> dict[str, object]:
        """Run one FT Steane QEC round using the 3-ancilla syndrome schedule.

        Args:
            simulator: Tableau simulator containing the encoded data block.
            data_locals: Local simulator indices for the seven data qubits.
            ancilla_locals: Local simulator indices for ancilla qubits.
            apply_physical_correction: Whether to apply decoded corrections to the data qubits.

        Returns:
            dict[str, object]: X/Z syndromes, decoded flip locations, and applied corrections.
        """
        if len(data_locals) != self.n:
            raise RuntimeError(f"{self.name}: expected {self.n} data qubits, got {len(data_locals)}")
        if len(ancilla_locals) < 3:
            raise RuntimeError(f"{self.name}: FT qec round requires at least 3 ancillas")

        green = ancilla_locals[0]
        blue = ancilla_locals[1]
        red = ancilla_locals[2]

        def decode_color_syndrome(green_bit: int, red_bit: int, blue_bit: int) -> int | None:
            """Decode one 3-bit color syndrome into a physical qubit index.

            Args:
                green_bit: Green syndrome bit.
                red_bit: Red syndrome bit.
                blue_bit: Blue syndrome bit.

            Returns:
                int | None: Qubit index to correct, or ``None`` for the zero syndrome.
            """
            value = green_bit + 2 * red_bit + 4 * blue_bit
            if value == 0:
                return None
            return value - 1

        def extract_syndromes_ft_round1() -> tuple[int, int, int]:
            """Extract X-green, Z-blue, and Z-red syndrome bits.

            Args:
                None.

            Returns:
                tuple[int, int, int]: ``(x_green, z_blue, z_red)``.
            """
            simulator.reset_z(green)
            simulator.reset_z(blue)
            simulator.reset_z(red)

            simulator.h(green)
            simulator.cx(data_locals[6], blue)
            simulator.cx(data_locals[6], red)
            simulator.cx(green, data_locals[6])
            simulator.cx(data_locals[4], blue)
            simulator.cx(green, data_locals[4])
            simulator.cx(data_locals[2], red)
            simulator.cx(green, data_locals[2])
            simulator.cx(data_locals[5], blue)
            simulator.cx(data_locals[5], red)
            simulator.cx(green, data_locals[0])
            simulator.cx(data_locals[3], blue)
            simulator.cx(data_locals[1], red)
            simulator.h(green)

            x_green = int(simulator.measure(green))
            z_blue = int(simulator.measure(blue))
            z_red = int(simulator.measure(red))
            return x_green, z_blue, z_red

        def extract_syndromes_ft_round2() -> tuple[int, int, int]:
            """Extract Z-green, X-blue, and X-red syndrome bits.

            Args:
                None.

            Returns:
                tuple[int, int, int]: ``(z_green, x_blue, x_red)``.
            """
            simulator.reset_z(green)
            simulator.reset_z(blue)
            simulator.reset_z(red)

            simulator.h(blue)
            simulator.h(red)
            simulator.cx(blue, data_locals[6])
            simulator.cx(red, data_locals[6])
            simulator.cx(data_locals[6], green)
            simulator.cx(blue, data_locals[4])
            simulator.cx(data_locals[4], green)
            simulator.cx(red, data_locals[2])
            simulator.cx(data_locals[2], green)
            simulator.cx(blue, data_locals[5])
            simulator.cx(red, data_locals[5])
            simulator.cx(data_locals[0], green)
            simulator.cx(blue, data_locals[3])
            simulator.cx(red, data_locals[1])

            z_green = int(simulator.measure(green))
            simulator.h(blue)
            simulator.h(red)
            x_blue = int(simulator.measure(blue))
            x_red = int(simulator.measure(red))
            return z_green, x_blue, x_red

        x_green, z_blue, z_red = extract_syndromes_ft_round1()
        z_green, x_blue, x_red = extract_syndromes_ft_round2()

        x_syndrome = [z_green, z_red, z_blue]
        z_syndrome = [x_green, x_red, x_blue]

        x_error_qubit = decode_color_syndrome(z_green, z_red, z_blue)
        z_error_qubit = decode_color_syndrome(x_green, x_red, x_blue)

        applied_x = None
        applied_z = None
        if apply_physical_correction:
            if x_error_qubit is not None:
                simulator.x(data_locals[x_error_qubit])
                applied_x = int(x_error_qubit)
            if z_error_qubit is not None:
                simulator.z(data_locals[z_error_qubit])
                applied_z = int(z_error_qubit)

        return {
            "x_syndrome": x_syndrome,
            "z_syndrome": z_syndrome,
            "x_error_qubit": x_error_qubit,
            "z_error_qubit": z_error_qubit,
            "applied_x": applied_x,
            "applied_z": applied_z,
        }


class Shor9(CSSCode):
    """[[9,1,3]] Shor code."""

    def __init__(self) -> None:
        """Initialize Shor [[9,1,3]] metadata.

        Args:
            None.

        Returns:
            None
        """
        super().__init__(name="[[9,1,3]]", n=9, k=1, d=3, x_support=[0, 1, 2], z_support=[0, 3, 6])

    def encode(self, circuit: stim.Circuit) -> None:
        """Append Shor encoding circuit.

        Args:
            circuit: Output Stim circuit to append to.

        Returns:
            None
        """
        o = 0
        circuit.append("CX", [o + 0, o + 3])
        circuit.append("CX", [o + 0, o + 6])
        circuit.append("H", [o + 0, o + 3, o + 6])
        circuit.append("CX", [o + 0, o + 1])
        circuit.append("CX", [o + 0, o + 2])
        circuit.append("CX", [o + 3, o + 4])
        circuit.append("CX", [o + 3, o + 5])
        circuit.append("CX", [o + 6, o + 7])
        circuit.append("CX", [o + 6, o + 8])

    def get_x_stabilizer_strings(self) -> List[str]:
        """Return Shor X stabilizers.

        Args:
            None.

        Returns:
            List[str]: X stabilizer strings.
        """
        return ["XXXXXXIII", "IIIXXXXXX"]

    def get_z_stabilizer_strings(self) -> List[str]:
        """Return Shor Z stabilizers.

        Args:
            None.

        Returns:
            List[str]: Z stabilizer strings.
        """
        return [
            "ZZIIIIIII",
            "IZZIIIIII",
            "IIIZZIIII",
            "IIIIZZIII",
            "IIIIIIZZI",
            "IIIIIIIZZ",
        ]

    def get_logical_x_string(self) -> str:
        """Return physical-string representation of Shor logical X.

        Args:
            None.

        Returns:
            str: Logical-X Pauli string.
        """
        chars = ["I"] * self.n
        for qubit in self.z_support:
            chars[qubit] = "Z"
        return "".join(chars)

    def get_logical_z_string(self) -> str:
        """Return physical-string representation of Shor logical Z.

        Args:
            None.

        Returns:
            str: Logical-Z Pauli string.
        """
        chars = ["I"] * self.n
        for qubit in self.x_support:
            chars[qubit] = "X"
        return "".join(chars)


class ReedMuller15(CSSCode):
    """[[15,1,3]] Reed-Muller CSS code."""

    _H_X: List[List[int]] = [
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    ]
    _H_Z: List[List[int]] = [
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
        [1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1],
    ]

    def __init__(self) -> None:
        """Initialize Reed-Muller [[15,1,3]] metadata.

        Args:
            None.

        Returns:
            None
        """
        super().__init__(name="[[15,1,3]]", n=15, k=1, d=3, x_support=[0, 1, 2], z_support=[0, 1, 2, 3, 4, 5, 6])

    def encode(self, circuit: stim.Circuit) -> None:
        """Append tableau-synthesized logical-|0> preparation for RM(15).

        Args:
            circuit: Output Stim circuit to append to.

        Returns:
            None
        """
        prep = _logical_zero_prep_circuit(self.get_x_stabilizer_strings(), self.get_z_stabilizer_strings(), self.get_logical_z_string())
        circuit += prep

    def get_x_stabilizer_strings(self) -> List[str]:
        """Return Reed-Muller X stabilizers.

        Args:
            None.

        Returns:
            List[str]: X stabilizer strings.
        """
        return _rows_to_pauli_strings(self._H_X, "X")

    def get_z_stabilizer_strings(self) -> List[str]:
        """Return Reed-Muller Z stabilizers.

        Args:
            None.

        Returns:
            List[str]: Z stabilizer strings.
        """
        return _rows_to_pauli_strings(self._H_Z, "Z")


class Golay23(CSSCode):
    """[[23,1,7]] Golay CSS code."""

    _H: List[List[int]] = [
        [1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ]

    def __init__(self) -> None:
        """Initialize Golay [[23,1,7]] metadata.

        Args:
            None.

        Returns:
            None
        """
        super().__init__(name="[[23,1,7]]", n=23, k=1, d=7, x_support=[0, 1, 2, 3, 9, 12, 21], z_support=[0, 1, 2, 3, 9, 12, 21])

    def encode(self, circuit: stim.Circuit) -> None:
        """Append tableau-synthesized logical-|0> preparation for Golay(23).

        Args:
            circuit: Output Stim circuit to append to.

        Returns:
            None
        """
        prep = _logical_zero_prep_circuit(self.get_x_stabilizer_strings(), self.get_z_stabilizer_strings(), self.get_logical_z_string())
        circuit += prep

    def get_x_stabilizer_strings(self) -> List[str]:
        """Return Golay X stabilizers.

        Args:
            None.

        Returns:
            List[str]: X stabilizer strings.
        """
        return _rows_to_pauli_strings(self._H, "X")

    def get_z_stabilizer_strings(self) -> List[str]:
        """Return Golay Z stabilizers.

        Args:
            None.

        Returns:
            List[str]: Z stabilizer strings.
        """
        return _rows_to_pauli_strings(self._H, "Z")


class BCH31(CSSCode):
    """[[31,1,7]] BCH CSS code."""

    _X_STABILIZERS = [
        "XIIIXXXXIIIIXIIXXIIIIIIIIIIIIII",
        "XXIIXIIIXIIIXXIXIXIIIIIIIIIIIII",
        "XXXIXIXXIXIIXXXXIIXIIIIIIIIIIII",
        "XXXXXIXIXIXIXXXIIIIXIIIIIIIIIII",
        "IXXXXXIXIXIXIXXXIIIIXIIIIIIIIII",
        "XIXXIIIXXIXIIIXIIIIIIXIIIIIIIII",
        "IXIXXIIIXXIXIIIXIIIIIIXIIIIIIII",
        "XIXIIIXXIXXIIIIXIIIIIIIXIIIIIII",
        "XXIXXXXIXIXXXIIXIIIIIIIIXIIIIII",
        "XXXIIIIIIXIXIXIXIIIIIIIIIXIIIII",
        "XXXXXXXXIIXIIIXXIIIIIIIIIIXIIII",
        "XXXXIIIIXIIXXIIIIIIIIIIIIIIXIII",
        "IXXXXIIIIXIIXXIIIIIIIIIIIIIIXII",
        "IIXXXXIIIIXIIXXIIIIIIIIIIIIIIXI",
        "IIIXXXXIIIIXIIXXIIIIIIIIIIIIIIX",
    ]

    _Z_STABILIZERS = [
        "ZIIIZZZZIIIIZIIZZIIIIIIIIIIIIII",
        "ZZIIZIIIZIIIZZIZIZIIIIIIIIIIIII",
        "ZZZIZIZZIZIIZZZZIIZIIIIIIIIIIII",
        "ZZZZZIZIZIZIZZZIIIIZIIIIIIIIIII",
        "IZZZZZIZIZIZIZZZIIIIZIIIIIIIIII",
        "ZIZZIIIZZIZIIIZIIIIIIZIIIIIIIII",
        "IZIZZIIIZZIZIIIZIIIIIIZIIIIIIII",
        "ZIZIIIZZIZZIIIIZIIIIIIIZIIIIIII",
        "ZZIZZZZIZIZZZIIZIIIIIIIIZIIIIII",
        "ZZZIIIIIIZIZIZIZIIIIIIIIIZIIIII",
        "ZZZZZZZZIIZIIIZZIIIIIIIIIIZIIII",
        "ZZZZIIIIZIIZZIIIIIIIIIIIIIIZIII",
        "IZZZZIIIIZIIZZIIIIIIIIIIIIIIZII",
        "IIZZZZIIIIZIIZZIIIIIIIIIIIIIIZI",
        "IIIZZZZIIIIZIIZZIIIIIIIIIIIIIIZ",
    ]

    def __init__(self) -> None:
        """Initialize BCH [[31,1,7]] metadata.

        Args:
            None.

        Returns:
            None
        """
        super().__init__( name="[[31,1,7]]", n=31, k=1, d=7, x_support=[0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 15], z_support=[0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 15])

    def encode(self, circuit: stim.Circuit) -> None:
        """Append tableau-synthesized logical-|0> preparation for BCH(31).

        Args:
            circuit: Output Stim circuit to append to.

        Returns:
            None
        """
        prep = _logical_zero_prep_circuit(self.get_x_stabilizer_strings(), self.get_z_stabilizer_strings(), self.get_logical_z_string())
        circuit += prep

    def get_x_stabilizer_strings(self) -> List[str]:
        """Return BCH X stabilizers.

        Args:
            None.

        Returns:
            List[str]: X stabilizer strings.
        """
        return list(self._X_STABILIZERS)

    def get_z_stabilizer_strings(self) -> List[str]:
        """Return BCH Z stabilizers.

        Args:
            None.

        Returns:
            List[str]: Z stabilizer strings.
        """
        return list(self._Z_STABILIZERS)


CSS_CODE_REGISTRY: Dict[str, CSSCode] = {
    "[[7,1,3]]": Steane713(),
    "[[9,1,3]]": Shor9(),
    "[[15,1,3]]": ReedMuller15(),
    "[[23,1,7]]": Golay23(),
    "[[31,1,7]]": BCH31(),
}
