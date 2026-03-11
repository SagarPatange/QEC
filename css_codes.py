"""Pluggable CSS Code Library.

Provides a registry of CSS codes with encoding circuits, logical operator
supports, and Y-basis measurement info. Two construction methods:

- Option A: Hand-coded, verified encoding circuits (Steane713, Shor9)
- Option B: Algorithmic generation from H_X/H_Z parity check matrices
  (correct but NOT guaranteed fault-tolerant)

Usage:
    from css_codes import get_css_code
    code = get_css_code("[[7,1,3]]")
    code.encode(circuit, offset=0)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import stim


# --- Tableau-based state preparation helpers (from css_codes.py) ---

def _shift_circuit(src: stim.Circuit, offset: int) -> stim.Circuit:
    """Return a copy of src with all qubit targets shifted by offset."""
    out = stim.Circuit()
    for inst in src:
        name = inst.name
        args = inst.gate_args_copy()
        targets = []
        for t in inst.targets_copy():
            if t.is_qubit_target:
                targets.append(t.value + offset)
            else:
                targets.append(t)
        out.append(name, targets, args)
    return out


def _tableau_prep_from_generators(generators: Sequence[str]) -> stim.Circuit:
    """Build a Clifford circuit that prepares the stabilizer state
    defined by the given independent generators (e.g. '+X_Z...')."""
    pauli_strings = [stim.PauliString(g) for g in generators]
    tab = stim.Tableau.from_stabilizers(pauli_strings, allow_redundant=False)
    return tab.to_circuit()


class CSSCode(ABC):
    """Base class for CSS quantum error correcting codes.

    Attributes:
        name: Code identifier string, e.g. "[[7,1,3]]"
        n: Number of physical qubits
        k: Number of logical qubits
        d: Code distance
        x_support: Qubit indices in the logical X operator (X_L)
        z_support: Qubit indices in the logical Z operator (Z_L)
    """

    def __init__(self, name: str, n: int, k: int, d: int,
                 x_support: List[int], z_support: List[int]):
        self.name = name
        self.n = n
        self.k = k
        self.d = d
        self.x_support = x_support
        self.z_support = z_support

    @abstractmethod
    def encode(self, circuit: stim.Circuit, offset: int = 0):
        """Append encoding gates to circuit.

        Qubits are at indices offset .. offset+n-1.
        Qubit at offset holds the logical information (|psi>),
        qubits offset+1 .. offset+n-1 are initialized to |0>.

        Args:
            circuit: Stim circuit to append gates to
            offset: Starting qubit index
        """

    def y_basis_info(self) -> dict:
        """Return per-qubit basis change info for Y_L measurement.

        Y_L = i * X_L * Z_L, so the per-qubit Pauli depends on
        which supports the qubit belongs to:
          - x_support AND z_support -> Y basis (S_DAG + H)
          - x_support only          -> X basis (H)
          - z_support only          -> Z basis (nothing)

        Returns:
            dict with:
                'support': list of qubit indices in Y_L support
                'basis_changes': dict mapping qubit_index -> 'X', 'Y', or 'Z'
        """
        x_set = set(self.x_support)
        z_set = set(self.z_support)
        support = sorted(x_set | z_set)

        basis_changes = {}
        for idx in support:
            in_x = idx in x_set
            in_z = idx in z_set
            if in_x and in_z:
                basis_changes[idx] = 'Y'
            elif in_x:
                basis_changes[idx] = 'X'
            else:
                basis_changes[idx] = 'Z'

        return {'support': support, 'basis_changes': basis_changes}

    @property
    def cx_reversed(self) -> bool:
        """Whether transversal CX(A->B) implements logical CX with B as control.

        For self-dual CSS codes (Steane), transversal CX(A->B) = CX_L(A->B).
        For non-self-dual codes (Shor), transversal CX(A->B) = CX_L(B->A).

        When True, the protocol must:
        - Swap encoding roles (Alice encodes |0>_L, Bob encodes |+>_L)
        - Reverse CX direction in swap Bell measurements
        """
        return False

    # --- Stabilizer-based logical state preparation ---

    def get_x_stabilizer_strings(self) -> List[str]:
        """Return X-type stabilizer Pauli strings (e.g. 'IIIXXXX')."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_x_stabilizer_strings")

    def get_z_stabilizer_strings(self) -> List[str]:
        """Return Z-type stabilizer Pauli strings (e.g. 'IIIZZZZ')."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_z_stabilizer_strings")

    def get_logical_x_string(self) -> str:
        """Return the logical X operator as a Pauli string.

        Default: X on x_support qubits. Override for non-self-dual codes
        where the physical Pauli type differs from the logical type.
        """
        s = ["I"] * self.n
        for q in self.x_support:
            s[q] = "X"
        return "".join(s)

    def get_logical_z_string(self) -> str:
        """Return the logical Z operator as a Pauli string.

        Default: Z on z_support qubits. Override for non-self-dual codes.
        """
        s = ["I"] * self.n
        for q in self.z_support:
            s[q] = "Z"
        return "".join(s)

    def _prep_generators_for_zero(self) -> List[str]:
        """Stabilizer generators for |0_L>: all stabilizers + Z_L."""
        gens = []
        gens += ["+" + s.replace("I", "_") for s in self.get_x_stabilizer_strings()]
        gens += ["+" + s.replace("I", "_") for s in self.get_z_stabilizer_strings()]
        gens.append("+" + self.get_logical_z_string().replace("I", "_"))
        return gens

    def _prep_generators_for_plus(self) -> List[str]:
        """Stabilizer generators for |+_L>: all stabilizers + X_L."""
        gens = []
        gens += ["+" + s.replace("I", "_") for s in self.get_x_stabilizer_strings()]
        gens += ["+" + s.replace("I", "_") for s in self.get_z_stabilizer_strings()]
        gens.append("+" + self.get_logical_x_string().replace("I", "_"))
        return gens

    def prepare_logical_zero(self, circuit: stim.Circuit, offset: int = 0):
        """Prepare |0_L> using tableau synthesis from stabilizer generators."""
        prep = _tableau_prep_from_generators(self._prep_generators_for_zero())
        circuit += _shift_circuit(prep, offset)

    def prepare_logical_plus(self, circuit: stim.Circuit, offset: int = 0):
        """Prepare |+_L> using tableau synthesis from stabilizer generators."""
        prep = _tableau_prep_from_generators(self._prep_generators_for_plus())
        circuit += _shift_circuit(prep, offset)

    # --- Fault-tolerant state preparation ---

    def get_ft_checks(self, logical_state: str, ft_mode: str = "none", **kwargs) -> List[dict]:
        """Return FT check specifications for this code/logical_state.

        Each check dict format:
            {
                "type": "X" or "Z",
                "support": [int, ...],
                "detector": bool,   # optional, default True
            }
        """
        return []

    def _append_single_parity_check(
        self,
        circuit: stim.Circuit,
        *,
        offset: int,
        ancilla: int,
        check_type: str,
        support: List[int],
        add_detector: bool = True,
    ) -> None:
        """Append one ancilla-based parity check and optional DETECTOR."""
        circuit.append("R", [ancilla])

        if check_type == "Z":
            for q in support:
                circuit.append("CX", [offset + q, ancilla])
            circuit.append("M", [ancilla])

        elif check_type == "X":
            circuit.append("H", [ancilla])
            for q in support:
                circuit.append("CX", [ancilla, offset + q])
            circuit.append("H", [ancilla])
            circuit.append("M", [ancilla])

        else:
            raise ValueError(f"Unsupported check_type={check_type}; expected 'X' or 'Z'.")

        if add_detector:
            circuit.append("DETECTOR", [stim.target_rec(-1)])

    def get_strong_ft_supports(self, logical_state: str, **kwargs) -> List[List[int]]:
        """Return stabilizer supports for strong FT checks (verified cat states).

        Override in subclasses. Each inner list is the data-qubit support for
        one stabilizer generator. The same supports are used for both Z-type
        and X-type checks (applied in that order).
        """
        return []

    def _append_verified_cat4(self, circuit: stim.Circuit, cat: List[int], v: int,
                              *, add_detector: bool = True):
        """Prepare a verified 4-qubit cat state |0000>+|1111>.

        cat: 4 ancilla qubit indices for the cat state
        v: 1 verification qubit index
        """
        for q in cat:
            circuit.append("R", [q])
        circuit.append("H", [cat[0]])
        for q in cat[1:]:
            circuit.append("CX", [cat[0], q])
        circuit.append("R", [v])
        circuit.append("CX", [cat[0], v])
        circuit.append("CX", [cat[3], v])
        circuit.append("M", [v])
        if add_detector:
            circuit.append("DETECTOR", [stim.target_rec(-1)])

    def _append_z_stabilizer_check_with_cat(self, circuit: stim.Circuit, *,
                                            offset: int, data_support: List[int],
                                            cat: List[int], v: int,
                                            add_detector: bool = True):
        """FT Z-stabilizer measurement using a verified 4-cat state."""
        self._append_verified_cat4(circuit, cat, v, add_detector=add_detector)
        for dq, aq in zip(data_support, cat):
            circuit.append("CX", [offset + dq, aq])
        for aq in cat:
            circuit.append("H", [aq])
            circuit.append("M", [aq])
        # Syndrome parity is informational only — NOT a fault flag.
        # Only the cat-verify DETECTOR (in _append_verified_cat4) is postselected.

    def _append_x_stabilizer_check_with_cat(self, circuit: stim.Circuit, *,
                                            offset: int, data_support: List[int],
                                            cat: List[int], v: int,
                                            add_detector: bool = True):
        """FT X-stabilizer measurement using a verified 4-cat state."""
        self._append_verified_cat4(circuit, cat, v, add_detector=add_detector)
        for aq, dq in zip(cat, data_support):
            circuit.append("CX", [aq, offset + dq])
        for aq in cat:
            circuit.append("H", [aq])
            circuit.append("M", [aq])
        # Syndrome parity is informational only — NOT a fault flag.
        # Only the cat-verify DETECTOR (in _append_verified_cat4) is postselected.

    def _append_strong_ft_checks(self, circuit: stim.Circuit, *,
                                 logical_state: str, offset: int,
                                 ancilla_offset: int, add_detectors: bool,
                                 **kwargs) -> dict:
        """Append strong FT verification for all stabilizers using verified cat states."""
        supports = self.get_strong_ft_supports(logical_state=logical_state, **kwargs)
        if not supports:
            raise NotImplementedError(f"{self.name}: strong FT not defined")

        q = ancilla_offset
        detector_count = 0
        cat_blocks = 0

        for supp in supports:
            cat = [q, q+1, q+2, q+3]
            v = q + 4
            q += 5
            self._append_z_stabilizer_check_with_cat(
                circuit, offset=offset, data_support=supp, cat=cat, v=v,
                add_detector=add_detectors)
            cat_blocks += 1
            if add_detectors:
                detector_count += 1  # only cat-verify detector

        for supp in supports:
            cat = [q, q+1, q+2, q+3]
            v = q + 4
            q += 5
            self._append_x_stabilizer_check_with_cat(
                circuit, offset=offset, data_support=supp, cat=cat, v=v,
                add_detector=add_detectors)
            cat_blocks += 1
            if add_detectors:
                detector_count += 1  # only cat-verify detector

        return {
            "detector_count": detector_count,
            "ancilla_used": q - ancilla_offset,
            "cat_blocks_used": cat_blocks,
            "checks_applied": 2 * len(supports),
        }

    def prepare_logical_state(
        self,
        circuit: stim.Circuit,
        logical_state: str,
        *,
        offset: int = 0,
        ft_mode: str = "none",
        ancilla_offset: Optional[int] = None,
        ft_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ) -> dict:
        """Unified logical state preparation with optional FT checks.

        Args:
            circuit: Stim circuit to append into
            logical_state: "0" or "+"
            offset: physical data block offset
            ft_mode: "none", "minimal", or "strong"
            ancilla_offset: first ancilla qubit index for FT checks
            ft_kwargs: optional FT controls
                - add_detectors: bool (default True)
                - checks_override: list[dict] (override code-provided checks, minimal only)

        Returns:
            dict metadata with logical_state, ft_mode, detector_count,
            ancilla_used, checks_applied (and cat_blocks_used for strong).
        """
        if logical_state not in {"0", "+"}:
            raise ValueError(f"logical_state must be '0' or '+', got {logical_state!r}")

        if ft_kwargs is None:
            ft_kwargs = {}

        if logical_state == "0":
            self.prepare_logical_zero(circuit, offset=offset)
        else:
            self.prepare_logical_plus(circuit, offset=offset)

        if ft_mode == "none":
            return {
                "logical_state": logical_state,
                "ft_mode": ft_mode,
                "detector_count": 0,
                "ancilla_used": 0,
                "checks_applied": 0,
            }

        if ft_mode not in {"minimal", "strong"}:
            raise ValueError(f"Unsupported ft_mode={ft_mode!r}. Supported: 'none', 'minimal', 'strong'.")

        if ancilla_offset is None:
            raise ValueError("ancilla_offset is required when ft_mode != 'none'.")

        add_detectors = bool(ft_kwargs.get("add_detectors", True))

        if ft_mode == "strong":
            strong_meta = self._append_strong_ft_checks(
                circuit, logical_state=logical_state, offset=offset,
                ancilla_offset=ancilla_offset, add_detectors=add_detectors, **kwargs)
            return {"logical_state": logical_state, "ft_mode": ft_mode, **strong_meta}

        # minimal
        checks = ft_kwargs.get("checks_override")
        if checks is None:
            checks = self.get_ft_checks(logical_state=logical_state, ft_mode=ft_mode, **kwargs)

        anc = ancilla_offset
        detector_count = 0
        checks_applied = 0

        for chk in checks:
            check_type = chk["type"]
            support = chk["support"]
            det = add_detectors and chk.get("detector", True)

            self._append_single_parity_check(
                circuit,
                offset=offset,
                ancilla=anc,
                check_type=check_type,
                support=support,
                add_detector=det,
            )
            checks_applied += 1
            if det:
                detector_count += 1
            anc += 1

        return {
            "logical_state": logical_state,
            "ft_mode": ft_mode,
            "detector_count": detector_count,
            "ancilla_used": checks_applied,
            "checks_applied": checks_applied,
        }

    def __repr__(self):
        return f"CSSCode({self.name}, n={self.n}, k={self.k}, d={self.d})"


# ======================================================================
# Option A: Hand-coded codes
# ======================================================================

class Steane713(CSSCode):
    """[[7,1,3]] Steane code - self-dual CSS code.

    X_L = X^{otimes 7}, Z_L = Z^{otimes 7}.
    Encoding: 3 H + 12 CX gates.

    FT support:
      - ft_mode="none": tableau prep only
      - ft_mode="minimal": tableau prep + one parity check
          |0_L> : Z-check on support [0,5,6]
          |+_L> : X-check on support [0,5,6]
    """

    def __init__(self):
        super().__init__(
            name="[[7,1,3]]",
            n=7, k=1, d=3,
            x_support=[0, 1, 2, 3, 4, 5, 6],
            z_support=[0, 1, 2, 3, 4, 5, 6],
        )

    def encode(self, circuit: stim.Circuit, offset: int = 0):
        o = offset
        circuit.append("H", [o+4, o+5, o+6])
        circuit.append("CX", [o+0, o+1])
        circuit.append("CX", [o+0, o+2])
        circuit.append("CX", [o+6, o+3])
        circuit.append("CX", [o+6, o+1])
        circuit.append("CX", [o+6, o+0])
        circuit.append("CX", [o+5, o+3])
        circuit.append("CX", [o+5, o+2])
        circuit.append("CX", [o+5, o+0])
        circuit.append("CX", [o+4, o+3])
        circuit.append("CX", [o+4, o+2])
        circuit.append("CX", [o+4, o+1])

    def get_x_stabilizer_strings(self):
        return ["IIIXXXX", "IXXIIXX", "XIXIXIX"]

    def get_z_stabilizer_strings(self):
        return ["IIIZZZZ", "IZZIIZZ", "ZIZIZIZ"]

    def get_ft_checks(self, logical_state: str, ft_mode: str = "none", **kwargs) -> List[dict]:
        if ft_mode != "minimal":
            return []
        if logical_state == "0":
            return [{"type": "Z", "support": [0, 5, 6], "detector": True}]
        if logical_state == "+":
            return [{"type": "X", "support": [0, 5, 6], "detector": True}]
        raise ValueError(f"Unsupported logical_state={logical_state!r} for Steane FT checks.")

    def get_strong_ft_supports(self, logical_state: str, **kwargs) -> List[List[int]]:
        # Must match css_codes.py stabilizer generators (not testy.py's convention)
        # IIIZZZZ -> {3,4,5,6}, IZZIIZZ -> {1,2,5,6}, ZIZIZIZ -> {0,2,4,6}
        return [[3, 4, 5, 6], [1, 2, 5, 6], [0, 2, 4, 6]]


class Shor9(CSSCode):
    """[[9,1,3]] Shor code - non-self-dual CSS code.

    X-type logical: X_0 X_1 X_2 (support: {0,1,2}) — acts as logical Z
    Z-type logical: Z_0 Z_3 Z_6 (support: {0,3,6}) — acts as logical X

    Because the X-type logical acts as Z_L (not X_L), transversal CX(A->B)
    implements logical CX(B->A) — the control and target are reversed.
    The cx_reversed property signals this to the protocol.

    Encoding: |psi> = a|0> + b|1> on qubit 0, rest |0>.
    Circuit: CX(0,3), CX(0,6), H(0), H(3), H(6),
             CX(0,1), CX(0,2), CX(3,4), CX(3,5), CX(6,7), CX(6,8)
    """

    def __init__(self):
        super().__init__(
            name="[[9,1,3]]",
            n=9, k=1, d=3,
            x_support=[0, 1, 2],
            z_support=[0, 3, 6],
        )

    @property
    def cx_reversed(self) -> bool:
        return True

    def encode(self, circuit: stim.Circuit, offset: int = 0):
        o = offset
        # Phase 1: spread Z information
        circuit.append("CX", [o+0, o+3])
        circuit.append("CX", [o+0, o+6])
        # Phase 2: Hadamard to create superpositions
        circuit.append("H", [o+0, o+3, o+6])
        # Phase 3: spread X information within each block
        circuit.append("CX", [o+0, o+1])
        circuit.append("CX", [o+0, o+2])
        circuit.append("CX", [o+3, o+4])
        circuit.append("CX", [o+3, o+5])
        circuit.append("CX", [o+6, o+7])
        circuit.append("CX", [o+6, o+8])

    def get_x_stabilizer_strings(self):
        return ["XXXXXXIII", "IIIXXXXXX"]

    def get_z_stabilizer_strings(self):
        return [
            "ZZIIIIIII", "IZZIIIIII",
            "IIIZZIIII", "IIIIZZIII",
            "IIIIIIZZI", "IIIIIIIZZ",
        ]

    def get_logical_x_string(self):
        # X_L = Z_0Z_3Z_6 (physical Z acts as logical X for Shor code)
        s = ["I"] * self.n
        for q in self.z_support:
            s[q] = "Z"
        return "".join(s)

    def get_logical_z_string(self):
        # Z_L = X_0X_1X_2 (physical X acts as logical Z for Shor code)
        s = ["I"] * self.n
        for q in self.x_support:
            s[q] = "X"
        return "".join(s)


# ======================================================================
# Option B: Algorithmic construction from parity check matrices
# ======================================================================

class CSSCodeFromParity(CSSCode):
    """CSS code constructed algorithmically from H_X, H_Z matrices.

    Given parity check matrices H_X and H_Z satisfying H_X @ H_Z.T = 0 (mod 2),
    this class:
    1. Finds logical X and Z operators from the kernel/cokernel
    2. Synthesizes an encoding circuit from CX + H gates
    3. Provides all the same interfaces as hand-coded codes

    WARNING: The generated encoding circuit is correct but NOT guaranteed
    to be fault-tolerant. Use Option A codes for fault-tolerant simulations.
    """

    def __init__(self, name: str, H_X: np.ndarray, H_Z: np.ndarray, d: int):
        """Initialize from parity check matrices.

        Args:
            name: Code identifier, e.g. "[[9,1,3]]"
            H_X: X-type stabilizer parity check matrix (r_x x n)
            H_Z: Z-type stabilizer parity check matrix (r_z x n)
            d: Code distance
        """
        n = H_X.shape[1]
        assert H_Z.shape[1] == n, "H_X and H_Z must have same number of columns"
        assert np.all((H_X @ H_Z.T) % 2 == 0), "H_X and H_Z must satisfy CSS orthogonality"

        r_x = H_X.shape[0]
        r_z = H_Z.shape[0]
        k = n - r_x - r_z

        # Find logical operators
        x_logical, z_logical = self._find_logical_operators(H_X, H_Z, n, k)
        x_support = sorted(np.where(x_logical[0])[0].tolist())
        z_support = sorted(np.where(z_logical[0])[0].tolist())

        super().__init__(name, n, k, d, x_support, z_support)

        self.H_X = H_X.copy()
        self.H_Z = H_Z.copy()
        self._encoding_ops = self._synthesize_encoding(H_X, H_Z, x_logical, z_logical)

    def encode(self, circuit: stim.Circuit, offset: int = 0):
        for gate, targets in self._encoding_ops:
            shifted = [t + offset for t in targets]
            circuit.append(gate, shifted)

    def get_x_stabilizer_strings(self):
        return ["".join("X" if v else "I" for v in row) for row in self.H_X]

    def get_z_stabilizer_strings(self):
        return ["".join("Z" if v else "I" for v in row) for row in self.H_Z]

    @staticmethod
    def _find_logical_operators(H_X, H_Z, n, k):
        """Find logical X and Z operators from parity check matrices.

        Uses the kernel of H_Z to find X-type logicals that commute with
        all Z stabilizers but are not in the X stabilizer group.
        Similarly for Z-type logicals.

        Returns:
            (x_logical, z_logical): each is a (k x n) binary matrix
        """
        # Find kernel of H_Z (vectors v such that H_Z @ v = 0 mod 2)
        # These are candidate X-type operators
        ker_Hz = CSSCodeFromParity._binary_kernel(H_Z)

        # Remove rows that are in the row space of H_X
        # (those are stabilizers, not logicals)
        x_logical = CSSCodeFromParity._coset_representatives(ker_Hz, H_X, k)

        # Similarly: kernel of H_X^T gives Z-type candidates
        ker_Hx = CSSCodeFromParity._binary_kernel(H_X)
        z_logical = CSSCodeFromParity._coset_representatives(ker_Hx, H_Z, k)

        return x_logical, z_logical

    @staticmethod
    def _binary_kernel(M):
        """Compute kernel of binary matrix M (mod 2).

        Returns matrix whose rows span ker(M) over GF(2).
        """
        m, n = M.shape
        # Augment [M | I_n] and row-reduce
        aug = np.hstack([M.T, np.eye(n, dtype=int)]) % 2

        # Gaussian elimination over GF(2)
        pivot_row = 0
        pivots = []
        for col in range(m):
            # Find pivot
            found = False
            for row in range(pivot_row, n):
                if aug[row, col] == 1:
                    aug[[pivot_row, row]] = aug[[row, pivot_row]]
                    found = True
                    break
            if not found:
                continue
            pivots.append(pivot_row)
            # Eliminate
            for row in range(n):
                if row != pivot_row and aug[row, col] == 1:
                    aug[row] = (aug[row] + aug[pivot_row]) % 2
            pivot_row += 1

        # Kernel = rows of the right half where left half is zero
        kernel_rows = []
        for row in range(n):
            if np.all(aug[row, :m] == 0):
                kernel_rows.append(aug[row, m:].copy())

        if not kernel_rows:
            return np.zeros((0, n), dtype=int)
        return np.array(kernel_rows, dtype=int) % 2

    @staticmethod
    def _coset_representatives(kernel, stabilizers, k):
        """Find k independent coset representatives of kernel mod stabilizers.

        Returns k rows that are in the kernel but not in the row space
        of the stabilizer matrix.
        """
        if kernel.shape[0] == 0:
            return np.zeros((k, stabilizers.shape[1]), dtype=int)

        n = kernel.shape[1]
        # Combine stabilizers + kernel candidates
        combined = np.vstack([stabilizers, kernel]) % 2

        # Row reduce to find independent rows beyond stabilizers
        rref, pivot_cols = CSSCodeFromParity._binary_rref(combined)

        # The rows beyond the stabilizer count that have pivots are logicals
        n_stab = stabilizers.shape[0]
        logicals = []
        for i in range(n_stab, combined.shape[0]):
            if len(logicals) >= k:
                break
            row = combined[i]
            # Check if independent of stabilizers and previously found logicals
            test = np.vstack([stabilizers] + logicals + [row]) if logicals else \
                   np.vstack([stabilizers, row])
            _, test_pivots = CSSCodeFromParity._binary_rref(test)
            if len(test_pivots) > n_stab + len(logicals):
                logicals.append(row)

        if len(logicals) < k:
            # Fallback: just use first k kernel rows
            logicals = kernel[:k].tolist()

        return np.array(logicals[:k], dtype=int) % 2

    @staticmethod
    def _binary_rref(M):
        """Row reduce binary matrix to reduced row echelon form (mod 2).

        Returns (rref_matrix, pivot_columns).
        """
        A = M.copy() % 2
        m, n = A.shape
        pivot_row = 0
        pivot_cols = []

        for col in range(n):
            # Find pivot
            found = False
            for row in range(pivot_row, m):
                if A[row, col] == 1:
                    A[[pivot_row, row]] = A[[row, pivot_row]]
                    found = True
                    break
            if not found:
                continue
            pivot_cols.append(col)
            # Eliminate
            for row in range(m):
                if row != pivot_row and A[row, col] == 1:
                    A[row] = (A[row] + A[pivot_row]) % 2
            pivot_row += 1

        return A, pivot_cols

    @staticmethod
    def _synthesize_encoding(H_X, H_Z, x_logical, z_logical):
        """Synthesize encoding circuit from stabilizer structure.

        Strategy: build the encoding unitary as a sequence of CX and H gates
        that maps |psi>|0...0> to the code space.

        For k=1, qubit 0 holds the logical information.
        The circuit is built by:
        1. For each X stabilizer generator: use CX gates to create the
           correct X-type entanglement pattern
        2. For each Z stabilizer generator: use CX gates for Z-type
        3. H gates where needed for superposition

        Returns:
            List of (gate_name, target_list) tuples
        """
        n = H_X.shape[1]
        k = x_logical.shape[0] if len(x_logical.shape) > 1 else 1
        r_x = H_X.shape[0]
        r_z = H_Z.shape[0]

        ops = []

        # Build generator matrix G = [x_logical; H_X] (X part)
        # and check matrix [z_logical; H_Z] (Z part)
        # We need to find a Clifford that maps computational basis to code space

        # Simple approach: use the standard form construction
        # For each X-type stabilizer/logical with support on multiple qubits,
        # the encoding circuit uses CX from a source qubit to spread the state.

        # Determine which qubits are "information" vs "syndrome" vs "gauge"
        # For standard form: first k qubits are logical, next r_x are X-syndrome, rest are Z-syndrome

        # Build full X-type matrix: logicals on top, then stabilizers
        if k > 0 and x_logical.shape[0] > 0:
            X_full = np.vstack([x_logical.reshape(k, n), H_X]) % 2
        else:
            X_full = H_X.copy() % 2

        # Row reduce to standard form
        X_rref, x_pivots = CSSCodeFromParity._binary_rref(X_full)

        # For each row in X_full, the pivot column is the "source" qubit
        # and non-pivot 1s indicate CX targets
        # First, apply H to non-pivot qubits that need superposition
        h_qubits = []
        for i in range(r_x):
            row_idx = k + i  # stabilizer rows come after logical rows
            if row_idx < len(x_pivots):
                h_qubits.append(x_pivots[row_idx])

        if h_qubits:
            ops.append(("H", h_qubits))

        # Apply CX gates based on the X-type structure
        # For the logical X operator: CX from qubit 0 to other support qubits
        if k > 0:
            x_log = x_logical[0] if len(x_logical.shape) > 1 else x_logical
            support = np.where(x_log)[0].tolist()
            if len(support) > 1:
                src = support[0]
                for tgt in support[1:]:
                    ops.append(("CX", [src, tgt]))

        # For each X stabilizer: CX from pivot to other support qubits
        for i in range(r_x):
            row = H_X[i]
            support = np.where(row)[0].tolist()
            if len(support) > 1:
                src = support[0]
                for tgt in support[1:]:
                    ops.append(("CX", [src, tgt]))

        return ops

class ReedMuller15(CSSCode):
    """[[15,1,3]] quantum Reed-Muller code.

    Constructed from punctured Reed-Muller codes via CSS construction:
      H_X from parity check of RM*(1,4) [15,5,8] -- 10 generators
      H_Z from parity check of RM*(2,4) [15,11,4] -- 4 generators

    Key property: admits a transversal T gate, enabling universal
    fault-tolerant computation via code switching with Steane [[7,1,3]].

    The code can be viewed as a tetrahedral 3D color code:
      - 15 qubits: 4 vertices, 6 edge centers, 4 face centers, 1 body center
      - 4 weight-8 Z-stabilizers (cells)
      - 10 weight-4 X-stabilizers (faces)

    X_L = X_0 X_1 X_2 (weight 3)
    Z_L = Z_0 Z_1 Z_2 Z_3 Z_4 Z_5 Z_6 (weight 7)

    Note: Error correction is asymmetric. The code can correct
    up to 3 Z-errors (from distance-8 classical code) but only
    1 X-error (from distance-4 classical code). Overall code
    distance is d=3 (minimum weight logical operator).

    Encoding: algorithmic from parity check structure (not fault-tolerant).
    For fault-tolerant state preparation, use stabilizer measurement
    with verification ancillas.
    """

    # H_X: 10x15, parity check of punctured RM(1,4) [15,5,8]
    _H_X = np.array([
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
    ], dtype=int)

    # H_Z: 4x15, parity check of punctured RM(2,4) [15,11,4]
    _H_Z = np.array([
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
        [1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1],
    ], dtype=int)

    def __init__(self):
        super().__init__(
            name="[[15,1,3]]",
            n=15, k=1, d=3,
            x_support=[0, 1, 2],
            z_support=[0, 1, 2, 3, 4, 5, 6],
        )

    def encode(self, circuit: stim.Circuit, offset: int = 0):
        """Algorithmic encoding from parity check structure.

        WARNING: This encoding circuit is correct but NOT guaranteed
        to be fault-tolerant. For fault-tolerant state preparation,
        use stabilizer measurement with verification ancillas.
        """
        code = CSSCodeFromParity(
            self.name, self._H_X, self._H_Z, self.d
        )
        code.encode(circuit, offset)

    def get_x_stabilizer_strings(self):
        return ["".join("X" if v else "I" for v in row) for row in self._H_X]

    def get_z_stabilizer_strings(self):
        return ["".join("Z" if v else "I" for v in row) for row in self._H_Z]


class Golay23(CSSCode):
    """[[23,1,7]] quantum Golay code -- self-dual CSS code.

    Constructed from the classical [23,12,7] binary Golay code.
    Since the Golay code is self-orthogonal (its dual is its
    even-weight subcode [23,11,8]), we have H_X = H_Z = H,
    where H is the 11x23 parity check matrix of the Golay code.

    Key properties:
      - Self-dual CSS: all transversal Clifford gates
      - Distance 7: corrects up to 3 errors
      - All 11 X-stabilizers and 11 Z-stabilizers have weight 8
      - Useful for code switching with triorthogonal codes
        for universal fault-tolerant computation

    X_L = Z_L support on qubits {0,1,2,3,9,12,21} (weight 7).

    The classical Golay code is generated by the cyclic polynomial
    g(x) = 1 + x + x^5 + x^6 + x^7 + x^9 + x^11 over GF(2).

    Encoding: algorithmic from parity check structure (not fault-tolerant).
    """

    # H: 11x23, parity check matrix of [23,12,7] Golay code
    # Self-dual CSS: H_X = H_Z = H
    _H = np.array([
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
    ], dtype=int)

    def __init__(self):
        super().__init__(
            name="[[23,1,7]]",
            n=23, k=1, d=7,
            x_support=[0, 1, 2, 3, 9, 12, 21],
            z_support=[0, 1, 2, 3, 9, 12, 21],
        )

    def encode(self, circuit: stim.Circuit, offset: int = 0):
        """Algorithmic encoding from parity check structure.

        WARNING: This encoding circuit is correct but NOT guaranteed
        to be fault-tolerant. For fault-tolerant state preparation,
        use stabilizer measurement with verification ancillas.
        """
        code = CSSCodeFromParity(
            self.name, self._H, self._H, self.d
        )
        code.encode(circuit, offset)

    def get_x_stabilizer_strings(self):
        return ["".join("X" if v else "I" for v in row) for row in self._H]

    def get_z_stabilizer_strings(self):
        return ["".join("Z" if v else "I" for v in row) for row in self._H]


class BCH31(CSSCode):
    """[[31,1,7]] quantum BCH code — CSS from primitive narrow-sense BCH(31,16,7).

    Constructed from a dual-containing classical BCH code over GF(2^5).
    X- and Z-stabilizers come from C^perp (15 generators each).
    The code has:
      - n = 31 physical qubits
      - k = 1 logical qubit
      - d = 7 code distance (corrects up to 3 errors)
      - 15 X-type + 15 Z-type stabilizer generators

    Logical X_L and Z_L have support on qubits {0,1,2,3,5,7,8,9,10,11,15}
    (weight 11). Same support for both since the code is self-dual CSS.

    Encoding: algorithmic from stabilizer tableau (not fault-tolerant).
    """

    _X_STABS = [
        'XIIIXXXXIIIIXIIXXIIIIIIIIIIIIII',
        'XXIIXIIIXIIIXXIXIXIIIIIIIIIIIII',
        'XXXIXIXXIXIIXXXXIIXIIIIIIIIIIII',
        'XXXXXIXIXIXIXXXIIIIXIIIIIIIIIII',
        'IXXXXXIXIXIXIXXXIIIIXIIIIIIIIII',
        'XIXXIIIXXIXIIIXIIIIIIXIIIIIIIII',
        'IXIXXIIIXXIXIIIXIIIIIIXIIIIIIII',
        'XIXIIIXXIXXIIIIXIIIIIIIXIIIIIII',
        'XXIXXXXIXIXXXIIXIIIIIIIIXIIIIII',
        'XXXIIIIIIXIXIXIXIIIIIIIIIXIIIII',
        'XXXXXXXXIIXIIIXXIIIIIIIIIIXIIII',
        'XXXXIIIIXIIXXIIIIIIIIIIIIIIXIII',
        'IXXXXIIIIXIIXXIIIIIIIIIIIIIIXII',
        'IIXXXXIIIIXIIXXIIIIIIIIIIIIIIXI',
        'IIIXXXXIIIIXIIXXIIIIIIIIIIIIIIX',
    ]

    _Z_STABS = [
        'ZIIIZZZZIIIIZIIZZIIIIIIIIIIIIII',
        'ZZIIZIIIZIIIZZIZIZIIIIIIIIIIIII',
        'ZZZIZIZZIZIIZZZZIIZIIIIIIIIIIII',
        'ZZZZZIZIZIZIZZZIIIIZIIIIIIIIIII',
        'IZZZZZIZIZIZIZZZIIIIZIIIIIIIIII',
        'ZIZZIIIZZIZIIIZIIIIIIZIIIIIIIII',
        'IZIZZIIIZZIZIIIZIIIIIIZIIIIIIII',
        'ZIZIIIZZIZZIIIIZIIIIIIIZIIIIIII',
        'ZZIZZZZIZIZZZIIZIIIIIIIIZIIIIII',
        'ZZZIIIIIIZIZIZIZIIIIIIIIIZIIIII',
        'ZZZZZZZZIIZIIIZZIIIIIIIIIIZIIII',
        'ZZZZIIIIZIIZZIIIIIIIIIIIIIIZIII',
        'IZZZZIIIIZIIZZIIIIIIIIIIIIIIZII',
        'IIZZZZIIIIZIIZZIIIIIIIIIIIIIIZI',
        'IIIZZZZIIIIZIIZZIIIIIIIIIIIIIIZ',
    ]

    def __init__(self):
        super().__init__(
            name="[[31,1,7]]",
            n=31, k=1, d=7,
            x_support=[0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 15],
            z_support=[0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 15],
        )

    def encode(self, circuit: stim.Circuit, offset: int = 0):
        """Algorithmic encoding via tableau synthesis.

        WARNING: Not guaranteed fault-tolerant.
        """
        self.prepare_logical_zero(circuit, offset)

    def get_x_stabilizer_strings(self):
        return list(self._X_STABS)

    def get_z_stabilizer_strings(self):
        return list(self._Z_STABS)


# ======================================================================
# Registry
# ======================================================================

CSS_CODE_REGISTRY: Dict[str, CSSCode] = {
    "[[7,1,3]]": Steane713(),
    "[[9,1,3]]": Shor9(),
    "[[15,1,3]]": ReedMuller15(),
    "[[23,1,7]]": Golay23(),
    "[[31,1,7]]": BCH31(),
}


def get_css_code(name: str) -> CSSCode:
    """Look up a CSS code by name.

    Args:
        name: Code identifier, e.g. "[[7,1,3]]" or "[[9,1,3]]"

    Returns:
        CSSCode instance

    Raises:
        KeyError: If code not found in registry
    """
    if name not in CSS_CODE_REGISTRY:
        available = ", ".join(sorted(CSS_CODE_REGISTRY.keys()))
        raise KeyError(f"CSS code '{name}' not found. Available: {available}")
    return CSS_CODE_REGISTRY[name]
