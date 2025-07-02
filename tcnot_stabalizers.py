from stabalizer_circuit import StabilizerCircuit, StabilizerCircuitError
import numpy as np

def pretty_print(arr):
    """
    Pretty-print any array-like structure with commas between values,
    using nested brackets to mark array/matrix boundaries, and
    replacing exact zeros with '0.0'.

    Args:
        arr (array-like): Input data (scalar, vector, matrix, or higher-D).
    """
    a = np.asarray(arr)
    
    def fmt(x):
        return "0.0" if x == 0 else str(x)
    
    def _recurse(x, indent, is_last):
        """
        Recursively print nested structure.

        Args:
            x (np.ndarray): Current sub-array or scalar.
            indent (int): Current indentation level (spaces).
            is_last (bool): Whether this is the last element in its container.
        """
        pad = " " * indent
        if x.ndim == 0:
            # Scalar
            line = pad + fmt(x.item())
            line += "" if is_last else ","
            print(line)
        elif x.ndim == 1:
            # 1D vector
            line = pad + "[ " + ", ".join(fmt(v) for v in x) + " ]"
            line += "" if is_last else ","
            print(line)
        else:
            # ND array (N > 1)
            print(pad + "[")
            for i in range(x.shape[0]):
                _recurse(x[i], indent + 2, i == x.shape[0] - 1)
            end = pad + "]"
            end += "" if is_last else ","
            print(end)

    # Kick off recursion, marking the top-level as last so no trailing comma
    _recurse(a, indent=0, is_last=True)

def state_vector_to_dm(psi: np.ndarray) -> np.ndarray:
    """
    Convert a state vector |ψ⟩ to its density matrix ρ = |ψ⟩⟨ψ|.

    Args:
        psi (np.ndarray): 1D complex state vector of length 2**n.

    Returns:
        np.ndarray: 2D complex density matrix of shape (2**n, 2**n).
    """
    # Ensure a flat complex array
    vec = np.asarray(psi, dtype=complex).reshape(-1)
    # Outer product |ψ⟩⟨ψ|
    dm = np.outer(vec, vec.conj())
    return dm

def dms_equal(
    dm1: np.ndarray,
    dm2: np.ndarray,
    tol: float = 1e-8
) -> bool:
    """
    Check if two density matrices are equal within a numerical tolerance.

    Args:
        dm1 (np.ndarray): First density matrix of shape (N, N).
        dm2 (np.ndarray): Second density matrix of shape (N, N).
        tol (float): Absolute tolerance for element-wise comparison.

    Returns:
        bool: True if dm1 and dm2 have the same shape and all entries
              satisfy |dm1 - dm2| ≤ tol; False otherwise.
    """
    # Quick shape check
    if dm1.shape != dm2.shape:
        return False

    # Hermiticity check (optional, ensures each is a valid DM)
    if not (np.allclose(dm1, dm1.conj().T, atol=tol) and
            np.allclose(dm2, dm2.conj().T, atol=tol)):
        return False

    # Trace check (optional, ensures trace = 1)
    if not (np.isclose(np.trace(dm1), 1.0, atol=tol) and
            np.isclose(np.trace(dm2), 1.0, atol=tol)):
        return False

    # Element-wise comparison
    return np.allclose(dm1, dm2, atol=tol)


# Circuit
sc = StabilizerCircuit(4)
sc.h(1)
sc.cx(1, 2)
sc.cx(0, 1)
sc.cx(2, 3)

sc.cx(1,3)
sc.cz(2,0)


## Print State Vector
try:
    psi = sc.state_vector()
    print(f"Statevector (length {len(psi)}):")
    pretty_print(psi)
except StabilizerCircuitError as e:
    print("Error obtaining state vector:", e)


## Print Density Matrix
try:
    qubits = list(range(sc.num_qubits))
    dm = sc.density_matrix(qubits)
    print("Density matrix of qubits", qubits, ":")
    pretty_print(dm)
except StabilizerCircuitError as e:
    print("Error computing density matrix:", e)

## Compare the two 
print("state_vector dm == method dm ?", dms_equal(state_vector_to_dm(psi), dm)) 
