import numpy as np
from qutip import Qobj, ptrace

def get_state_dm(state):
    psi_qobj = Qobj(state, dims=[[2, 2], [1, 1]])  # Define as a 2-qubit system   ### TODO: check this 
    rho = psi_qobj * psi_qobj.dag()
    s1_final_dm = ptrace(rho, 0).full()
    s2_final_dm = ptrace(rho, 1).full()
    return s1_final_dm, s2_final_dm


def partial_trace_manual(rho, measured_qubits, total_qubits=4):
    """
    Computes the reduced density matrix by tracing out the measured qubits manually using NumPy.
    
    Parameters:
    - rho: (2^n x 2^n) NumPy array representing the 4-qubit density matrix.
    - measured_qubits: List of qubits to trace out (e.g., [0, 2] to remove qubits 0 and 2).
    - total_qubits: Total number of qubits (default: 4).
    
    Returns:
    - (4x4) reduced density matrix as a NumPy array.
    """
    # Define dimensions
    dim = 2 ** total_qubits  # Full density matrix size
    keep_qubits = [q for q in range(total_qubits) if q not in measured_qubits]  # Qubits to keep
    reduced_dim = 2 ** len(keep_qubits)  # Reduced matrix dimension

    # Reshape the density matrix into tensor form
    rho_tensor = rho.reshape([2] * (2 * total_qubits))  # Expands into 2D tensor indices

    # Perform partial trace over measured qubits
    for qubit in sorted(measured_qubits, reverse=True):  # Trace over last indices first
        rho_tensor = np.trace(rho_tensor, axis1=qubit, axis2=qubit + total_qubits)

    # Reshape back to a square reduced density matrix
    reduced_rho = rho_tensor.reshape((reduced_dim, reduced_dim))

    return reduced_rho

def extract_pure_state(rho, tol=1e-6):
    """
    Extracts the pure state from a density matrix if possible.
    
    Args:
        rho (np.ndarray): Density matrix.
        tol (float): Tolerance for numerical precision (default: 1e-6).

    Returns:
        np.ndarray or None: The pure state vector if rho is pure, otherwise None.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(rho)

    # Identify the largest eigenvalue
    max_eigenval_index = np.argmax(eigenvalues)
    max_eigenval = eigenvalues[max_eigenval_index]

    # Check if the density matrix represents a pure state (one dominant eigenvalue close to 1)
    if np.isclose(max_eigenval, 1.0, atol=tol) and np.allclose(np.sum(eigenvalues), 1.0, atol=tol):
        return eigenvectors[:, max_eigenval_index]  # Extract pure state vector
    return None  # Mixed state, no unique pure state

def measurement_result(state):
    return int(np.allclose(state, [0.+0.j, 1.+0.j]))