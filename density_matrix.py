import numpy as np
from qutip import Qobj, ptrace

def get_state_dm(state):
    psi_qobj = Qobj(state, dims=[[2, 2], [1, 1]])  # Define as a 2-qubit system   ### TODO: check this 
    rho = psi_qobj * psi_qobj.dag()
    s1_final_dm = ptrace(rho, 0).full()
    s2_final_dm = ptrace(rho, 1).full()
    return s1_final_dm, s2_final_dm

def partial_trace_2q(rho, trace_out):
    """
    Computes the partial trace of a 2-qubit density matrix.
    
    Parameters:
    - rho: 4x4 NumPy array (density matrix of 2 qubits)
    - trace_out: The qubit to trace out (0 for first, 1 for second).
    
    Returns:
    - Reduced 2x2 density matrix after tracing out the specified qubit.
    """
    return np.array([
        [rho[0, 0] + rho[2, 2], rho[0, 1] + rho[2, 3]],
        [rho[1, 0] + rho[3, 2], rho[1, 1] + rho[3, 3]]
    ]) if trace_out == 0 else np.array([
        [rho[0, 0] + rho[1, 1], rho[0, 2] + rho[1, 3]],
        [rho[2, 0] + rho[3, 1], rho[2, 2] + rho[3, 3]]
    ])
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

import numpy as np

def quantum_xor(state1, state2):
    """
    Computes the element-wise XOR of two quantum states in vector representation.
    
    Parameters:
    - state1: NumPy array representing the first quantum state.
    - state2: NumPy array representing the second quantum state.

    Returns:
    - NumPy array representing the XORed quantum state.
    """
    # Ensure both states are NumPy arrays
    state1 = np.array(state1, dtype=int)
    state2 = np.array(state2, dtype=int)

    # Compute element-wise XOR (modulo 2 addition)
    xor_state = np.bitwise_xor(state1, state2)

    return xor_state

# Example usage:
state_a = np.array([1, 0])  # |0⟩ in vector form
state_b = np.array([0, 1])  # |1⟩ in vector form

xor_result = quantum_xor(state_a, state_b)
print(xor_result)  # Output: [1 1] (represents |1⟩ ⊕ |0⟩)
