from qutip import basis, tensor, ket2dm, identity, to_super, sigmax, sigmaz, Qobj
from qutip.qip.operations import cnot as qutip_cnot
import numpy as np


def depolarizing_noise(state, error_prob):
    """
    Applies depolarizing noise to a quantum state.

    Args:
        state (Qobj): Quantum state as a density matrix.
        error_prob (float): Depolarizing error probability.

    Returns:
        Qobj: Noisy quantum state.
    """
    dims = state.dims
    d = state.shape[0]
    identity_op = identity(d)
    identity_op.dims = dims
    return (1 - error_prob) * state + (error_prob / d) * identity_op


def noisy_cnot(state, cnot_error, control, target, total_qubits):
    """
    Applies a noisy CNOT gate to a multi-qubit system.

    Args:
        state (Qobj): Input quantum state.
        cnot_error (float): Error probability for the CNOT gate.
        control (int): Control qubit index.
        target (int): Target qubit index.
        total_qubits (int): Total number of qubits in the system.

    Returns:
        Qobj: Quantum state after noisy CNOT.
    """
    # Use QuTiP's built-in CNOT operator
    cnot = qutip_cnot(N=total_qubits, control=control, target=target)

    # Apply the CNOT gate to the state
    state_after_cnot = cnot * state * cnot.dag()

    # Add depolarizing noise
    return depolarizing_noise(state_after_cnot, cnot_error)


def teleported_cnot_super_operator(bell_error=0.05, cnot_error=0.05, measurement_error=0.05):
    """
    Simulates the imperfect teleported CNOT gate.

    Args:
        bell_error (float): Error probability for the Bell state.
        cnot_error (float): Error probability for the CNOT gates.
        measurement_error (float): Measurement error probability.

    Returns:
        Qobj: Super-operator for the teleported CNOT gate.
    """
    # Step 1: Prepare Bell state |Φ+⟩ = (|00⟩ + |11⟩) / √2
    bell_state = (tensor(basis(2, 0), basis(2, 0)) + tensor(basis(2, 1), basis(2, 1))).unit()
    bell_dm = ket2dm(bell_state)
    noisy_bell = depolarizing_noise(bell_dm, bell_error)

    # Step 2: Prepare input state
    input_state = tensor(ket2dm(basis(2, 0)), ket2dm(basis(2, 0)))  # |00⟩ input
    combined_state = tensor(input_state, noisy_bell)

    # Step 3: Apply noisy CNOT gates
    combined_state = noisy_cnot(combined_state, cnot_error, control=2, target=0, total_qubits=4)
    combined_state = noisy_cnot(combined_state, cnot_error, control=1, target=3, total_qubits=4)

    # Step 4: Simulate measurement outcomes
    def simulate_measurement(state, error_prob):
        outcomes = [0, 1]
        probabilities = [1 - error_prob, error_prob]
        return np.random.choice(outcomes, p=probabilities)

    c1_result = simulate_measurement(combined_state, measurement_error)
    c2_result = simulate_measurement(combined_state, measurement_error)

    # Step 5: Apply corrections
    if c1_result == 1:
        combined_state = tensor(identity(2), identity(2), identity(2), sigmaz()) * combined_state
    if c2_result == 1:
        combined_state = tensor(sigmax(), identity(2), identity(2), identity(2)) * combined_state

    return to_super(combined_state)


def generate_logical_bell_pair(super_op, qec_code=[7, 1, 3]):
    """
    Generates a logical encoded Bell pair using the teleported CNOT gate.

    Args:
        super_op (Qobj): Super-operator of the teleported CNOT gate.
        qec_code (list): Quantum error correction code, e.g., [7,1,3].

    Returns:
        tuple: Density matrix of encoded Bell pair, Logical fidelity.
    """
    n = qec_code[0]  # Extract the number of physical qubits
    logical_zero = tensor([basis(2, 0)] * n)  # Encoded |0⟩
    logical_one = tensor([basis(2, 1)] * n)  # Encoded |1⟩
    logical_plus = (logical_zero + logical_one).unit()

    # Create encoded Bell state
    encoded_bell = (tensor(logical_zero, logical_zero) + tensor(logical_plus, logical_plus)).unit()
    encoded_bell_dm = ket2dm(encoded_bell)

    # Apply teleported CNOT super-operator
    final_state = super_op * encoded_bell_dm * super_op.dag()

    # Calculate logical fidelity
    logical_fidelity = (logical_zero.dag() * final_state * logical_zero).tr()
    return final_state, logical_fidelity


# Example Usage
teleported_cnot = teleported_cnot_super_operator(bell_error=0.05, cnot_error=0.05, measurement_error=0.05)
encoded_bell, fidelity = generate_logical_bell_pair(teleported_cnot, qec_code=[7, 1, 3])
print("Encoded Bell Pair Density Matrix:")
print(encoded_bell)
print(f"Logical Fidelity: {fidelity}")
