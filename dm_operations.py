from qutip import Qobj, ptrace
import numpy as np


class DM_Operations:  ## TODO: add docstring

    @staticmethod
    def reduce_qubits_0_and_3(density_matrix, keys): ## TODO: make more generalizable 
        """
        Reduce a multi-qubit density matrix to:
        1. rho_0: 2x2 density matrix of qubit 0 (logical)
        2. rho_3: 2x2 density matrix of qubit 3 (logical)
        3. rho_03: 4x4 joint density matrix of qubits 0 and 3 (logical)

        Args:
            density_matrix (np.ndarray or Qobj): Full 2^n x 2^n density matrix.
            keys (List[int]): Order of qubits in the matrix, maps logical qubits to physical indices.

        Returns:
            Tuple[Qobj, Qobj, Qobj]: (rho_0, rho_3, rho_03)
        """
        n_qubits = len(keys)
        rho = Qobj(density_matrix, dims=[[2] * n_qubits, [2] * n_qubits])

        # Find physical indices of logical qubits 0 and 3
        idx_0 = keys.index(0)
        idx_3 = keys.index(3)

        # Single-qubit reduced states
        rho_0 = ptrace(rho, [idx_0])
        rho_3 = ptrace(rho, [idx_3])

        # Joint 2-qubit reduced state (in sorted order)
        rho_03 = ptrace(rho, sorted([idx_0, idx_3]))

        return rho_0, rho_3, rho_03

    @staticmethod
    def partial_trace(rho: np.ndarray, dims: list[int], keep: list[int]) -> np.ndarray:
        """Partial trace over subsystems not in `keep`."""
        n = len(dims)
        rho_tensor = rho.reshape(dims + dims)
        keep = list(keep)
        other = [i for i in range(n) if i not in keep]
        perm = keep + other + [i + n for i in keep] + [i + n for i in other]
        rho_perm = np.transpose(rho_tensor, perm)
        dk = [dims[i] for i in keep]
        do = [dims[i] for i in other]
        rho_perm = rho_perm.reshape(np.prod(dk), np.prod(do), np.prod(dk), np.prod(do))
        return np.trace(rho_perm, axis1=1, axis2=3)

    @staticmethod
    def extract_two_qubit_dm(full_rho: np.ndarray, keys: list[int],
                            q1: int, q2: int) -> np.ndarray:
        """Extract 4x4 DM of logical qubits q1, q2 from 16x16 full DM using `keys`."""
        phys = sorted([keys.index(q1), keys.index(q2)])
        return DM_Operations.partial_trace(full_rho, dims=[2,2,2,2], keep=phys)
    
    @staticmethod
    def extract_single_qubit_dm(full_rho: np.ndarray, keys: list[int], q: int) -> np.ndarray:
        """
        Extracts the 2x2 density matrix of logical qubit `q` from the 16x16 full state.
        """
        phys = keys.index(q)
        return DM_Operations.partial_trace(full_rho, dims=[2, 2, 2, 2], keep=[phys])