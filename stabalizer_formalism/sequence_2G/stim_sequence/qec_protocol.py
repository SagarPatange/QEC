"""
=====================================================================
qec_713_protocol.py
---------------------------------------------------------------------
Implementation of [[7,1,3]] quantum error correction code with
entanglement purification protocol.
=====================================================================
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import random

from enhanced_stabilizer_circuit import EnhancedStabilizerCircuit


class QEC713Protocol:
    """
    Implements the [[7,1,3]] quantum error correction code operations.
    """
    
    @staticmethod
    def encode_713(circuit: EnhancedStabilizerCircuit, qubit_indices: List[int]) -> EnhancedStabilizerCircuit:
        """
        Encode a physical qubit into [[7,1,3]] logical qubit.
        
        Args:
            circuit: Base circuit
            qubit_indices: List of 7 qubit indices [q0, q1, ..., q6]
            
        Returns:
            Modified circuit with encoding applied
        """
        c = circuit.copy()
        q = qubit_indices
        
        # Apply Hadamards
        c.h(q[4])
        c.h(q[5]) 
        c.h(q[6])
        
        # First set of CNOTs
        c.cx(q[0], q[1])
        c.cx(q[0], q[2])
        
        # q6 controls
        c.cx(q[6], q[3])
        c.cx(q[6], q[1])
        c.cx(q[6], q[0])
        
        # q5 controls
        c.cx(q[5], q[3])
        c.cx(q[5], q[2])
        c.cx(q[5], q[0])
        
        # q4 controls
        c.cx(q[4], q[3])
        c.cx(q[4], q[2])
        c.cx(q[4], q[1])
        
        return c
    
    @staticmethod
    def decode_713(circuit: EnhancedStabilizerCircuit, qubit_indices: List[int]) -> EnhancedStabilizerCircuit:
        """
        Decode [[7,1,3]] logical qubit back to physical qubit.
        
        Args:
            circuit: Base circuit
            qubit_indices: List of 7 qubit indices [q0, q1, ..., q6]
            
        Returns:
            Modified circuit with decoding applied
        """
        c = circuit.copy()
        q = qubit_indices
        
        # Inverse operations in reverse order
        # q4 controls
        c.cx(q[4], q[1])
        c.cx(q[4], q[2])
        c.cx(q[4], q[3])
        
        # q5 controls
        c.cx(q[5], q[0])
        c.cx(q[5], q[2])
        c.cx(q[5], q[3])
        
        # q6 controls
        c.cx(q[6], q[0])
        c.cx(q[6], q[1])
        c.cx(q[6], q[3])
        
        # Final CNOTs
        c.cx(q[0], q[2])
        c.cx(q[0], q[1])
        
        # Hadamards
        c.h(q[4])
        c.h(q[5])
        c.h(q[6])
        
        return c
    
    @staticmethod
    def stabilizers_713(circuit: EnhancedStabilizerCircuit,
                       data_qubits: List[int],
                       ancilla_qubits: List[int]) -> EnhancedStabilizerCircuit:
        """
        Measure stabilizers for [[7,1,3]] code.
        
        Args:
            circuit: Base circuit
            data_qubits: List of 7 data qubit indices
            ancilla_qubits: List of 6 ancilla qubit indices
            
        Returns:
            Modified circuit with stabilizer measurements
        """
        c = circuit.copy()
        d = data_qubits
        a = ancilla_qubits
        
        # --- Bit-flip syndrome extraction ---
        # Stabilizer 1: X₁X₃X₅X₇
        c.cx(d[0], a[0])
        c.cx(d[2], a[0])
        c.cx(d[4], a[0])
        c.cx(d[6], a[0])
        
        # Stabilizer 2: X₂X₃X₆X₇
        c.cx(d[1], a[1])
        c.cx(d[2], a[1])
        c.cx(d[5], a[1])
        c.cx(d[6], a[1])
        
        # Stabilizer 3: X₄X₅X₆X₇
        c.cx(d[3], a[2])
        c.cx(d[4], a[2])
        c.cx(d[5], a[2])
        c.cx(d[6], a[2])
        
        # --- Phase-flip syndrome extraction ---
        # Apply H to phase ancillas
        c.h(a[3])
        c.h(a[4])
        c.h(a[5])
        
        # Stabilizer 4: Z₁Z₃Z₅Z₇
        c.cx(a[3], d[0])
        c.cx(a[3], d[2])
        c.cx(a[3], d[4])
        c.cx(a[3], d[6])
        
        # Stabilizer 5: Z₂Z₃Z₆Z₇
        c.cx(a[4], d[1])
        c.cx(a[4], d[2])
        c.cx(a[4], d[5])
        c.cx(a[4], d[6])
        
        # Stabilizer 6: Z₄Z₅Z₆Z₇
        c.cx(a[5], d[3])
        c.cx(a[5], d[4])
        c.cx(a[5], d[5])
        c.cx(a[5], d[6])
        
        # Apply H back
        c.h(a[3])
        c.h(a[4])
        c.h(a[5])
        
        # Measure all ancillas
        c.measure_batch(a[0:6], basis='Z')
        
        return c


class EntanglementPurificationProtocol:
    """
    Full entanglement purification protocol using [[7,1,3]] QEC.
    """
    
    def __init__(self):
        self.total_qubits = 80
        self.block_length = 7
        
        # Station 1 qubit assignments
        self.station1 = {
            'n1_memory': list(range(0, 7)),           # [0-6]
            'n1_communication': list(range(7, 14)),   # [7-13]
            'n2_communication': list(range(14, 21)),  # [14-20]
            'n2_memory': list(range(21, 28)),         # [21-27]
            'n1_ancilla': list(range(56, 62)),        # [56-61]
            'n2_ancilla': list(range(62, 68))         # [62-67]
        }
        
        # Station 2 qubit assignments
        self.station2 = {
            'n1_memory': list(range(28, 35)),         # [28-34]
            'n1_communication': list(range(35, 42)),  # [35-41]
            'n2_communication': list(range(42, 49)),  # [42-48]
            'n2_memory': list(range(49, 56)),         # [49-55]
            'n1_ancilla': list(range(68, 74)),        # [68-73]
            'n2_ancilla': list(range(74, 80))         # [74-79]
        }
        
        self.qec = QEC713Protocol()
    
    def tcnot(self, circuit: EnhancedStabilizerCircuit, station: Dict) -> EnhancedStabilizerCircuit:
        """
        Apply transversal CNOT between memory and communication qubits.
        
        Args:
            circuit: Base circuit
            station: Dictionary with qubit assignments
            
        Returns:
            Modified circuit
        """
        c = circuit.copy()
        
        n1_mem = station['n1_memory']
        n1_comm = station['n1_communication']
        n2_comm = station['n2_communication']
        n2_mem = station['n2_memory']
        
        for i in range(self.block_length):
            # Prepare
            c.h(n1_mem[i])
            c.h(n1_comm[i])
            
            # Entangle communication qubits
            c.cx(n1_comm[i], n2_comm[i])
            
            # Local operations
            c.cx(n1_mem[i], n1_comm[i])
            c.cx(n2_comm[i], n2_mem[i])
            
            # Measure and correct
            c.measure(n1_comm[i])
            c.cx_conditional(-1, n2_mem[i])
            
            c.h(n2_comm[i])
            c.measure(n2_comm[i])
            c.cz_conditional(-1, n1_mem[i])
        
        return c
    
    def entanglement_swapping(self, circuit: EnhancedStabilizerCircuit,
                            qubit_indices: List[int]) -> EnhancedStabilizerCircuit:
        """
        Perform entanglement swapping on 4 qubits.
        
        Args:
            circuit: Base circuit
            qubit_indices: [q0, q1, q2, q3] where we swap q1-q2
            
        Returns:
            Modified circuit with swapping applied
        """
        c = circuit.copy()
        q = qubit_indices
        
        # Bell measurement on middle qubits
        c.cx(q[1], q[2])
        c.h(q[1])
        c.measure(q[1])
        c.measure(q[2])
        
        # Pauli corrections on outer qubit
        c.cx_conditional(-1, q[3])  # Controlled by q[2] measurement
        c.cz_conditional(-2, q[3])  # Controlled by q[1] measurement
        
        return c
    
    def create_purification_circuit(self, 
                                  error_probability: float = 0.1,
                                  apply_errors: bool = True) -> EnhancedStabilizerCircuit:
        """
        Create the full entanglement purification circuit.
        
        Args:
            error_probability: Probability of X error on first qubit
            apply_errors: Whether to apply random errors
            
        Returns:
            Complete circuit
        """
        # Initialize circuit with all qubits
        circuit = EnhancedStabilizerCircuit(self.total_qubits)
        
        # Initialize all qubits (identity gate for proper initialization)
        circuit.i(list(range(self.total_qubits)))
        
        # Encode all memory blocks into [[7,1,3]]
        circuit = self.qec.encode_713(circuit, self.station1['n1_memory'])
        circuit = self.qec.encode_713(circuit, self.station1['n2_memory'])
        circuit = self.qec.encode_713(circuit, self.station2['n1_memory'])
        circuit = self.qec.encode_713(circuit, self.station2['n2_memory'])
        
        # Optionally inject error and perform error correction
        if apply_errors and random.random() < error_probability:
            circuit.x(0)  # Apply X error on first qubit
            
            # Measure stabilizers to detect error
            circuit = self.qec.stabilizers_713(
                circuit,
                self.station1['n1_memory'],
                self.station1['n1_ancilla']
            )
            
            # Apply correction (simplified - in real QEC would decode syndrome)
            circuit.x(0)  # Correct the error
        
        # Apply transversal CNOTs
        circuit = self.tcnot(circuit, self.station1)
        circuit = self.tcnot(circuit, self.station2)
        
        # Perform entanglement swapping for each logical qubit
        for i in range(self.block_length):
            swap_qubits = [
                self.station1['n1_memory'][i],
                self.station1['n2_memory'][i],
                self.station2['n1_memory'][i],
                self.station2['n2_memory'][i]
            ]
            circuit = self.entanglement_swapping(circuit, swap_qubits)
        
        # Decode the logical qubits back to physical
        circuit = self.qec.decode_713(circuit, self.station1['n1_memory'])
        circuit = self.qec.decode_713(circuit, self.station2['n2_memory'])
        
        return circuit
    
    def calculate_fidelity(self, circuit: EnhancedStabilizerCircuit,
                         target_qubits: Tuple[int, int] = (0, 49),
                         shots: int = 10000) -> float:
        """
        Calculate fidelity with Bell state |Φ+⟩.
        
        Args:
            circuit: The purification circuit
            target_qubits: Pair of qubits to measure fidelity on
            shots: Number of tomography shots
            
        Returns:
            Fidelity value between 0 and 1
        """
        # Perform tomography
        rho = circuit.tomography_dm(target_qubits, shots=shots)
        
        # Define |Φ+⟩ = (|00⟩ + |11⟩)/√2
        phi_plus = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        
        # Calculate fidelity F = ⟨Φ+|ρ|Φ+⟩
        fidelity = float(np.real(np.vdot(phi_plus, rho @ phi_plus)))
        
        return fidelity