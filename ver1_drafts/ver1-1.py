from sequence.kernel.timeline import Timeline
from sequence.topology.node import Node
from sequence.components.memory import Memory
from sequence.components.optical_channel import QuantumChannel, ClassicalChannel
from sequence.kernel.quantum_manager import QuantumManagerKet
from sequence.components.circuit import Circuit  
from qutip import *
import numpy as np

# Initialize the simulation timeline
timeline = Timeline()

# Initialize Quantum Manager using Ket Vector Formalism
quantum_manager = QuantumManagerKet()
timeline.quantum_manager = quantum_manager  # Attach quantum manager to timeline

# Define a simple quantum node with two memories (Qubit1, Qubit2 for Node1 and Qubit3, Qubit4 for Node2)
class QuantumNode(Node):
    def __init__(self, name, timeline):
        super().__init__(name, timeline)
        # Initialize two quantum memories (qubits)
        memory1 = Memory(f"{name}_memory1", timeline, fidelity=0.9, frequency=2000, efficiency=1, coherence_time=-1, wavelength=500)
        memory2 = Memory(f"{name}_memory2", timeline, fidelity=0.9, frequency=2000, efficiency=1, coherence_time=-1, wavelength=500)

        self.add_component(memory1)
        self.add_component(memory2)

        self.memory1 = memory1  # Qubit1 (Node1) / Qubit3 (Node2)
        self.memory2 = memory2  # Qubit2 (Node1) / Qubit4 (Node2)

        # Register the memories in QuantumManager
        self.memory1.qstate_key = quantum_manager.new()
        self.memory2.qstate_key = quantum_manager.new()

# Create two quantum nodes
node1 = QuantumNode("Node1", timeline)
node2 = QuantumNode("Node2", timeline)

# Create a quantum channel from node1 to node2
quantum_channel = QuantumChannel("QuantumChannel", timeline, distance=10, attenuation=0.0002)
quantum_channel.set_ends(node1, node2.name)

# Create classical channels for bidirectional communication
classical_channel_1to2 = ClassicalChannel("ClassicalChannel1to2", timeline, distance=10, delay=1e9)
classical_channel_1to2.set_ends(node1, node2.name)

classical_channel_2to1 = ClassicalChannel("ClassicalChannel2to1", timeline, distance=10, delay=1e9)
classical_channel_2to1.set_ends(node2, node1.name)

# Step 1: Generate Bell State Between Qubit2 (Node1) and Qubit3 (Node2)
circuit_entanglement = Circuit(size=2)
circuit_entanglement.h(0)  # Hadamard on Qubit2
circuit_entanglement.cx(0, 1)  # CNOT(Q2 → Q3) to create Bell state |Φ+>

# Run the entanglement circuit
quantum_manager.run_circuit(circuit_entanglement, keys=[node1.memory2.qstate_key, node2.memory1.qstate_key])

# Step 2: Initialize Qubit1 (Node1) in an Arbitrary State
a, b = np.random.rand(2)
a, b = a / np.sqrt(a**2 + b**2), b / np.sqrt(a**2 + b**2)  # Normalize
qubit1_state = [a, b]

# Qubit4 is not entangled yet
quantum_manager.set([node1.memory1.qstate_key], qubit1_state)

# Initialize the timeline to prepare for simulation
timeline.init()
timeline.run()

# Retrieve qubit state keys
qstate_key_1 = node1.memory1.qstate_key  # Qubit1
qstate_key_2 = node1.memory2.qstate_key  # Qubit2
qstate_key_3 = node2.memory1.qstate_key  # Qubit3
qstate_key_4 = node2.memory2.qstate_key  # Qubit4 (Initially empty)

# Step 3: Perform the Quantum Teleportation Circuit
circuit_teleport = Circuit(size=3)  # Only operates on Qubit1, Qubit2, and Qubit3
circuit_teleport.cx(0, 1)  # CNOT(Q1 → Q2) to entangle the input qubit with the Bell pair
circuit_teleport.h(0)  # Hadamard on Qubit1
circuit_teleport.measure(0)  # Measure Qubit1
circuit_teleport.measure(1)  # Measure Qubit2

# Run the teleportation circuit (Measure Q1 and Q2)
measurement_results = quantum_manager.run_circuit(circuit_teleport, keys=[qstate_key_1, qstate_key_2, qstate_key_3], meas_samp=np.random.rand())

# Step 4: Extract Measurement Results
m1_result = measurement_results[qstate_key_1]  # Measurement result for Qubit1
m2_result = measurement_results[qstate_key_2]  # Measurement result for Qubit2

# Step 5: Apply Pauli Corrections to Qubit4
circuit_correction = Circuit(size=1)  # Only operates on Qubit4

if m2_result == 1:
    circuit_correction.x(0)  # Apply X correction to Qubit4
if m1_result == 1:
    circuit_correction.z(0)  # Apply Z correction to Qubit4

# Apply corrections to Qubit4
quantum_manager.run_circuit(circuit_correction, keys=[qstate_key_4])

# Retrieve final states after correction
full_state = quantum_manager.get(qstate_key_4).state
state_qobj = Qobj(full_state, dims=[[2], [1]])
density_matrix = state_qobj.proj()

final_state_4 = density_matrix.ptrace(0)  # Qubit4 state after correction

# ✅ Clean Output: Print only the matrix values
print("📜 **Final Output Matrices**")
print("Node2 (Q4) Final State (After Correction):\n", final_state_4.full())  # Extract only the matrix data
