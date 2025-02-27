from sequence.kernel.timeline import Timeline
from sequence.topology.node import Node, BSMNode
from sequence.components.memory import Memory
from sequence.components.optical_channel import QuantumChannel, ClassicalChannel
from sequence.entanglement_management.generation import EntanglementGenerationA
from sequence.message import Message
from sequence.components.circuit import Circuit  
import numpy as np
from qutip import Qobj, ptrace


class SimpleManager:
    def __init__(self, owner, memo_name):
        self.owner = owner
        self.memo_name = memo_name
        self.raw_counter = 0
        self.ent_counter = 0

    def update(self, protocol, memory, state):
        if state == 'RAW':
            self.raw_counter += 1
            memory.reset()
        else:
            self.ent_counter += 1

    def create_protocol(self, middle: str, other: str):
        self.owner.protocols = [EntanglementGenerationA(self.owner, '%s.eg' % self.owner.name, middle, other,
                                                      self.owner.components[self.memo_name])]


class TeleportingCNOTNode(Node):
    def __init__(self, name: str, tl: Timeline):
        super().__init__(name, tl)
        self.timeline = tl

        memo_name = '%s.memo' % name
        communication_qubit = Memory(memo_name, self.timeline, fidelity=0.9, frequency=2000, efficiency=1, coherence_time=-1, wavelength=500)
        communication_qubit.add_receiver(self)
        self.add_component(communication_qubit)

        storage_qubit = Memory(f"{name}_storage_qubit", self.timeline, fidelity=0.9, frequency=2000, efficiency=1, coherence_time=-1, wavelength=500)
        self.add_component(storage_qubit)

        self.communication_qubit = communication_qubit
        self.storage_qubit = storage_qubit

        self.communication_qubit.qstate_key = self.timeline.quantum_manager.new()
        self.storage_qubit.qstate_key = self.timeline.quantum_manager.new()

        self.resource_manager = SimpleManager(self, memo_name)
        self.qm = self.timeline.quantum_manager 

    def init(self):
        memory = self.get_components_by_type("Memory")[0]
        memory.reset()

    def receive_message(self, src: str, msg: "Message") -> None:
        self.protocols[0].received_message(src, msg)

    def get(self, photon, **kwargs):
        self.send_qubit(kwargs['dst'], photon)

    def get_qubits(self):
        return self.qm.get(self.communication_qubit.qstate_key).state, self.qm.get(self.storage_qubit.qstate_key).state



def pair_protocol(node1: Node, node2: Node):
    p1 = node1.protocols[0]
    p2 = node2.protocols[0]
    node1_memo_name = node1.get_components_by_type("Memory")[0].name
    node2_memo_name = node2.get_components_by_type("Memory")[0].name
    p1.set_others(p2.name, node2.name, [node2_memo_name])
    p2.set_others(p1.name, node1.name, [node1_memo_name])



tl = Timeline()

node1 = TeleportingCNOTNode('node1', tl)
node2 = TeleportingCNOTNode('node2', tl)
bsm_node = BSMNode('bsm_node', tl, ['node1', 'node2'])
node1.set_seed(0)
node2.set_seed(1)
bsm_node.set_seed(2)

bsm = bsm_node.get_components_by_type("SingleAtomBSM")[0]
bsm.update_detectors_params('efficiency', 1)

qc1 = QuantumChannel('qc1', tl, attenuation=0, distance=1000)
qc2 = QuantumChannel('qc2', tl, attenuation=0, distance=1000)
qc1.set_ends(node1, bsm_node.name)
qc2.set_ends(node2, bsm_node.name)

nodes = [node1, node2, bsm_node]

for i in range(3):
    for j in range(3):
        if i != j:
            cc = ClassicalChannel('cc_%s_%s' % (nodes[i].name, nodes[j].name), tl, 1000, 1e8)
            cc.set_ends(nodes[i], nodes[j].name)

tl.init()

while node1.resource_manager.ent_counter < 1:

    tl.time = tl.now() + 1e11
    node1.resource_manager.create_protocol('bsm_node', 'node2')
    node2.resource_manager.create_protocol('bsm_node', 'node1')
    pair_protocol(node1, node2)

    memory1 = node1.get_components_by_type("Memory")[0]
    memory1.reset()
    memory2 = node2.get_components_by_type("Memory")[0]
    memory2.reset()

    node1.protocols[0].start()
    node2.protocols[0].start()
    tl.run()
print("node1 entangled memories : available memories")
print(node1.resource_manager.ent_counter, ':', node1.resource_manager.raw_counter)

state1 = node1.get_qubits()
state2 = node2.get_qubits()
# print("\n############################################# Entangled Qubit States:")
# print(f"Node1: {state1}")
# print(f"Node2: {state2}")

# Retrieve entangled qubit states from timeline quantum manager

s1_key = node1.storage_qubit.qstate_key
c1_key = node1.communication_qubit.qstate_key
s2_key = node2.storage_qubit.qstate_key
c2_key = node2.communication_qubit.qstate_key

a, b, c, d = np.random.rand(4)
a, b = a / np.sqrt(a**2 + b**2), b / np.sqrt(a**2 + b**2)  
c, d = c / np.sqrt(c**2 + d**2), d / np.sqrt(c**2 + d**2)  
s1_state = [a.item(), b.item()]
s2_state = [c.item(), d.item()]

print(f"Original s1 state: {s1_state}")
print(f"Original s2 state: {s2_state}")

node1.qm.set([s1_key], s1_state)
node2.qm.set([s2_key], s2_state)

cir1 = Circuit(size=4)  
cir1.cx(0, 1)
cir1.cx(2, 3)  

cir1.h(1)
cir1.measure(1)  # Measurement in H + X measurement = measurement in Z basis

m1 = node1.qm.run_circuit(cir1, keys=[s1_key, c1_key, c2_key, s2_key], meas_samp=np.random.rand())

c1_measurement_result = m1[c1_key]

cir2 = Circuit(size=3)  
if c1_measurement_result == 1:
    cir2.x(2)  # Apply X correction to storage qubit #2

cir2.measure(1) # Measurement in X basis  

m2 = node1.qm.run_circuit(cir2, keys=[s1_key, c2_key, s2_key], meas_samp=np.random.rand())
c2_measurement_result = m2[c2_key]

final_cir = Circuit(size=2)
if c2_measurement_result == 1:
    final_cir.z(0)  # Apply Z correction to storage qubit #1 

post_correction_state = node1.qm.run_circuit(final_cir, keys=[s1_key, s2_key])

print(post_correction_state)
print("\n############################################### State after correction")
state1 = node1.get_qubits()[0]
state2 = node2.get_qubits()[0]
print(f"Node1: {state1}")
print(f"Node2: {state2}")

print("\n############################################### Density Matricies of s1 and s2")
final_state = np.array(node1.get_qubits()[1])
psi_qobj = Qobj(final_state, dims=[[2, 2], [1, 1]])  # Define as a 2-qubit system
rho = psi_qobj * psi_qobj.dag()
s1_final_dm = ptrace(rho, 0).full()
s2_final_dm = ptrace(rho, 1).full()

# # Print results
# print("Reduced Density Matrix of Storage Qubit on Node 1:")
# print(s1_final_dm)

# print("\nReduced Density Matrix of Storage Qubit on Node 2:")
# print(s2_final_dm)


def extract_state_from_density_matrix(rho):
    """Extracts the pure state from a density matrix if possible."""
    eigenvalues, eigenvectors = np.linalg.eigh(rho)

    # Find the largest eigenvalue (close to 1)
    max_eigenval_index = np.argmax(eigenvalues)
    max_eigenval = eigenvalues[max_eigenval_index]

    # If eigenvalue is close to 1, return the corresponding eigenvector (pure state)
    if np.isclose(max_eigenval, 1.0):
        return eigenvectors[:, max_eigenval_index]  # Pure state
    else:
        return None  # Mixed state, no unique state vector

state_vector1 = extract_state_from_density_matrix(s1_final_dm)
state_vector2 = extract_state_from_density_matrix(s2_final_dm)

if state_vector1 is not None:
    print("Derived state vector:", state_vector1)
else:
    print("Density matrix represents a mixed state.")

if state_vector2 is not None:
    print("Derived state vector:", state_vector2)
else:
    print("Density matrix represents a mixed state.")


