from enum import Enum, auto
from sequence.kernel.timeline import Timeline
from sequence.topology.node import Node, BSMNode
from sequence.components.memory import Memory
from sequence.components.optical_channel import QuantumChannel, ClassicalChannel
from sequence.entanglement_management.generation import EntanglementGenerationA
from sequence.message import Message
# from sequence.utils.message import Message
from Error_Correction_Project.ver1_drafts.remote_cnot import RemoteCNOT
import logging 

import numpy as np
from qutip import Qobj, ptrace


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

    def create_entaglement_protocol(self, middle: str, other: str):
        self.owner.protocols = [EntanglementGenerationA(self.owner, '%s.eg' % self.owner.name, middle, other,
                                                      self.owner.components[self.memo_name])]
    def create_tcnot_protocol(self, other: str):
        self.owner.protocols = [RemoteCNOT(self.owner, '%s.tc' % self.owner.name, other)]


class RemoteCNOTNode(Node):
    def __init__(self, name: str, tl: Timeline):
        super().__init__(name, tl)
        self.timeline = tl

        memo_name = '%s.memo' % name
        communication_qubit = Memory(memo_name, self.timeline, fidelity=0.9, frequency=2000, efficiency=1, coherence_time=-1, wavelength=500)
        communication_qubit.add_receiver(self)
        self.add_component(communication_qubit)

        storage_qubit = Memory(f"{name}.storage_qubit", self.timeline, fidelity=0.9, frequency=2000, efficiency=1, coherence_time=-1, wavelength=500)
        self.add_component(storage_qubit)

        self.communication_qubit = communication_qubit
        self.storage_qubit = storage_qubit


        self.resource_manager = SimpleManager(self, memo_name)
        self.qm = self.timeline.quantum_manager 

        ## TODO: make a and b complex numbers 

        a, b = np.random.rand(2)
        a, b = a / np.sqrt(a**2 + b**2), b / np.sqrt(a**2 + b**2)  
        storage_qubit_state = [a.item(), b.item()]
        self.original_storage_qubit_state = storage_qubit_state

        self.qm.set([self.storage_qubit.qstate_key], storage_qubit_state)

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

node1 = RemoteCNOTNode('node1', tl)
node2 = RemoteCNOTNode('node2', tl)
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
    node1.resource_manager.create_entaglement_protocol('bsm_node', 'node2')
    node2.resource_manager.create_entaglement_protocol('bsm_node', 'node1')
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

###############################################################################################################
logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)

tl.time = tl.now() + 1e11
node1.resource_manager.create_tcnot_protocol('node2')
node2.resource_manager.create_tcnot_protocol('node1')
pair_protocol(node1, node2)

node1.protocols[0].start()
node2.protocols[0].start()
tl.run()

print("\n#################### Final State vs Original State ####################")
state1 = node1.get_qubits()[0]
state2 = node2.get_qubits()[0]
print(f"Original Node1 Storage Qubit State :: {node1.original_storage_qubit_state}")
print(f"Original Node2 Storage Qubit State :: {node2.original_storage_qubit_state}")

print(f"Node1 Communication Qubit Measurement Result:: {state1}")
print(f"Node2 Communication Qubit Measurement Result:: {state2}")

final_state = np.array(node1.get_qubits()[1])

psi_qobj = Qobj(final_state, dims=[[2, 2], [1, 1]])  # Define as a 2-qubit system   ### TODO: check this 
rho = psi_qobj * psi_qobj.dag()
s1_final_dm = ptrace(rho, 0).full()
s2_final_dm = ptrace(rho, 1).full()

state_vector1 = extract_pure_state(s1_final_dm)
state_vector2 = extract_pure_state(s2_final_dm)

if state_vector1 is not None:
    print("Final Node1 Storage Qubit State:", state_vector1)
else:
    print("Density matrix represents a mixed state.")

if state_vector2 is not None:
    print("Final Node2 Storage Qubit State:", state_vector2)
else:
    print("Density matrix represents a mixed state.")

###############################################################################################################

