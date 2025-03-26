# This script sets up a quantum network simulation using SeQUeNCe components to
# perform entanglement generation and a remote CNOT operation between two nodes.
# It uses Bell State Measurements (BSM), quantum channels, and classical channels
# and includes quantum state tracking via a quantum manager.

from enum import Enum, auto
from sequence.kernel.timeline import Timeline
from sequence.topology.node import Node, BSMNode
from sequence.components.memory import Memory
from sequence.components.optical_channel import QuantumChannel, ClassicalChannel
from sequence.entanglement_management.generation import EntanglementGenerationA
from sequence.message import Message
from remote_cnot import RemoteCNOT
import logging 
import math
import numpy as np
from density_matrix import get_state_dm, extract_pure_state, measurement_result, partial_trace_2q


class SimpleManager:
    """
    Tracks quantum memory usage for a node by monitoring RAW and ENTANGLED states.
    It is used by nodes to interface with entanglement and remote CNOT protocols.
    """
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
        """Instantiates an entanglement generation protocol for the node."""
        self.owner.protocols = [EntanglementGenerationA(
            self.owner, f'{self.owner.name}.eg', middle, other,
            self.owner.components[self.memo_name])]

    def create_tcnot_protocol(self, other: str):
        """Instantiates a remote CNOT protocol for the node."""
        self.owner.protocols = [RemoteCNOT(self.owner, f'{self.owner.name}.tc', other)]


class RemoteCNOTNode(Node):
    """
    Represents a node with both storage and communication qubits.
    Used to initialize, store, and manipulate qubit states.
    """
    def __init__(self, name: str, tl: Timeline, qubit_state = None):
        super().__init__(name, tl)
        self.timeline = tl

        memo_name = f'{name}.memo'

        # Create storage and communication qubits
        storage_qubit = Memory(f"{name}.storage_qubit", tl, fidelity=0.9, frequency=2000, efficiency=1, coherence_time=-1, wavelength=500)
        communication_qubit = Memory(memo_name, tl, fidelity=0.9, frequency=2000, efficiency=1, coherence_time=-1, wavelength=500)
        communication_qubit.add_receiver(self)

        self.add_component(storage_qubit)
        self.add_component(communication_qubit)

        self.communication_qubit = communication_qubit
        self.storage_qubit = storage_qubit

        self.resource_manager = SimpleManager(self, memo_name)
        self.qm = tl.quantum_manager

        # Initialize storage qubit state (random or specified)
        if qubit_state is None:
            a, b = np.random.rand(2)
            norm = np.sqrt(a**2 + b**2)
            self.storage_qubit_state = [a / norm, b / norm]
        else:
            assert isinstance(qubit_state, list) and len(qubit_state) == 2 and math.isclose(math.sqrt(qubit_state[0]**2 + qubit_state[1]**2), 1.0, rel_tol=1e-9)
            self.storage_qubit_state = qubit_state

        self.original_storage_qubit_state = self.storage_qubit_state
        self.qm.set([self.storage_qubit.qstate_key], self.storage_qubit_state)

    def receive_message(self, src: str, msg: "Message") -> None:
        self.protocols[0].received_message(src, msg)

    def get(self, photon, **kwargs):
        self.send_qubit(kwargs['dst'], photon)

    def get_qubits(self):
        """Returns current state of communication and storage qubits."""
        return (
            self.qm.get(self.communication_qubit.qstate_key).state,
            self.qm.get(self.storage_qubit.qstate_key).state,
        )


def pair_protocol(node1: Node, node2: Node):
    """Establishes protocol connection between two nodes."""
    p1 = node1.protocols[0]
    p2 = node2.protocols[0]
    node1_memo_name = node1.get_components_by_type("Memory")[0].name
    node2_memo_name = node2.get_components_by_type("Memory")[0].name
    p1.set_others(p2.name, node2.name, [node2_memo_name])
    p2.set_others(p1.name, node1.name, [node1_memo_name])


# --- Simulation Setup ---
tl = Timeline()

# Initialize nodes with specific or random qubit states
node1 = RemoteCNOTNode('node1', tl, [1/np.sqrt(2), 1/np.sqrt(2)])
node2 = RemoteCNOTNode('node2', tl, [0, 1])
bsm_node = BSMNode('bsm_node', tl, ['node1', 'node2'])

# Set seeds for reproducibility
node1.set_seed(0)
node2.set_seed(1)
bsm_node.set_seed(2)

# Make detectors ideal (no loss)
bsm = bsm_node.get_components_by_type("SingleAtomBSM")[0]
bsm.update_detectors_params('efficiency', 1)

# Create quantum channels
qc1 = QuantumChannel('qc1', tl, attenuation=0, distance=1000)
qc2 = QuantumChannel('qc2', tl, attenuation=0, distance=1000)
qc1.set_ends(node1, bsm_node.name)
qc2.set_ends(node2, bsm_node.name)

# Create classical channels between all node pairs
nodes = [node1, node2, bsm_node]
for i in range(3):
    for j in range(3):
        if i != j:
            cc = ClassicalChannel(f'cc_{nodes[i].name}_{nodes[j].name}', tl, 1000, 1e8)
            cc.set_ends(nodes[i], nodes[j].name)

tl.init()

# Attempt entanglement generation until success
while node1.resource_manager.ent_counter < 1:
    tl.time = tl.now() + 1e11
    node1.resource_manager.create_entaglement_protocol('bsm_node', 'node2')
    node2.resource_manager.create_entaglement_protocol('bsm_node', 'node1')
    pair_protocol(node1, node2)
    node1.protocols[0].start()
    node2.protocols[0].start()
    tl.run()

print("node1 entangled memories : available memories")
print(node1.resource_manager.ent_counter, ':', node1.resource_manager.raw_counter)

# Retrieve post-entanglement states
state1 = node1.get_qubits()
state2 = node2.get_qubits()

# --- Remote CNOT Phase ---
logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)

tl.time = tl.now() + 1e11
node1.resource_manager.create_tcnot_protocol('node2')
node2.resource_manager.create_tcnot_protocol('node1')
pair_protocol(node1, node2)
node1.protocols[0].start()
node2.protocols[0].start()
tl.run()

print("\n#################### Final State vs Original State ####################")
communication1_qubit_measurement_result = measurement_result(node1.get_qubits()[0])
communication2_qubit_measurement_result = measurement_result(node2.get_qubits()[0])

print(f"Original Node1 Storage Qubit State :: {node1.original_storage_qubit_state}")
print(f"Original Node2 Storage Qubit State :: {node2.original_storage_qubit_state}")
print(f"Node1 Communication Qubit Measurement Result:: {communication1_qubit_measurement_result}")
print(f"Node2 Communication Qubit Measurement Result:: {communication2_qubit_measurement_result}")

final_state = np.array(node1.get_qubits()[1])
final_state_dm = np.outer(final_state, np.conjugate(final_state))
print(final_state_dm)

# Partial traces to extract reduced states
print(partial_trace_2q(final_state_dm, 0))
print(partial_trace_2q(final_state_dm, 1))

# Extract final density matrices
s1_final_dm, s2_final_dm = get_state_dm(final_state)
state_vector1 = extract_pure_state(s1_final_dm)
state_vector2 = extract_pure_state(s2_final_dm)

# Display results
if state_vector1 is not None:
    print("Final Node1 Storage Qubit State:", state_vector1)
else:
    print("Density matrix represents a mixed state:", s1_final_dm)

if state_vector2 is not None:
    print("Final Node2 Storage Qubit State:", state_vector2)
else:
    print("Density matrix represents a mixed state:", s2_final_dm)
