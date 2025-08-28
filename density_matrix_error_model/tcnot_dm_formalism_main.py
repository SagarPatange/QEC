# --- IMPORTS ---
# Import core classes from SeQUeNCe framework and necessary modules
from sequence.kernel.timeline import Timeline
from sequence.topology.node import Node, BSMNode
from sequence.components.memory import Memory
from sequence.components.optical_channel import QuantumChannel, ClassicalChannel
from sequence.entanglement_management.generation import EntanglementGenerationA
from sequence.message import Message
from density_matrix_error_model.remote_cnot import RemoteCNOT
import logging
import math
import numpy as np
from sequence.kernel.quantum_manager import (DENSITY_MATRIX_FORMALISM)
from sequence.components.circuit import Circuit
from density_matrix_error_model.dm_operations import DM_Operations


class SimpleManager:
    """
    Manages simple memory bookkeeping: tracks number of raw and entangled memories.
    Handles memory reset after usage.
    Used by RemoteCNOTNode to manage entanglement lifecycle.
    """
    def __init__(self, owner, memo_name, depolarization_noise):
        self.owner = owner
        self.memo_name = memo_name
        self.raw_counter = 0  # Number of raw memories seen
        self.ent_counter = 0  # Number of successful entanglements
        self.depolarization_noise = depolarization_noise

    def update(self, protocol, memory, state):
        if state == 'RAW':
            self.raw_counter += 1
            memory.reset()
        else:
            self.ent_counter += 1

    def create_entaglement_protocol(self, middle: str, other: str):
        """Create EntanglementGenerationA protocol for this node."""
        self.owner.protocols = [EntanglementGenerationA(
            self.owner, f'{self.owner.name}.eg', middle, other,
            self.owner.components[self.memo_name])]

    def create_tcnot_protocol(self, other: str):
        """Create a RemoteCNOT protocol for remote CNOT operation."""
        self.owner.protocols = [RemoteCNOT(self.owner, f'{self.owner.name}.tc', other, depolarization_noise = self.depolarization_noise)]


class RemoteCNOTNode(Node):
    """
    Specialized Node that owns two memories: a communication qubit and a storage qubit.
    Used for entanglement distribution and remote CNOT gate operations.
    """
    def __init__(self, name: str, tl: Timeline, qubit_state = None, fidelity = 1, frequency = 2000, efficiency = 1, coherence_time = -1, wavelength = 500, depolarization_noise = True):
        super().__init__(name, tl)
        self.timeline = tl

        memo_name = f'{name}.memo'

        # Initialize storage and communication qubits
        if name == 'node1':
            storage_qubit = Memory(f"{name}.storage_qubit", tl, fidelity=fidelity, frequency=frequency, efficiency=efficiency, coherence_time=coherence_time, wavelength=wavelength)
            communication_qubit = Memory(memo_name, tl, fidelity=fidelity, frequency=frequency, efficiency=efficiency, coherence_time=coherence_time, wavelength=wavelength)
            communication_qubit.add_receiver(self)
        else: 
            communication_qubit = Memory(memo_name, tl, fidelity=fidelity, frequency=frequency, efficiency=efficiency, coherence_time=coherence_time, wavelength=wavelength)
            communication_qubit.add_receiver(self)
            storage_qubit = Memory(f"{name}.storage_qubit", tl, fidelity=fidelity, frequency=frequency, efficiency=efficiency, coherence_time=coherence_time, wavelength=wavelength)


        self.add_component(storage_qubit)
        self.add_component(communication_qubit)

        self.communication_qubit = communication_qubit
        self.storage_qubit = storage_qubit

        self.resource_manager = SimpleManager(self, memo_name, depolarization_noise = depolarization_noise)

        self.qm = tl.quantum_manager

        # Initialize storage qubit state (randomized if not specified)
        if qubit_state is None:
            a, b = np.random.rand(2)
            norm = np.sqrt(a**2 + b**2)
            self.storage_qubit_state = [a / norm, b / norm]
        else:
            assert (isinstance(qubit_state, list) and len(qubit_state) == 2 and math.isclose(
            math.sqrt(abs(qubit_state[0])**2 + abs(qubit_state[1])**2), 1.0, rel_tol=1e-9)   
)
            self.storage_qubit_state = qubit_state

        self.original_storage_qubit_state = self.storage_qubit_state

        # Set storage qubit state in quantum manager
        self.qm.set([self.storage_qubit.qstate_key], self.storage_qubit_state)

    def receive_message(self, src: str, msg: "Message") -> None:
        """Forward received classical message to protocol stack."""
        self.protocols[0].received_message(src, msg)

    def get(self, photon, **kwargs):
        """Handle receiving a photon through a quantum channel."""
        self.send_qubit(kwargs['dst'], photon)

    def get_qubits(self):
        """Return communication and storage qubit states."""
        return (
            self.qm.get(self.communication_qubit.qstate_key).state,
            self.qm.get(self.storage_qubit.qstate_key).state,
        )


def pair_protocol(node1: Node, node2: Node):
    """Pair two nodes' protocols together by setting remote references."""
    p1 = node1.protocols[0]
    p2 = node2.protocols[0]
    node1_memo_name = node1.get_components_by_type("Memory")[0].name
    node2_memo_name = node2.get_components_by_type("Memory")[0].name
    p1.set_others(p2.name, node2.name, [node2_memo_name])
    p2.set_others(p1.name, node1.name, [node1_memo_name])


# --- SIMULATION SETUP ---

def simulate(internode_distance, attenuation, entaglment_fidelity, photon_frequency, entanglement_efficiency, entaglement_coherence_time, photon_wavelength, initial_node_1_storage_qubit_state, initial_node_2_storage_qubit_state, depolarization_noise):

    tl = Timeline(formalism=DENSITY_MATRIX_FORMALISM)


    # Create two remote nodes with specified initial storage qubit states
    node1 = RemoteCNOTNode('node1', tl, initial_node_1_storage_qubit_state, entaglment_fidelity, photon_frequency, entanglement_efficiency, entaglement_coherence_time, photon_wavelength, depolarization_noise = depolarization_noise)
    node2 = RemoteCNOTNode('node2', tl, initial_node_2_storage_qubit_state, entaglment_fidelity, photon_frequency, entanglement_efficiency, entaglement_coherence_time, photon_wavelength, depolarization_noise = depolarization_noise)

    ##### State prep

    initialize_circuit = Circuit(size=2)
    
    keys = [0,3]
    tl.quantum_manager.run_circuit(initialize_circuit, keys=keys)

    #####
    # Create a middle Bell State Measurement (BSM) node
    bsm_node = BSMNode('bsm_node', tl, ['node1', 'node2'])

    # Set seeds for random generators for reproducibility
    node1.set_seed(0)
    node2.set_seed(1)
    bsm_node.set_seed(2)

    bsm = bsm_node.get_components_by_type("SingleAtomBSM")[0]
    bsm.update_detectors_params('efficiency', 1)

    qc1 = QuantumChannel('qc1', tl, attenuation=attenuation, distance=internode_distance)
    qc2 = QuantumChannel('qc2', tl, attenuation=attenuation, distance=internode_distance)
    qc1.set_ends(node1, bsm_node.name)
    qc2.set_ends(node2, bsm_node.name)

    nodes = [node1, node2, bsm_node]
    for i in range(3):
        for j in range(3):
            if i != j:
                cc = ClassicalChannel(f'cc_{nodes[i].name}_{nodes[j].name}', tl, internode_distance, 1e8)
                cc.set_ends(nodes[i], nodes[j].name)

    tl.init()

    # --- ENTANGLEMENT GENERATION PHASE ---

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


    # --- REMOTE CNOT PHASE ---

    logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)
    tl.time = tl.now() + 1e11
    node1.resource_manager.create_tcnot_protocol('node2')
    node2.resource_manager.create_tcnot_protocol('node1')

    pair_protocol(node1, node2)

    node1.protocols[0].start()
    node2.protocols[0].start()

    tl.run()

    # --- POST SIMULATION: STATE ANALYSIS ---

    print("\n#################### Final State vs Original State ####################\n")
    rho = tl.quantum_manager.get(0).state
    keys = tl.quantum_manager.states[0].keys
    rho_q0, rho_q3 , rho_03 = DM_Operations.reduce_qubits_0_and_3(rho, keys)
    # Display results
    print("Reduced density matrix for qubit 0:")
    print(rho_q0.full())

    print("\nReduced density matrix for qubit 3:")
    print(rho_q3.full())

    print("\nFull 2-Qubit Density Matrix")
    print(rho_03.full())

def main():
    internode_distance = 1000 # in m 
    attenuation = 0 # in db/m
    entaglment_fidelity=1 
    photon_frequency=2000 # Hz
    entanglement_efficiency=1 
    entaglement_coherence_time=-1 # -1 means infinite
    photon_wavelength=500 # in nm
    initial_node_1_storage_qubit_state = [1+0j, 0+0j]
    initial_node_2_storage_qubit_state = [1+0j, 0+0j]
    depolarization_noise = True


    simulate(internode_distance = internode_distance, 
             attenuation = attenuation,
             entaglment_fidelity=entaglment_fidelity, 
             photon_frequency=photon_frequency, 
             entanglement_efficiency = entanglement_efficiency,
             entaglement_coherence_time = entaglement_coherence_time,
             photon_wavelength = photon_wavelength,
             initial_node_1_storage_qubit_state = initial_node_1_storage_qubit_state, 
             initial_node_2_storage_qubit_state = initial_node_2_storage_qubit_state, 
             depolarization_noise = depolarization_noise)

if __name__ == "__main__":
    main()
