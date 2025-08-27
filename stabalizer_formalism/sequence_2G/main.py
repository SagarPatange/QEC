from sequence.kernel.timeline import Timeline
from sequence.topology.node import Node, BSMNode
from sequence.components.memory import Memory
from sequence.components.optical_channel import QuantumChannel, ClassicalChannel
from sequence.entanglement_management.generation import EntanglementGenerationA
from sequence.message import Message


class SimpleManager:
    def __init__(self, owner, memo_name):
        self.owner = owner
        self.memo_name = memo_name
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


class Entaglement_2G_Purification_Node(Node):
    def __init__(self, name: str, tl: Timeline):
        super().__init__(name, tl)

        memo_name = '%s.memo' % name
        memory = Memory(memo_name, tl, 0.9, 2000, 1, -1, 500)
        memory.add_receiver(self)
        self.add_component(memory)

        self.resource_manager = SimpleManager(self, memo_name)

    def init(self):
        memory = self.get_components_by_type("Memory")[0]
        memory.reset()

    def receive_message(self, src: str, msg: "Message") -> None:
        self.protocols[0].received_message(src, msg)

    def get(self, qubit, **kwargs):
        self.send_qubit(kwargs['dst'], qubit)


def pair_protocol(node1: Node, node2: Node):
    p1 = node1.protocols[0]
    p2 = node2.protocols[0]
    node1_memo_name = node1.get_components_by_type("Memory")[0].name
    node2_memo_name = node2.get_components_by_type("Memory")[0].name
    p1.set_others(p2.name, node2.name, [node2_memo_name])
    p2.set_others(p1.name, node1.name, [node1_memo_name])



tl = Timeline()

node1 = Entaglement_2G_Purification_Node('node1', tl)
node2 = Entaglement_2G_Purification_Node('node2', tl)

internode_distance = 100 # in 1 km
bsm_repeater_per_10km = 1  # number of BSM repeaters per 10 km
num_of_bsm_nodes = int(internode_distance / bsm_repeater_per_10km)

bsm_nodes = []
for i in range (num_of_bsm_nodes):
    bsm_node = StabalizerBSMNode(f'stabalizer_bsm_node_{i}', tl, ['node1', 'node2'])
    # bsm_node.set_seed(i + 2)  # Ensure unique seeds for each BSM node
    node1.add_component(bsm_node)
    node2.add_component(bsm_node)
    bsm_nodes.append(bsm_node)


# bsm = bsm_node.get_components_by_type("SingleAtomBSM")[0]
# bsm.update_detectors_params('efficiency', 1)

# qc1 = QuantumChannelStabalizer('qc1', tl, attenuation=0, distance=1000)
# qc2 = QuantumChannel('qc2', tl, attenuation=0, distance=1000)
# qc1.set_ends(node1, bsm_node.name)
# qc2.set_ends(node2, bsm_node.name)


nodes = [node1, node2, *bsm_nodes]


classical_channels = []  # Store the created ClassicalChannel instances

num_nodes = len(nodes)  # Get the number of nodes

for i in range(num_nodes):
    for j in range(num_nodes):
        if i != j:
            cc = ClassicalChannel('cc_%s_%s' % (nodes[i].name, nodes[j].name), tl, 1000, 1e8)
            cc.set_ends(nodes[i], nodes[j].name)
            classical_channels.append(cc)  

tl.init()
for i in range(1000):
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

# print("node1 entangled memories : available memories")
# print(node1.resource_manager.ent_counter, ':', node1.resource_manager.raw_counter)
