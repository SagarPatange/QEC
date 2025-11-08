from networkx import Graph, single_source_dijkstra, exception
from sequence.topology.topology import Topology as Topo
from sequence.topology.router_net_topo import RouterNetTopo
from sequence.topology.node import BSMNode, QuantumRouter, QuantumRouter2ndGeneration
from sequence.constants import SPEED_OF_LIGHT, MICROSECOND

class RouterNetTopo2G(RouterNetTopo):
    """Router Network Topology with 2nd Generation Quantum Router support.
    
    Extends RouterNetTopo to handle QuantumRouter2ndGeneration nodes.
    The JSON config can specify "generation": 2 to create a 2nd gen router.
    
    JSON Example:
        {
            "nodes": [{
                "name": "router_0",
                "type": "QuantumRouter",
                "generation": 2,  # This triggers 2nd gen creation
                "memo_size": 7,
                "data_memo_size": 7,     # Optional, defaults to 7
                "ancilla_memo_size": 6,  # Optional, defaults to 6
                ...
            }]
        }
    """
    
    def _add_nodes(self, config: dict):
        """Add nodes to the network topology.
        
        Overrides parent to handle QuantumRouter2ndGeneration creation.
        Everything else remains the same.
        """
        # First pass: create all nodes, but check for 2nd gen routers BEFORE creation
        for node in config[Topo.ALL_NODE]:
            seed = node[Topo.SEED]
            node_type = node[Topo.TYPE]
            name = node[Topo.NAME]
            template_name = node.get(Topo.TEMPLATE, None)
            template = self.templates.get(template_name, {})

            if node_type == self.BSM_NODE:
                others = self.bsm_to_router_map[name]
                node_obj = BSMNode(name, self.tl, others, seed=seed, component_templates=template)
            elif node_type == self.QUANTUM_ROUTER:
                # Check for 2nd generation BEFORE creating router
                if node.get("generation") == 2:
                    # Create 2nd generation router directly
                    memo_size = node.get(self.MEMO_ARRAY_SIZE, 7)
                    data_memo_size = node.get("data_memo_size", 7)
                    ancilla_memo_size = node.get("ancilla_memo_size", 6)
                    gate_fid = node.get(Topo.GATE_FIDELITY, 1.0)
                    meas_fid = node.get(Topo.MEASUREMENT_FIDELITY, 1.0)
                    
                    node_obj = QuantumRouter2ndGeneration(
                        name=name,
                        timeline=self.tl,
                        memo_size=memo_size,
                        seed=seed,
                        component_templates=template,
                        gate_fid=gate_fid,
                        meas_fid=meas_fid,
                        data_memo_size=data_memo_size,
                        ancilla_memo_size=ancilla_memo_size
                    )
                else:
                    # Create standard QuantumRouter
                    memo_size = node.get(self.MEMO_ARRAY_SIZE, 50)
                    gate_fid = node.get(Topo.GATE_FIDELITY, 1.0)
                    meas_fid = node.get(Topo.MEASUREMENT_FIDELITY, 1.0)
                    
                    node_obj = QuantumRouter(
                        name=name,
                        tl=self.tl,
                        memo_size=memo_size,
                        seed=seed,
                        component_templates=template,
                        gate_fid=gate_fid,
                        meas_fid=meas_fid
                    )
            else:
                raise ValueError("Unknown type of node '{}'".format(node_type))

            node_obj.set_seed(seed)
            self.nodes[node_type].append(node_obj)
            
    def _generate_forwarding_table(self, config: dict):
        """For static routing.
           Also updating the classical communication delay

        Args:
            config (dict): the config file
        """

        all_paths = {}  # (src, dst) -> (length: float, hop: int, path: tuple)

        graph = Graph()
        for node in config[Topo.ALL_NODE]:
            if node[Topo.TYPE] == self.QUANTUM_ROUTER:
                graph.add_node(node[Topo.NAME])

        costs = {}

        for qc in self.qchannels:
            # update all_paths
            router, bsm = qc.sender.name, qc.receiver
            all_paths[(router, bsm)] = (qc.distance, 0, (router, bsm))
            all_paths[(bsm, router)] = (qc.distance, 0, (bsm, router))

            if bsm not in costs:
                costs[bsm] = [router, qc.distance]
            else:
                costs[bsm] = [router] + costs[bsm]
                costs[bsm][-1] += qc.distance


        graph.add_weighted_edges_from(costs.values())
        for src in self.nodes[self.QUANTUM_ROUTER]:
            for dst_name in graph.nodes:
                if src.name == dst_name:
                    continue
                try:
                    if dst_name > src.name:
                        length, path = single_source_dijkstra(graph, src.name, dst_name)
                    else:
                        length, path = single_source_dijkstra(graph, dst_name, src.name)
                        path = path[::-1]
                    # update all_paths
                    hop_count = len(path) - 2
                    all_paths[(src.name, dst_name)] = (length, hop_count, tuple(path))
                    
                    next_hop = path[1]
                    # routing protocol locates at the bottom of the stack
                    routing_protocol = src.network_manager.protocol_stack[0]  # guarantee that [0] is the routing protocol?
                    routing_protocol.add_forwarding_rule(dst_name, next_hop)
                except exception.NetworkXNoPath:
                    pass
        
        # update the classical delay and the distance
        def classical_delay(distance: float, hop_count: int) -> float:
            """Model the classical delay as a function of distance and hop count
            """
            return distance / SPEED_OF_LIGHT + hop_count * 20 * MICROSECOND 

        for cc in self.cchannels:
            src = cc.sender.name
            dst = cc.receiver
            length, hop_count, path = all_paths[(src, dst)]
            cc.delay = classical_delay(length, hop_count)
            cc.distance = length   # not important
            # print(f'{path}: {cc.delay/1e6}us')