import numpy as np
from pathlib import Path
from sequence.topology.router_net_topo import RouterNetTopo
from router_net_topo_2G import RouterNetTopo2G
import sequence.utils.log as log
from sequence.kernel.quantum_manager import QuantumManager
from sequence.constants import STABILIZER_FORMALISM
from sequence.entanglement_management.generation import EntanglementGenerationA, EntanglementGenerationB

from request_app import RequestAppThroughput
from RequestLogicalPairApp import RequestLogicalPairApp
from sequence.topology.node import QuantumRouter2ndGeneration


def resolve_config_path(config_file: str) -> str:
    config_path = Path(config_file)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_file
    return str(config_path)


def run_five_node_pair(label: str, css_code: str, non_ft_config: str, ft_config: str):
    """Run paired 5-node comparisons, skipping configs that do not exist yet."""
    print("\n" + "=" * 70)
    print(f"  {label}")
    print("=" * 70)

    run_ft = False
    mode_configs = [("Non-FT", non_ft_config)]
    if run_ft:
        mode_configs.append(("FT", ft_config))

    for mode_label, config_file in mode_configs:
        config_path = Path(resolve_config_path(config_file))
        print(f"\n[{mode_label}] {config_file}")
        if not config_path.exists():
            print(f"Skipping missing config: {config_file}")
            continue
        five_node_logical_pair_with_app(
            config_file=config_file,
            css_code=css_code,
        )


def two_node_physical_pair_with_app_ketstate(verbose=False):
    """Test with stabilizer formalism using RequestAppThroughput."""
    print('\nPhysical Entangled Pairs using KetState with RequestApp:')
    

    
    log_filename = 'log/one_memory_ketstate_app'
    network_config = 'config/line_2_physical_ketState.json'
    
    network_topo = RouterNetTopo(network_config)
    tl = network_topo.get_timeline()
    
    log.set_logger(__name__, tl, log_filename)
    log.set_logger_level('DEBUG')
    modules = ['timeline', 'network_manager', 'resource_manager', 'rule_manager', 
               'generation', 'purification', 'swapping', 'bsm', 'barret_kok']
    for module in modules:
        log.track_module(module)
    
    # Set up applications
    apps = []
    src_node_name = 'router_0'
    dest_node_name = 'router_1'
    src_app = None
    src_node = None
    
    for router in network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER):
        app = RequestAppThroughput(router)
        apps.append(app)
        if router.name == src_node_name:
            src_app = app
            src_node = router
    
    # Configure request
    start_time = 1e12  # 1 second
    end_time = 10e12   # 10 seconds
    memory_size = 1   # Number of memories
    target_fidelity = 0.8
    
    # Set maximum entanglements
    src_app.max_entanglements = 1
    
    # Start the request through the app
    src_app.start(dest_node_name, start_time, end_time, memory_size, target_fidelity)
    
    tl.init()
    tl.run()
    
    # Display results from RequestApp
    print("\n--- Results from RequestApp ---")
    
    # Time to service
    time_to_service = src_app.get_time_to_service()
    if time_to_service:
        print(f"Time to service samples (first 10): {[f'{t:.3f}' for t in time_to_service[:10]]}")
        print(f"Average time to service: {np.mean(time_to_service):.3f} µs")
    
    # memory_manager = src_node.resource_manager.memory_manager
    # for info in memory_manager:
    #     if info.state == "ENTANGLED":
    #         src_app.get_memory(info)
    
    # Fidelity
    fidelities = []
    for res_fidelities in src_app.entanglement_fidelities.values():
        fidelities.extend(res_fidelities)

    if fidelities:
        print(f"Fidelity samples (first 10): {[f'{f:.5f}' for f in fidelities[:10]]}")
        print(f"Average fidelity: {np.mean(fidelities):.5f}")
    
    # Throughput
    throughput_dict = src_app.get_request_to_throughput()
    if 'summary' in throughput_dict:
        print(f"Overall throughput: {throughput_dict['summary']:.2f} entanglements/second")
        print(f"Total entanglements: {throughput_dict['total_entanglements']}")
        print(f"Duration: {throughput_dict['duration_seconds']:.2f} seconds")
    
    # Statistics
    stats = src_app.get_statistics()
    print(f"\n--- Statistics ---")
    print(f"Completed: {stats['completed']}")
    print(f"Timeout triggered: {stats['timeout_triggered']}")
    print(f"Total entanglements generated: {stats['total_entanglements']}")


def two_node_physical_pair_with_app_stabilizer():
    """Test with stabilizer formalism using RequestAppThroughput."""
    print('\nPhysical Entangled Pairs using Stabilizers with RequestApp:')
    
    QuantumManager.set_global_manager_formalism(STABILIZER_FORMALISM)
    EntanglementGenerationA.set_global_type('barret_kok_stabilizer')
    EntanglementGenerationB.set_global_type('barret_kok_stabilizer')
    
    log_filename = 'log/one_memory_stabilizer_app'
    network_config = 'config/line_2_physical_stabilizer.json'
    
    network_topo = RouterNetTopo(network_config)
    tl = network_topo.get_timeline()
    
    log.set_logger(__name__, tl, log_filename)
    log.set_logger_level('INFO')
    modules = ['timeline', 'network_manager', 'resource_manager', 'rule_manager', 
               'generation', 'purification', 'swapping', 'bsm', "barret_kok"]
    for module in modules:
        log.track_module(module)
    
    # Set up applications
    src_node_name = 'router_0'
    dest_node_name = 'router_1'
    src_app = None
    dest_app = None
    src_node = None

    for router in network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER):
        app = RequestAppThroughput(router)
        if router.name == src_node_name:
            src_app = app
            src_node = router
        elif router.name == dest_node_name:
            dest_app = app
    
    # Configure and start
    start_time = 1e12
    end_time = 10e12
    memory_size = 6
    target_fidelity = 0.8
    src_app.max_entanglements = 1
    
    src_app.start(dest_node_name, start_time, end_time, memory_size, target_fidelity)
    
    tl.init()
    tl.run()

    # For stabilizer formalism, entanglements are counted on the responder (dest_app)
    stats = dest_app.get_statistics()
    
    print("\n" + "="*70)
    print(f"Total entanglements: {stats['total_entanglements']}")
    
    # Print fidelities for each pair
    if stats.get('all_fidelities'):
        print(f"\nFidelities:")
        for i, fid in enumerate(stats['all_fidelities']):
            print(f"  Pair {i}: {fid:.6f}")
        
        # Print average
        if stats.get('overall_fidelity_stats'):
            print(f"  Average: {stats['overall_fidelity_stats']['avg']:.6f}")
    
    # Print throughput
    if stats.get('throughput') and 'summary' in stats['throughput']:
        print(f"\nThroughput: {stats['throughput']['summary']:.2f} entanglements/s")
    
    print("="*70 + "\n")
    

def five_node_line_topology_smoke_test(config_file='config/line_5.json'):
    """Smoke test the new 5-router line topology with one physical request.

    This verifies that the topology loads, router_0 can issue a request to
    router_4 through the multihop chain, and the simulation completes.
    """
    print('\n5-node linear topology smoke test:')

    log_filename = 'log/line_5_smoke_test'
    network_topo = RouterNetTopo(config_file)
    tl = network_topo.get_timeline()

    log.set_logger(__name__, tl, log_filename)
    log.set_logger_level('INFO')

    src_node_name = 'router_0'
    dest_node_name = 'router_4'
    src_app = None

    routers = network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER)
    for router in routers:
        app = RequestAppThroughput(router)
        if router.name == src_node_name:
            src_app = app

    if src_app is None:
        raise RuntimeError(f"Failed to find source app for {src_node_name}")

    start_time = 1e12
    end_time = 5e12
    memory_size = 1
    target_fidelity = 0.01
    src_app.max_entanglements = 1
    src_app.start(dest_node_name, start_time, end_time, memory_size, target_fidelity)

    tl.init()
    tl.run()

    stats = src_app.get_statistics()
    print(f"Routers: {len(routers)}")
    print(f"Completed: {stats['completed']}")
    print(f"Timeout triggered: {stats['timeout_triggered']}")
    print(f"Total entanglements generated: {stats['total_entanglements']}")
    if src_app.entanglement_fidelities:
        fidelities = []
        for values in src_app.entanglement_fidelities.values():
            fidelities.extend(values)
        if fidelities:
            print(f"Average fidelity: {np.mean(fidelities):.6f}")


def five_node_physical_pair_with_app_ketstate(config_file='config/line_5.json'):
    """True no-QEC 5-node baseline using raw physical entanglement."""
    print('\n5-node physical pairs using KetState with RequestApp:')
    QuantumManager.clear_active_formalism()

    log_filename = 'log/line_5_physical_ketstate_app'
    network_topo = RouterNetTopo(config_file)
    tl = network_topo.get_timeline()

    log.set_logger(__name__, tl, log_filename)
    log.set_logger_level('INFO')

    src_node_name = 'router_0'
    dest_node_name = 'router_4'
    src_app = None

    for router in network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER):
        app = RequestAppThroughput(router)
        if router.name == src_node_name:
            src_app = app

    if src_app is None:
        raise RuntimeError(f"Failed to find source app for {src_node_name}")

    start_time = 1e12
    end_time = 10e12
    memory_size = 1
    target_fidelity = 0.01
    src_app.max_entanglements = 1
    src_app.start(dest_node_name, start_time, end_time, memory_size, target_fidelity)

    tl.init()
    tl.run()

    stats = src_app.get_statistics()
    print(f"Completed: {stats['completed']}")
    print(f"Timeout triggered: {stats['timeout_triggered']}")
    print(f"Total entanglements generated: {stats['total_entanglements']}")
    if stats.get('overall_fidelity_stats'):
        print(f"Average fidelity: {stats['overall_fidelity_stats']['avg']:.6f}")
    return stats


def five_node_physical_pair_with_app_stabilizer(config_file='config/line_5_physical_stabilizer.json'):
    """True no-QEC 5-node baseline under stabilizer formalism."""
    print('\n5-node physical pairs using Stabilizers with RequestApp:')

    QuantumManager.set_global_manager_formalism(STABILIZER_FORMALISM)
    EntanglementGenerationA.set_global_type('barret_kok_stabilizer')
    EntanglementGenerationB.set_global_type('barret_kok_stabilizer')

    log_filename = 'log/line_5_physical_stabilizer_app'
    network_topo = RouterNetTopo(config_file)
    tl = network_topo.get_timeline()

    log.set_logger(__name__, tl, log_filename)
    log.set_logger_level('INFO')

    src_node_name = 'router_0'
    dest_node_name = 'router_4'
    src_app = None
    dest_app = None

    for router in network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER):
        app = RequestAppThroughput(router)
        if router.name == src_node_name:
            src_app = app
        elif router.name == dest_node_name:
            dest_app = app

    if src_app is None or dest_app is None:
        raise RuntimeError("Failed to find source/destination apps for 5-node stabilizer baseline")

    start_time = 1e12
    end_time = 10e12
    memory_size = 1
    target_fidelity = 0.01
    src_app.max_entanglements = 1
    dest_app.max_entanglements = 1
    src_app.start(dest_node_name, start_time, end_time, memory_size, target_fidelity)

    tl.init()
    tl.run()

    stats = dest_app.get_statistics()
    print(f"Completed: {stats['completed']}")
    print(f"Timeout triggered: {stats['timeout_triggered']}")
    print(f"Total entanglements generated: {stats['total_entanglements']}")
    if stats.get('overall_fidelity_stats'):
        print(f"Average fidelity: {stats['overall_fidelity_stats']['avg']:.6f}")
    return stats


def create_logical_bell_pair(verbose=True):
    """Create one logical Bell pair using [[7,1,3]] encoding on each node.
    """  
    QuantumManager.set_global_manager_formalism(STABILIZER_FORMALISM)    
    EntanglementGenerationA.set_global_type('barret_kok_stabilizer')
    EntanglementGenerationB.set_global_type('barret_kok_stabilizer')
    
    log_filename = 'log/stabilizer_logical_pair_entanglement_generation_app'
    network_config = 'config/line_2_physical_stabilizer.json'
    
    network_topo = RouterNetTopo(network_config)
    tl = network_topo.get_timeline()
    
    log.set_logger(__name__, tl, log_filename)
    log.set_logger_level('INFO')
    # CRITICAL: Don't track 'request_app' module
    modules = ['timeline', 'network_manager', 'resource_manager', 'rule_manager', 
               'generation', 'purification', 'swapping', 'bsm']
    for module in modules:
        log.track_module(module)
    
    src_node_name = 'router_0'
    dest_node_name = 'router_1'
    src_node = None
    
    for router in network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER):
        if router.name == src_node_name:
            src_node = router
            break
    
    # Use the old approach for stability
    start_time = 1e12
    end_time = 10e12
    entanglement_number = 7
    nm = src_node.network_manager
    nm.request(dest_node_name, start_time=start_time, end_time=end_time, 
               memory_size=entanglement_number, target_fidelity=0.8)
    
    tl.init()
    tl.run()
    
    # Display results
    latencies = []
    if verbose:
        print(f"{src_node_name} memories:")
        print("{:5}  {:14}  {:8}  {:>7}".format("Index", "Entangled Node", "Fidelity", "Latency"))
    
    for info in src_node.resource_manager.memory_manager:
        latency = (info.entangle_time - start_time) * 1e-12
        if latency < 0:
            break
        latencies.append(latency)
        if verbose:
            print("{:5}  {:>14}  {:8.5f}  {:.5f}".format(
                info.index, str(info.remote_node), float(info.fidelity), latency))
    
    if latencies:
        latency = np.average(latencies)
        print(f'Average latency = {latency:.4f}s; rate = {1/latency:.3f}/s')
        print(f'Total entanglements generated: {len(latencies)}')


def three_node_logical_pair_with_app(verbose=False, config_file='config/line_3_2G.json', css_code="[[7,1,3]]"):
    """Create a logical Bell pair between router_0 and router_2 via router_1.

    3-node linear chain: router_0 -- router_1 -- router_2

    Pipeline:
    1. Generate n physical Bell pairs on each link in parallel
    2. Encode logical qubits at all 3 nodes (router_1 encodes two: left and right)
    3. Teleported CNOT on each link to create logical entanglement per link
    4. Logical entanglement swapping at router_1 to extend to router_0 <-> router_2
    """

    # Set quantum manager to use stabilizer formalism
    QuantumManager.set_global_manager_formalism(STABILIZER_FORMALISM)

    # Set protocols to use stabilizer versions
    EntanglementGenerationA.set_global_type('barret_kok_stabilizer')
    EntanglementGenerationB.set_global_type('barret_kok_stabilizer')

    log_filename = 'log/three_node_logical_pair'
    network_config = config_file

    network_topo = RouterNetTopo2G(network_config)
    tl = network_topo.get_timeline()

    # Set gate fidelities on quantum manager from node config
    routers = network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER)
    tl.quantum_manager.gate_fid = getattr(routers[0], 'gate_fid', 1.0)
    tl.quantum_manager.two_qubit_gate_fid = getattr(routers[0], 'two_qubit_gate_fid', 1.0)
    idle_pauli_weights = {"x": 0.05, "y": 0.05, "z": 0.90}
    idle_data_coherence_time_sec = 1e30
    idle_comm_coherence_time_sec = 1e30

    log.set_logger(__name__, tl, log_filename)
    log.set_logger_level('DEBUG')
    modules = ['timeline', 'network_manager', 'resource_manager', 'rule_manager',
               'generation', 'purification', 'swapping', 'bsm']
    for module in modules:
        log.track_module(module)

    # ========================================================================
    # Set up applications on all 3 nodes
    # ========================================================================
    node_names = []
    apps = {}

    for router in network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER):
        if not isinstance(router, QuantumRouter2ndGeneration):
            raise TypeError(f"Node {router.name} must be QuantumRouter2ndGeneration")

        app = RequestLogicalPairApp(router, css_code=css_code)
        apps[router.name] = app
        node_names.append(router.name)

    # Sort to ensure linear order
    node_names.sort(key=lambda name: int(name.split('_')[-1]))
    assert len(node_names) >= 2, f"Need at least 2 nodes, got {len(node_names)}"

    # ========================================================================
    # Configure swap schedule (LOCC compliant — per-node configs)
    # ========================================================================
    swap_configs, final_swap_node = RequestLogicalPairApp.build_swap_schedule(node_names)
    for node_name, config in swap_configs.items():
        apps[node_name].set_swap_config(config)

    # Designate which node stops the simulation
    if final_swap_node:
        apps[final_swap_node].set_final_action_node()
    else:
        # No swaps (2-node chain) — initiator stops after link fidelity
        apps[node_names[0]].set_final_action_node()

    # ========================================================================
    # Configure requests - one reservation per link
    # Each link: left node (Alice/initiator) -> right node (Bob/responder)
    # All links start at the same time so Bell pairs generate in parallel
    # ========================================================================
    start_time = 1e12   # 1 second
    end_time = 10e12    # 10 seconds
    target_fidelity = 0.8

    num_links = len(node_names) - 1
    for i in range(num_links):
        apps[node_names[i]].start(
            responder=node_names[i + 1],
            start_time=start_time,
            end_time=end_time,
            target_fidelity=target_fidelity
        )

    # ========================================================================
    # Run simulation
    # ========================================================================
    tl.init()
    tl.run()

    print(f"\n--- {num_links}-link pipeline complete ---")

    return apps


def five_node_logical_pair_with_app(verbose=False, config_file='config/line_5_2G_near_term.json', css_code="[[7,1,3]]"):
    """Create end-to-end logical entanglement across a 5-node chain.

    5-node linear chain: router_0 -- router_1 -- router_2 -- router_3 -- router_4

    Pipeline per link:
    1. Generate 7 physical Bell pairs on each link in parallel
    2. Encode logical qubits at all nodes (middle nodes encode two: left and right)
    3. Teleported CNOT on each link to create logical entanglement per link
    4. Logical entanglement swapping at each middle node
    """

    # Set quantum manager to use stabilizer formalism
    QuantumManager.set_global_manager_formalism(STABILIZER_FORMALISM)

    # Set protocols to use stabilizer versions
    EntanglementGenerationA.set_global_type('barret_kok_stabilizer')
    EntanglementGenerationB.set_global_type('barret_kok_stabilizer')

    log_filename = 'log/five_node_logical_pair'
    network_config = resolve_config_path(config_file)

    network_topo = RouterNetTopo2G(network_config)
    tl = network_topo.get_timeline()

    # Set gate fidelities on quantum manager from node config
    routers = network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER)
    tl.quantum_manager.gate_fid = getattr(routers[0], 'gate_fid', 1.0)
    tl.quantum_manager.two_qubit_gate_fid = getattr(routers[0], 'two_qubit_gate_fid', 1.0)
    idle_pauli_weights = {"x": 0.0, "y": 0.0, "z": 0.0}
    idle_data_coherence_time_sec = 1e12
    idle_comm_coherence_time_sec = 1e12

    log.set_logger(__name__, tl, log_filename)
    log.set_logger_level('DEBUG')
    modules = ['timeline', 'network_manager', 'resource_manager', 'rule_manager',
               'generation', 'purification', 'swapping', 'bsm']
    for module in modules:
        log.track_module(module)

    # ========================================================================
    # Set up applications on all 5 nodes
    # ========================================================================
    node_names = []
    apps = {}

    for router in network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER):
        if not isinstance(router, QuantumRouter2ndGeneration):
            raise TypeError(f"Node {router.name} must be QuantumRouter2ndGeneration")

        app = RequestLogicalPairApp(router, css_code=css_code)
        apps[router.name] = app
        node_names.append(router.name)

    # Sort to ensure linear order
    node_names.sort(key=lambda name: int(name.split('_')[-1]))
    assert len(node_names) == 5, f"Expected 5 nodes, got {len(node_names)}"

    # ========================================================================
    # Configure swap schedule (LOCC compliant — per-node configs)
    # ========================================================================
    swap_configs, final_swap_node = RequestLogicalPairApp.build_swap_schedule(node_names)
    for node_name, config in swap_configs.items():
        apps[node_name].set_swap_config(config)

    # Designate which node stops the simulation
    if final_swap_node:
        apps[final_swap_node].set_final_action_node()
    else:
        apps[node_names[0]].set_final_action_node()

    # ========================================================================
    # Configure requests - one reservation per link
    # Each link: left node (Alice/initiator) -> right node (Bob/responder)
    # All links start at the same time so Bell pairs generate in parallel
    # ========================================================================
    start_time = 1e12   # 1 second
    end_time = 10e12    # 10 seconds
    target_fidelity = 0.8

    num_links = len(node_names) - 1
    for i in range(num_links):
        apps[node_names[i]].start(
            responder=node_names[i + 1],
            start_time=start_time,
            end_time=end_time,
            target_fidelity=target_fidelity
        )

    # ========================================================================
    # Run simulation
    # ========================================================================
    tl.init()
    tl.run()

    metrics = {
        "css_code": css_code,
        "config_file": config_file,
        "gate_fidelity": tl.quantum_manager.gate_fid,
        "two_qubit_gate_fidelity": tl.quantum_manager.two_qubit_gate_fid,
        "link_rows": [],
        "avg_initial_phys": float("nan"),
        "avg_link_logical": float("nan"),
        "end_to_end_logical": float("nan"),
    }

    print("\nLink Summary")
    initial_values = []
    logical_values = []
    for i in range(num_links):
        left = node_names[i]
        right = node_names[i + 1]
        left_app = apps[left]
        initial_phys = left_app.current_run["initial_link_fidelities"].get(right, float("nan"))
        logical = left_app.current_run["link_logical_fidelities"].get(right, float("nan"))
        metrics["link_rows"].append(
            {
                "left": left,
                "right": right,
                "link": f"{left}-{right}",
                "link_index": i,
                "initial_phys": float(initial_phys),
                "prep": float(left_app.prep_fidelity),
                "ft_mode": left_app.ft_prep_mode,
                "logical": float(logical),
            }
        )
        if not np.isnan(initial_phys):
            initial_values.append(initial_phys)
        if not np.isnan(logical):
            logical_values.append(logical)
        print(
            f"{left} <-> {right} | "
            f"initial_phys={initial_phys:.6f} | "
            f"prep={left_app.prep_fidelity:.6f} | "
            f"ft={left_app.ft_prep_mode} | "
            f"logical={logical:.6f}"
        )

    final_app = apps[node_names[0]]
    if final_app.current_run["final_end_to_end_fidelity"] is not None:
        metrics["end_to_end_logical"] = float(final_app.current_run["final_end_to_end_fidelity"])
        print("\nEnd-to-End")
        print(
            f"{node_names[0]} <-> {node_names[-1]} | "
            f"end_to_end_logical={final_app.current_run['final_end_to_end_fidelity']:.6f}"
        )

    if initial_values:
        metrics["avg_initial_phys"] = float(np.mean(initial_values))
    if logical_values:
        metrics["avg_link_logical"] = float(np.mean(logical_values))

    print(f"\n--- {num_links}-link pipeline complete ---")

    return {"apps": apps, "metrics": metrics}


def n_node_logical_pair_with_app(verbose: bool = False):
    """Create end-to-end logical entanglement across an N-node linear chain.

    Args:
        verbose: Whether to print verbose diagnostics.

    Returns:
        dict[str, object]: App objects and collected run metrics.
    """
    config_file = 'config/line_5_2G_near_term.json'
    css_code = "[[7,1,3]]"

    log_filename = 'log/n_node_logical_pair'
    network_config = resolve_config_path(config_file)

    network_topo = RouterNetTopo2G(network_config)
    tl = network_topo.get_timeline()

    routers = network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER)
    tl.quantum_manager.gate_fid = getattr(routers[0], 'gate_fid', 1.0)
    tl.quantum_manager.two_qubit_gate_fid = getattr(routers[0], 'two_qubit_gate_fid', 1.0)
    idle_pauli_weights = {"x": 0.05, "y": 0.05, "z": 0.90}
    idle_data_coherence_time_sec = 1e-1
    idle_comm_coherence_time_sec = 1e-1

    log.set_logger(__name__, tl, log_filename)
    log.set_logger_level('DEBUG')
    modules = ['timeline', 'network_manager', 'resource_manager', 'rule_manager', 'generation', 'purification', 'swapping', 'bsm', 'barret_kok', 'RequestLogicalPairApp', 'TeleportedCNOT', 'QREProtocol']
    for module in modules:
        log.track_module(module)

    node_names = []
    routers_by_name = {}
    for router in routers:
        if not isinstance(router, QuantumRouter2ndGeneration):
            raise TypeError(f"Node {router.name} must be QuantumRouter2ndGeneration")
        routers_by_name[router.name] = router
        node_names.append(router.name)

    node_names.sort(key=lambda name: int(name.split('_')[-1]))
    assert len(node_names) >= 2, f"Expected at least 2 nodes, got {len(node_names)}"

    start_time_ps = int(1e12)
    window_duration_ps = int(1e9)
    round_spacing_ps = int(5e9)
    default_target_fidelity = 0.8
    num_logical_pairs = 30

    apps = {}
    for node_name in node_names:
        routers_by_name[node_name].idle_pauli_weights = dict(idle_pauli_weights)
        routers_by_name[node_name].idle_data_coherence_time_sec = float(idle_data_coherence_time_sec)
        routers_by_name[node_name].idle_comm_coherence_time_sec = float(idle_comm_coherence_time_sec)
        apps[node_name] = RequestLogicalPairApp(
            routers_by_name[node_name],
            css_code=css_code,
            path_node_names=node_names)

    num_links = len(node_names) - 1
    for i in range(num_links):
        apps[node_names[i]].start(
            responder=node_names[i + 1],
            start_t=start_time_ps,
            end_t=start_time_ps + window_duration_ps,
            fidelity=default_target_fidelity,
            num_logical_pairs=num_logical_pairs,
            round_spacing_ps=round_spacing_ps)

    tl.init()
    tl.run()

    def _collect_metrics() -> dict[str, object]:
        """Build metrics for the completed run.

        Args:
            None.

        Returns:
            dict[str, object]: Aggregated metrics dictionary for this run.
        """
        # Run-level metadata and output containers.
        metrics = {
            "css_code": css_code,
            "config_file": config_file,
            "gate_fidelity": tl.quantum_manager.gate_fid,
            "two_qubit_gate_fidelity": tl.quantum_manager.two_qubit_gate_fid,
            "num_nodes": len(node_names),
            "num_links": num_links,
            "num_logical_pairs_requested": num_logical_pairs,
            "num_logical_pairs_completed": 0,
            "link_rows": [],
            "end_to_end_rows": [],
            "avg_initial_phys": float("nan"),
            "avg_link_logical": float("nan"),
            "avg_end_to_end_logical": float("nan"),
            "avg_latency_ps": float("nan"),
            "avg_latency_s": float("nan"),
            "avg_throughput_pairs_per_s": float("nan"),
        }

        # Per-link metrics from each left-side app instance.
        for link_index, (left, right) in enumerate(zip(node_names[:-1], node_names[1:])):
            left_app = apps[left]
            left_run_stats = None
            if left_app.run_stats:
                left_latest_run_id = max(left_app.run_stats)
                left_run_stats = left_app.run_stats[left_latest_run_id]

            initial_phys = float("nan")
            logical = float("nan")
            if left_run_stats is not None:
                initial_phys = float(left_run_stats["initial_link_fidelities"].get(right, float("nan")))
                logical = float(left_run_stats["link_logical_fidelities"].get(right, float("nan")))

            row = {
                "left": left,
                "right": right,
                "link": f"{left}-{right}",
                "link_index": link_index,
                "initial_phys": initial_phys,
                "prep": float(left_app.prep_fidelity),
                "ft_mode": left_app.ft_prep_mode,
                "logical": logical,
            }
            metrics["link_rows"].append(row)

        final_app = apps[node_names[0]]
        if final_app.run_stats:
            for run_id in sorted(final_app.run_stats):
                run_stats = final_app.run_stats[run_id]
                latency_ps = run_stats["latency_ps"]
                latency_s = float("nan")
                throughput_pairs_per_s = float("nan")
                if latency_ps is not None:
                    latency_ps = float(latency_ps)
                    latency_s = latency_ps * 1e-12
                    throughput_pairs_per_s = 0.0 if latency_s <= 0.0 else 1.0 / latency_s

                metrics["end_to_end_rows"].append({"run_id": int(run_id), "fidelity": float(run_stats["final_end_to_end_fidelity"]) if run_stats["final_end_to_end_fidelity"] is not None else float("nan"), "latency_ps": latency_ps if latency_ps is not None else float("nan"), "latency_s": latency_s, "throughput_pairs_per_s": throughput_pairs_per_s})
        metrics["num_logical_pairs_completed"] = len(metrics["end_to_end_rows"])
        # Aggregate link-level averages.
        initial_values = [r["initial_phys"] for r in metrics["link_rows"] if not np.isnan(r["initial_phys"])]
        logical_values = [r["logical"] for r in metrics["link_rows"] if not np.isnan(r["logical"])]
        if initial_values:
            metrics["avg_initial_phys"] = float(np.mean(initial_values))
        if logical_values:
            metrics["avg_link_logical"] = float(np.mean(logical_values))
        e2e_values = [r["fidelity"] for r in metrics["end_to_end_rows"] if not np.isnan(r["fidelity"])]
        latency_values = [r["latency_ps"] for r in metrics["end_to_end_rows"] if not np.isnan(r["latency_ps"])]
        throughput_values = [r["throughput_pairs_per_s"] for r in metrics["end_to_end_rows"] if not np.isnan(r["throughput_pairs_per_s"])]
        if e2e_values:
            metrics["avg_end_to_end_logical"] = float(np.mean(e2e_values))
        if latency_values:
            metrics["avg_latency_ps"] = float(np.mean(latency_values))
            metrics["avg_latency_s"] = metrics["avg_latency_ps"] * 1e-12
        if throughput_values:
            metrics["avg_throughput_pairs_per_s"] = float(np.mean(throughput_values))

        return metrics

    def _print_metrics_summary(metrics: dict[str, object]) -> None:
        """Print a compact human-readable summary from collected metrics.

        Args:
            metrics: Metrics dictionary returned by _collect_metrics.

        Returns:
            None.
        """
        final_app = apps[node_names[0]]

        # Header and run parameters.
        print("\n=== Run Summary ===")
        print(f"code={css_code} | nodes={len(node_names)} | links={num_links} | gate_fid={metrics['gate_fidelity']:.4f} | twoq_fid={metrics['two_qubit_gate_fidelity']:.4f}")
        print(f"params: n={final_app.n} ft_mode={final_app.ft_prep_mode} idle_weights={final_app.idle_pauli_weights} data_t2={final_app.idle_data_coherence_time_sec:.3e}s comm_t2={final_app.idle_comm_coherence_time_sec:.3e}s")
        print(f"logical_pairs: requested={metrics['num_logical_pairs_requested']} completed={metrics['num_logical_pairs_completed']}")

        if verbose:
            header = f"{'Link':<18} {'Phys':>8} {'Logical':>8} {'Prep':>8} {'FT':>10}"
            print(header)
            print("-" * len(header))
            for row in metrics["link_rows"]:
                print(f"{row['link']:<18} {row['initial_phys']:>8.4f} {row['logical']:>8.4f} {row['prep']:>8.4f} {str(row['ft_mode']):>10}")
            print(f"\nAverages: phys={metrics['avg_initial_phys']:.4f} | logical={metrics['avg_link_logical']:.4f}")
        if verbose and metrics["end_to_end_rows"]:
            print(f"\nLogical Pair Runs ({node_names[0]} <-> {node_names[-1]})")
            header = f"{'Run':>4} {'E2E Fid':>10} {'Latency (ps)':>14} {'Latency (ms)':>14} {'Throughput':>14}"
            print(header)
            print("-" * len(header))
            for row in metrics["end_to_end_rows"]:
                print(f"{row['run_id']:>4d} {row['fidelity']:>10.4f} {row['latency_ps']:>14.0f} {row['latency_ps'] * 1e-9:>14.6f} {row['throughput_pairs_per_s']:>14.6e}")

            print(f"{'Avg':>4} {metrics['avg_end_to_end_logical']:>10.4f} {metrics['avg_latency_ps']:>14.0f} {metrics['avg_latency_ps'] * 1e-9:>14.6f} {metrics['avg_throughput_pairs_per_s']:>14.6e}")
        if not np.isnan(metrics["avg_end_to_end_logical"]):
            print(f"Avg end-to-end ({node_names[0]} <-> {node_names[-1]}): {metrics['avg_end_to_end_logical']:.4f}")
        if not np.isnan(metrics["avg_latency_ps"]):
            print(f"Avg latency: {metrics['avg_latency_ps']:.0f} ps ({metrics['avg_latency_s']:.6e} s, {metrics['avg_latency_ps'] * 1e-9:.6f} ms)")
            print(f"Avg throughput: {metrics['avg_throughput_pairs_per_s']:.6e} pairs/s")
    
    metrics = _collect_metrics()
    _print_metrics_summary(metrics)
    print(f"=== {num_links}-link pipeline complete ===")
    return {"apps": apps, "metrics": metrics}

if __name__ == "__main__":

    # ----- N-node runs -----
    n_node_logical_pair_with_app(verbose=False)

    # Add more N-node runs as needed:
    # n_node_logical_pair_with_app(verbose=True)
