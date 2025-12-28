import numpy as np
from sequence.topology.router_net_topo import RouterNetTopo
from router_net_topo_2G import RouterNetTopo2G
import sequence.utils.log as log
from sequence.kernel.quantum_manager import QuantumManager
from sequence.constants import STABILIZER_FORMALISM
from sequence.entanglement_management.generation import EntanglementGenerationA, EntanglementGenerationB

from request_app import RequestAppThroughput
from RequestLogicalPairApp import RequestLogicalPairApp
from sequence.topology.node import QuantumRouter2ndGeneration


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


def two_node_logical_pair_with_app(verbose=False):
    """Demonstrate [[7,1,3]] QEC encoding with RequestLogicalPairApp.

    Generates 7 Bell pairs (in communication memories) and separately
    encodes |0⟩_L (in data memories) to showcase both capabilities.
    """
    print('\n[[7,1,3]] Steane Code Demonstration:')

    # Set quantum manager to use stabilizer formalism
    QuantumManager.set_global_manager_formalism(STABILIZER_FORMALISM)

    # Set protocols to use stabilizer versions
    EntanglementGenerationA.set_global_type('barret_kok_stabilizer')
    EntanglementGenerationB.set_global_type('barret_kok_stabilizer')

    log_filename = 'log/logical_pair_generation_app'
    network_config = 'config/line_2_2nd_gen_stabilizer.json'
    # network_config = 'config/line_3_2G.json'


    network_topo = RouterNetTopo2G(network_config)
    tl = network_topo.get_timeline()
    
    log.set_logger(__name__, tl, log_filename)
    log.set_logger_level('DEBUG')
    modules = ['timeline', 'network_manager', 'resource_manager', 'rule_manager', 
               'generation', 'purification', 'swapping', 'bsm']
    for module in modules:
        log.track_module(module)
    
    # Set up applications on BOTH nodes
    apps = []
    src_node_name = 'router_0'
    dest_node_name = 'router_1' # TODO: change to router_2 for 3-node test
    src_app = None
    dest_app = None
    
    for router in network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER):
        # Verify it's 2nd generation
        if not isinstance(router, QuantumRouter2ndGeneration):
            raise TypeError(f"Node {router.name} must be QuantumRouter2ndGeneration")
        
        # Create app on each node
        app = RequestLogicalPairApp(router)
        apps.append(app)
        
        if router.name == src_node_name:
            src_app = app
        elif router.name == dest_node_name:
            dest_app = app
    
    # Configure request
    start_time = 1e12   # 1 second
    end_time = 10e12    # 10 seconds
    target_fidelity = 0.8
    
    # Request 1 or 2 logical pairs (limited by memory constraints)
    num_logical_pairs = 1  # Start with 1 for testing
    
    for i in range(num_logical_pairs):
        request_start = start_time + i * 2e11  # 200ms between requests
        
        # Only source initiates

        src_app.start(
            responder=dest_node_name,
            start_time=request_start,
            end_time=end_time,
            target_fidelity=target_fidelity,
            teleported_cnot_enabled=True
        )
    
    tl.init()
    tl.run()

    # Display results
    print("\n--- Results from RequestLogicalPairApp ---")

    # Get results from source
    src_results = src_app.get_results()

    # Display available metrics
    print(f"\nSource Node ({src_results['node_name']}) - Role: {src_results['role']}")

    # Timing information
    if src_results['timing'].get('generation_time'):
        print(f"  Bell pair generation time: {src_results['timing']['generation_time']:.4f}s")

    if src_results['encoding'].get('encoding_time'):
        print(f"  Encoding time: {src_results['encoding']['encoding_time']:.6f}s")

    # Calculate total time if available
    if src_results['timing'].get('first_pair_time') and src_results['timing'].get('last_pair_time'):
        total_time = src_results['timing']['last_pair_time'] - src_results['timing']['first_pair_time']
        print(f"  Total time: {total_time:.4f}s")

    # Extract and display Bell pair fidelities
    fidelities = [bp['fidelity'] for bp in src_results['bell_pairs'] if bp['fidelity'] is not None]
    if fidelities:
        print(f"  Bell pair fidelities: {[f'{f:.4f}' for f in fidelities]}")
        print(f"  Average Bell fidelity: {np.mean(fidelities):.4f}")

    print(f"  Encoding complete: {src_results['encoding']['success']}")

    # Display logical Bell pair fidelity if available
    if 'logical_bell_pair' in src_results and src_results['logical_bell_pair']['fidelity'] is not None:
        print(f"\n  Logical Bell Pair Fidelity: {src_results['logical_bell_pair']['fidelity']:.4f}")
        correlations = src_results['logical_bell_pair']['correlations']
        print(f"    <XX> = {correlations['XX']:.4f}")
        print(f"    <YY> = {correlations['YY']:.4f}")
        print(f"    <ZZ> = {correlations['ZZ']:.4f}")

    # Display product state fidelity if available (when teleported_cnot_enabled=False)
    if 'product_state_fidelity' in src_results:
        prod_fid = src_results['product_state_fidelity']
        print(f"\n  Product State Fidelity (|+>_L x |0>_L):")
        print(f"    Alice P(X_L=+1) for |+>_L: {prod_fid['alice_x_prob']:.4f}")
        print(f"    Bob   P(Z_L=+1) for |0>_L: {prod_fid['bob_z_prob']:.4f}")
        print(f"    Combined fidelity: {prod_fid['fidelity']:.4f}")

    # Destination results
    dest_results = dest_app.get_results()
    print(f"\nDestination Node ({dest_results['node_name']}) - Role: {dest_results['role']}")
    print(f"  Encoding complete: {dest_results['encoding']['success']}")

    return src_app, dest_app


if __name__ == "__main__":

    # two_node_physical_pair_with_app_ketstate(verbose=False)
    
    # two_node_physical_pair_with_app_stabilizer()
    
    two_node_logical_pair_with_app()
        
    pass