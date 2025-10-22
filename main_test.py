import numpy as np
from sequence.topology.router_net_topo import RouterNetTopo, RouterNetTopo2G
import sequence.utils.log as log
from sequence.kernel.quantum_manager import QuantumManager
from sequence.constants import STABILIZER_FORMALISM
from sequence.entanglement_management.generation import EntanglementGenerationA, EntanglementGenerationB

from request_app import RequestAppThroughput

from RequestLogicalPairApp import RequestLogicalPairApp
from sequence.topology.node import QuantumRouter2ndGeneration

def two_node_physical(verbose=False):


    print('\nPhysical Entagled Pairs/s:')

    log_filename = 'log/linear_entanglement_generation'
    # level = logging.DEBUG
    # logging.basicConfig(level=level, filename='', filemode='w')
    
    network_config = 'config/line_2_physical.json'
    # network_config = 'config/random_5.json'
    network_topo = RouterNetTopo(network_config)
    tl = network_topo.get_timeline()

    log.set_logger(__name__, tl, log_filename)
    log.set_logger_level('INFO')
    modules = ['timeline', 'network_manager', 'resource_manager', 'rule_manager', 'generation', 'purification', 'swapping', 'bsm']
    for module in modules:
        log.track_module(module)

    src_node_name  = 'router_0'
    dest_node_name = 'router_1'
    src_node = None
    for router in network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER):
        if router.name == src_node_name:
            src_node = router
            break
    
    start_time = 1e12
    end_time   = 10e12
    entanglement_number = 20
    nm = src_node.network_manager
    nm.request(dest_node_name, start_time=start_time, end_time=end_time, memory_size=entanglement_number, target_fidelity=0.8)

    tl.init()
    tl.run()

    latencies = []
    if verbose:
        print(src_node_name, "memories:")
        print("{:5}  {:14}  {:8}  {:>7}".format("Index", "Entangled Node", "Fidelity", "Latency"))
    for info in src_node.resource_manager.memory_manager:
        latency = (info.entangle_time - start_time) * 1e-12
        if latency < 0:
            break
        latencies.append(latency)
        if verbose:
            print("{:5}  {:>14}  {:8.5f}  {:.5f}".format(info.index, str(info.remote_node), float(info.fidelity), latency))
    latency = np.average(latencies)
    print(f'average latency = {latency:.4f}s; rate = {1/latency:.3f}/s')

    pass

# Two node topology, 20 memory generating entanglements, Ket State formalism
# see how many entanglement pairs are generated in 10 seconds with 1.0 fidelity and 0.9 efficiency
def two_node_physical_twenty_memories_ketState_old(verbose=False):


    print('Physical Entangled Pairs using Ket State:')

    log_filename = 'log/ketstate_entanglement_generation'  
    network_config = 'config/line_2_physical_ketState.json'

    network_topo = RouterNetTopo(network_config)
    tl = network_topo.get_timeline()

    log.set_logger(__name__, tl, log_filename)
    log.set_logger_level('INFO')
    modules = ['timeline', 'network_manager', 'resource_manager', 'rule_manager', 'generation', 'purification', 'swapping', 'bsm']
    for module in modules:
        log.track_module(module)

    src_node_name  = 'router_0'
    dest_node_name = 'router_1'
    src_node = None
    for router in network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER):
        if router.name == src_node_name:
            src_node = router
            break
    
    start_time = 1e12
    end_time   = 10e12
    entanglement_number = 10
    nm = src_node.network_manager
    nm.request(dest_node_name, start_time=start_time, end_time=end_time, memory_size=entanglement_number, target_fidelity=0.8)

    tl.init()
    tl.run()

    latencies = []
    if verbose:
        print(src_node_name, "memories:")
        print("{:5}  {:14}  {:8}  {:>7}".format("Index", "Entangled Node", "Fidelity", "Latency"))
    for info in src_node.resource_manager.memory_manager:
        latency = (info.entangle_time - start_time) * 1e-12
        if latency < 0:
            break
        latencies.append(latency)
        if verbose:
            print("{:5}  {:>14}  {:8.5f}  {:.5f}".format(info.index, str(info.remote_node), float(info.fidelity), latency))
    latency = np.average(latencies)
    print(f'average latency = {latency:.4f}s; rate = {1/latency:.3f}/s')

    pass

# Two node topology, 20 memories generating entanglements, Stabilizer formalism
# see how many entanglement pairs are generated in 10 seconds with 1.0 fidelity and 0.9 efficiency
def two_node_physical_twenty_memories_stabilizer_old(verbose=True):
    print('Physical Entangled Pairs using Stabilizers:')

    # Set quantum manager to use stabilizer formalism
    QuantumManager.set_global_manager_formalism(STABILIZER_FORMALISM)
    
    # Set protocols to use stabilizer versions
    EntanglementGenerationA.set_global_type('barret_kok_stabilizer')
    EntanglementGenerationB.set_global_type('barret_kok_stabilizer')
    log_filename = 'log/stabilizer_entanglement_generation'
    # level = logging.DEBUG
    # logging.basicConfig(level=level, filename='', filemode='w')
    
    network_config = 'config/line_2_physical_stabilizer.json'
    # network_config = 'config/random_5.json'
    network_topo = RouterNetTopo(network_config)
    tl = network_topo.get_timeline()

    log.set_logger(__name__, tl, log_filename)
    log.set_logger_level('INFO')
    modules = ['timeline', 'network_manager', 'resource_manager', 'rule_manager', 'generation', 'purification', 'swapping', 'bsm']
    for module in modules:
        log.track_module(module)

    src_node_name  = 'router_0'
    dest_node_name = 'router_1'
    src_node = None
    for router in network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER):
        if router.name == src_node_name:
            src_node = router
            break
    
    start_time = 1e12
    end_time   = 10e12
    entanglement_number = 10
    nm = src_node.network_manager
    nm.request(dest_node_name, start_time=start_time, end_time=end_time, memory_size=entanglement_number, target_fidelity=0.8)

    tl.init()
    tl.run()

    latencies = []
    if verbose:
        print(src_node_name, "memories:")
        print("{:5}  {:14}  {:8}  {:>7}".format("Index", "Entangled Node", "Fidelity", "Latency"))
    for info in src_node.resource_manager.memory_manager:
        latency = (info.entangle_time - start_time) * 1e-12
        if latency < 0:
            break
        latencies.append(latency)
        if verbose:
            print("{:5}  {:>14}  {:8.5f}  {:.5f}".format(info.index, str(info.remote_node), float(info.fidelity), latency))
    latency = np.average(latencies)
    print(f'average latency = {latency:.4f}s; rate = {1/latency:.3f}/s')

    pass


# Two node topology, 20 memory generating entanglements, Ket State formalism
# see how many entanglement pairs are generated in 10 seconds with 1.0 fidelity and 0.9 efficiency
def two_node_physical_twenty_memories_ketState(verbose=False):


    print('Physical Entangled Pairs using Ket State:')

    log_filename = 'log/ketstate_entanglement_generation'  
    network_config = 'config/line_2_physical_ketState.json'

    network_topo = RouterNetTopo(network_config)
    tl = network_topo.get_timeline()

    log.set_logger(__name__, tl, log_filename)
    log.set_logger_level('INFO')
    modules = ['timeline', 'network_manager', 'resource_manager', 'rule_manager', 'generation', 'purification', 'swapping', 'bsm']
    for module in modules:
        log.track_module(module)

    src_node_name  = 'router_0'
    dest_node_name = 'router_1'
    src_node = None
    for router in network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER):
        if router.name == src_node_name:
            src_node = router
            break
    
    start_time = 1e12
    end_time   = 10e12
    entanglement_number = 100
    nm = src_node.network_manager
    nm.request(dest_node_name, start_time=start_time, end_time=end_time, memory_size=entanglement_number, target_fidelity=0.8)

    tl.init()
    tl.run()

    latencies = []
    if verbose:
        print(src_node_name, "memories:")
        print("{:5}  {:14}  {:8}  {:>7}".format("Index", "Entangled Node", "Fidelity", "Latency"))
    for info in src_node.resource_manager.memory_manager:
        latency = (info.entangle_time - start_time) * 1e-12
        if latency < 0:
            break
        latencies.append(latency)
        if verbose:
            print("{:5}  {:>14}  {:8.5f}  {:.5f}".format(info.index, str(info.remote_node), float(info.fidelity), latency))
    latency = np.average(latencies)
    print(f'average latency = {latency:.4f}s; rate = {1/latency:.3f}/s')

    pass

# Two node topology, 20 memories generating entanglements, Stabilizer formalism
# see how many entanglement pairs are generated in 10 seconds with 1.0 fidelity and 0.9 efficiency
def two_node_physical_twenty_memories_stabilizer(verbose=True):
    print('Physical Entangled Pairs using Stabilizers:')

    # Set quantum manager to use stabilizer formalism
    QuantumManager.set_global_manager_formalism(STABILIZER_FORMALISM)
    
    # Set protocols to use stabilizer versions
    EntanglementGenerationA.set_global_type('barret_kok_stabilizer')
    EntanglementGenerationB.set_global_type('barret_kok_stabilizer')
    log_filename = 'log/stabilizer_entanglement_generation'
    # level = logging.DEBUG
    # logging.basicConfig(level=level, filename='', filemode='w')
    
    network_config = 'config/line_2_physical_stabilizer.json'
    # network_config = 'config/random_5.json'
    network_topo = RouterNetTopo(network_config)
    tl = network_topo.get_timeline()

    log.set_logger(__name__, tl, log_filename)
    log.set_logger_level('INFO')
    modules = ['timeline', 'network_manager', 'resource_manager', 'rule_manager', 'generation', 'purification', 'swapping', 'bsm']
    for module in modules:
        log.track_module(module)

    src_node_name  = 'router_0'
    dest_node_name = 'router_1'
    src_node = None
    for router in network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER):
        if router.name == src_node_name:
            src_node = router
            break
    
    start_time = 1e12
    end_time   = 10e12
    entanglement_number = 100
    nm = src_node.network_manager
    nm.request(dest_node_name, start_time=start_time, end_time=end_time, memory_size=entanglement_number, target_fidelity=0.8)

    tl.init()
    tl.run()

    latencies = []
    if verbose:
        print(src_node_name, "memories:")
        print("{:5}  {:14}  {:8}  {:>7}".format("Index", "Entangled Node", "Fidelity", "Latency"))
    for info in src_node.resource_manager.memory_manager:
        latency = (info.entangle_time - start_time) * 1e-12
        if latency < 0:
            break
        latencies.append(latency)
        if verbose:
            print("{:5}  {:>14}  {:8.5f}  {:.5f}".format(info.index, str(info.remote_node), float(info.fidelity), latency))
    latency = np.average(latencies)
    print(f'average latency = {latency:.4f}s; rate = {1/latency:.3f}/s')

    pass


def two_node_physical_twenty_memories_ketState_with_app(verbose=False):
    """Test with ket state formalism - FIXED version.
    
    Uses same parameters as old version to ensure comparable behavior.
    """    
    log_filename = 'log/ketstate_entanglement_generation_app_fixed'
    network_config = 'config/line_2_physical_ketState.json'
    
    network_topo = RouterNetTopo(network_config)
    tl = network_topo.get_timeline()
    
    log.set_logger(__name__, tl, log_filename)
    log.set_logger_level('INFO')
    # CRITICAL: Don't track 'request_app' module to reduce logging
    modules = ['timeline', 'network_manager', 'resource_manager', 'rule_manager', 
               'generation', 'purification', 'swapping', 'bsm']
    for module in modules:
        log.track_module(module)
    
    # Set up nodes WITHOUT apps first
    src_node_name = 'router_0'
    dest_node_name = 'router_1'
    src_node = None
    
    for router in network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER):
        if router.name == src_node_name:
            src_node = router
            break
    
    # OPTION 1: Use the old approach (nm.request) for comparison
    # This ensures we get the same behavior as the old version
    start_time = 1e12
    end_time = 10e12
    entanglement_number = 20  # Same as old version
    nm = src_node.network_manager
    nm.request(dest_node_name, start_time=start_time, end_time=end_time, 
               memory_size=entanglement_number, target_fidelity=0.8)
    
    # OPTION 2: If you must use RequestAppThroughput, uncomment below:
    """
    # Create app AFTER setting up the network
    src_app = RequestAppThroughput(src_node)
    src_app.max_entanglements = 50  # Set reasonable limit
    
    # Start request with same parameters as old version
    src_app.start(dest_node_name, start_time, end_time, 
                  memory_size=10, fidelity=0.8)
    """
    
    tl.init()
    tl.run()
    
    # Display results (same as old version)
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


def two_node_physical_twenty_memories_stabilizer_with_app(verbose=False):
    """Test with stabilizer formalism - FIXED version.
    
    Uses same parameters as old version to ensure comparable behavior.
    """    
    # Set quantum manager to use stabilizer formalism
    QuantumManager.set_global_manager_formalism(STABILIZER_FORMALISM)
    
    # Set protocols to use stabilizer versions
    EntanglementGenerationA.set_global_type('barret_kok_stabilizer')
    EntanglementGenerationB.set_global_type('barret_kok_stabilizer')
    
    log_filename = 'log/stabilizer_entanglement_generation_app_fixed'
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
    entanglement_number = 20
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


def two_node_physical_twenty_memories_ketState_with_app_2(verbose=False):
    """Test with ket state formalism using RequestAppThroughput."""
    print('\nPhysical Entangled Pairs using Ket State with RequestApp:')
    
    log_filename = 'log/ketstate_entanglement_generation_app'
    network_config = 'config/line_2_physical_ketState.json'
    
    network_topo = RouterNetTopo(network_config)
    tl = network_topo.get_timeline()
    
    log.set_logger(__name__, tl, log_filename)
    log.set_logger_level('INFO')
    modules = ['timeline', 'network_manager', 'resource_manager', 'rule_manager', 
               'generation', 'purification', 'swapping', 'bsm']
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
    
    # Configure request parameters
    start_time = 1e12  # 1 second
    end_time = 10e12   # 10 seconds
    memory_size = 1   # Number of memories to use
    target_fidelity = 0.8
    
    # Set maximum entanglements to prevent runaway
    src_app.max_entanglements = 20
    
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
    
    # Fidelity
    fidelities = src_app.get_fidelity()
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
    
    # Also show traditional metrics from memory manager
    if verbose:
        latencies = []
        print(f"\n{src_node_name} memories:")
        print("{:5}  {:14}  {:8}  {:>7}".format("Index", "Entangled Node", "Fidelity", "Latency"))
        
        for info in src_node.resource_manager.memory_manager:
            latency = (info.entangle_time - start_time) * 1e-12
            if latency < 0:
                break
            latencies.append(latency)
            print("{:5}  {:>14}  {:8.5f}  {:.5f}".format(
                info.index, str(info.remote_node), float(info.fidelity), latency))
        
        if latencies:
            latency = np.average(latencies)
            print(f'Average latency = {latency:.4f}s; rate = {1/latency:.3f}/s')


def two_node_physical_twenty_memories_stabilizer_with_app_2(verbose=False):
    """Test with stabilizer formalism using RequestAppThroughput."""
    print('\nPhysical Entangled Pairs using Stabilizers with RequestApp:')
    
    # Set quantum manager to use stabilizer formalism
    QuantumManager.set_global_manager_formalism(STABILIZER_FORMALISM)
    
    # Set protocols to use stabilizer versions
    EntanglementGenerationA.set_global_type('barret_kok_stabilizer')
    EntanglementGenerationB.set_global_type('barret_kok_stabilizer')
    
    log_filename = 'log/stabilizer_entanglement_generation_app'
    network_config = 'config/line_2_physical_stabilizer.json'
    
    network_topo = RouterNetTopo(network_config)
    tl = network_topo.get_timeline()
    
    log.set_logger(__name__, tl, log_filename)
    log.set_logger_level('INFO')
    modules = ['timeline', 'network_manager', 'resource_manager', 'rule_manager', 
               'generation', 'purification', 'swapping', 'bsm']
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
    memory_size = 20   # Number of memories
    target_fidelity = 0.8
    
    # Set maximum entanglements
    src_app.max_entanglements = 20
    
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
    
    # Fidelity
    fidelities = src_app.get_fidelity()
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
    
    # Also show traditional metrics
    if verbose:
        latencies = []
        print(f"\n{src_node_name} memories:")
        print("{:5}  {:14}  {:8}  {:>7}".format("Index", "Entangled Node", "Fidelity", "Latency"))
        
        for info in src_node.resource_manager.memory_manager:
            latency = (info.entangle_time - start_time) * 1e-12
            if latency < 0:
                break
            latencies.append(latency)
            print("{:5}  {:>14}  {:8.5f}  {:.5f}".format(
                info.index, str(info.remote_node), float(info.fidelity), latency))
        
        if latencies:
            latency = np.average(latencies)
            print(f'Average latency = {latency:.4f}s; rate = {1/latency:.3f}/s')


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



    
  
    # # SKELETON SECTION FOR ENCODING
    # encoding_time = start_time + 5e11  # Schedule encoding after Bell pairs are ready
    
    # def apply_encoding():
    #     """Apply encoding to memories 7-13 on each node"""
    #     print("\nApplying [[7,1,3]] encoding...")
        
    #     qm = tl.quantum_manager
    #     src_array = src_node.components[src_node.memo_arr_name]
    #     # dest_array = dest_node.components[dest_node.memo_arr_name]
        
    #     # Get keys for memories 7-13 on each node
    #     src_encode_keys = [src_array[i].qstate_key for i in range(7, 14)]
    #     dest_encode_keys = [dest_array[i].qstate_key for i in range(7, 14)]
        
    #     # Create circuit and call encode_713 from qec_protocols
    #     circuit = stim.Circuit()
    #     circuit = encode_713_stim(circuit, src_encode_keys)
    #     circuit = encode_713_stim(circuit, dest_encode_keys)
        
    #     print("Encoding applied to memories 7-13 on both nodes")
    
    # process = Process(None, "encode", [])
    # process.func = apply_encoding
    # event = Event(encoding_time, process)
    # tl.schedule(event)
    # # END OF SKELETON SECTION
    # tl.init()
    # tl.run()

    # latencies = []
    # if verbose:
    #     print(src_node_name, "memories:")
    #     print("{:5}  {:14}  {:8}  {:>7}".format("Index", "Entangled Node", "Fidelity", "Latency"))
    # for info in src_node.resource_manager.memory_manager:
    #     latency = (info.entangle_time - start_time) * 1e-12
    #     if latency < 0:
    #         break
    #     latencies.append(latency)
    #     if verbose:
    #         print("{:5}  {:>14}  {:8.5f}  {:.5f}".format(info.index, str(info.remote_node), float(info.fidelity), latency))
    # latency = np.average(latencies)
    # print(f'average latency = {latency:.4f}s; rate = {1/latency:.3f}/s')


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

    network_topo = RouterNetTopo2G(network_config)
    tl = network_topo.get_timeline()
    
    log.set_logger(__name__, tl, log_filename)
    log.set_logger_level('INFO')
    modules = ['timeline', 'network_manager', 'resource_manager', 'rule_manager', 
               'generation', 'purification', 'swapping', 'bsm']
    for module in modules:
        log.track_module(module)
    
    # Set up applications on BOTH nodes
    apps = []
    src_node_name = 'router_0'
    dest_node_name = 'router_1'
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
            start_t=request_start,
            end_t=end_time,
            fidelity=target_fidelity,
            id=i
        )
    
    tl.init()
    tl.run()
    
    # Display results
    print("\n--- Results from RequestLogicalPairApp ---")

    # Get metrics from source
    src_metrics = src_app.get_metrics()

    # Display available metrics
    print(f"\nSource Node Metrics:")
    if src_metrics['bell_pair_generation_time']:
        print(f"  Bell pair generation time: {src_metrics['bell_pair_generation_time']:.4f}s")
    if src_metrics['encoding_time']:
        print(f"  Encoding time: {src_metrics['encoding_time']:.6f}s")
    if src_metrics['total_time']:
        print(f"  Total time: {src_metrics['total_time']:.4f}s")
    if src_metrics['bell_pair_fidelities']:
        print(f"  Bell pair fidelities: {[f'{f:.4f}' for f in src_metrics['bell_pair_fidelities']]}")
        print(f"  Average Bell fidelity: {src_metrics['average_bell_fidelity']:.4f}")
    print(f"  Encoding complete: {src_metrics['encoding_complete']}")

    # Destination metrics
    dest_metrics = dest_app.get_metrics()
    print(f"\nDestination Node Metrics:")
    print(f"  Encoding complete: {dest_metrics['encoding_complete']}")
    
    return src_app, dest_app

if __name__ == "__main__":


    # print("\n=== KET STATE 20 Physical Entagled Pairs===")
    # two_node_physical_twenty_memories_ketState_old(verbose=True)
    # print("\n=== STABILIZER 20 Physical Entagled Pairs===")
    # two_node_physical_twenty_memories_stabilizer_old(verbose=True)

    # print("\n=== STABILIZER 1 Logical Entagled Pair===")
    # two_node_physical_twenty_memories_stabilizer(verbose=True)

    # print("\n=== APP: KET STATE 20 Physical Entagled Pairs===")
    # two_node_physical_twenty_memories_ketState_with_app(verbose=False)
    # print("\n=== APP: STABILIZER 20 Physical Entagled Pairs===")
    # two_node_physical_twenty_memories_stabilizer_with_app(verbose=False)

    print("\n=== APP: KET STATE 20 Physical Entagled Pairs===")
    two_node_physical_twenty_memories_ketState_with_app_2(verbose=False)
    # print("\n=== APP: STABILIZER 20 Physical Entagled Pairs===")
    # two_node_physical_twenty_memories_stabilizer_with_app_2(verbose=False)

    # create_logical_bell_pair(verbose=False)

    # two_node_logical_pair_with_app(verbose=True)

    pass