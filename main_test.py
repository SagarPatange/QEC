import logging
import argparse
from collections import defaultdict
import numpy as np
from sequence.topology.router_net_topo import RouterNetTopo
from sequence.constants import MILLISECOND
import sequence.utils.log as log
from qec_protocols import encode_713_stim, decode_713_stim
from sequence.kernel.event import Event
from sequence.kernel.process import Process
import stim
from sequence.kernel.quantum_manager import QuantumManager
from sequence.constants import STABILIZER_FORMALISM
from sequence.entanglement_management.generation import EntanglementGenerationA, EntanglementGenerationB

from request_app import RequestAppThroughput


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
    """Test with ket state formalism using RequestAppThroughput.
    
    Generates entangled pairs over 10 seconds and measures throughput.
    """
    print('\nPhysical Entangled Pairs using Ket State with RequestApp:')
    
    # No need to set formalism - ket state is default
    log_filename = 'log/ketstate_entanglement_generation_app'
    network_config = 'config/line_2_physical_ketState.json'
    
    network_topo = RouterNetTopo(network_config)
    tl = network_topo.get_timeline()
    
    log.set_logger(__name__, tl, log_filename)
    log.set_logger_level('INFO')
    modules = ['timeline', 'network_manager', 'resource_manager', 'rule_manager', 
               'generation', 'purification', 'swapping', 'bsm', 'request_app']
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
    memory_size = 20  # Number of memories
    target_fidelity = 0.8
    
    # Start the request through the app
    src_app.start(dest_node_name, start_time, end_time, memory_size, target_fidelity)
    
    tl.init()
    tl.run()
    
    # Display results
    print("\n--- Results from RequestApp ---")
    
    # Time to service
    time_to_service = src_app.get_time_to_service()
    if time_to_service:
        print(f"Time to service samples (first 10): {time_to_service[:10]}")
        print(f"Average time to service: {np.mean(time_to_service):.3f} ns")
    
    # Fidelity
    fidelities = src_app.get_fidelity()
    if fidelities:
        print(f"Fidelity samples (first 10): {fidelities[:10]}")
        print(f"Average fidelity: {np.mean(fidelities):.5f}")
    
    # Throughput
    throughput_dict = src_app.get_request_to_throughput()
    for reservation, throughput in throughput_dict.items():
        print(f"Throughput: {throughput:.2f} entanglements/second")
    
    # Also show traditional metrics
    latencies = []
    if verbose:
        print(f"\n{src_node_name} memories:")
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


def two_node_physical_twenty_memories_stabilizer_with_app(verbose=False):
    """Test with stabilizer formalism using RequestAppThroughput.
    
    Generates entangled pairs over 10 seconds using stabilizer formalism.
    """
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
               'generation', 'purification', 'swapping', 'bsm', 'request_app']
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
    memory_size = 20  # Number of memories
    target_fidelity = 0.8
    
    # Start the request through the app
    src_app.start(dest_node_name, start_time, end_time, memory_size, target_fidelity)
    
    tl.init()
    tl.run()
    
    # Display results
    print("\n--- Results from RequestApp ---")
    
    # Time to service
    time_to_service = src_app.get_time_to_service()
    if time_to_service:
        print(f"Time to service samples (first 10): {time_to_service[:10]}")
        print(f"Average time to service: {np.mean(time_to_service):.3f} ns")
    
    # Fidelity
    fidelities = src_app.get_fidelity()
    if fidelities:
        print(f"Fidelity samples (first 10): {fidelities[:10]}")
        print(f"Average fidelity: {np.mean(fidelities):.5f}")
    
    # Throughput
    throughput_dict = src_app.get_request_to_throughput()
    for reservation, throughput in throughput_dict.items():
        print(f"Throughput: {throughput:.2f} entanglements/second")
    
    # Also show traditional metrics
    latencies = []
    if verbose:
        print(f"\n{src_node_name} memories:")
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



def create_logical_bell_pair(verbose=True):
    print('Physical Entangled Pairs using Stabilizers:')
    
    QuantumManager.set_global_manager_formalism(STABILIZER_FORMALISM)
    EntanglementGenerationA.set_global_type('barret_kok_stabilizer')
    EntanglementGenerationB.set_global_type('barret_kok_stabilizer')
    
    log_filename = 'log/stabilizer_entanglement_generation'
    network_config = 'config/line_2_physical_stabilizer.json'
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
    entanglement_number = 7  
    nm = src_node.network_manager
    nm.request(dest_node_name, start_time=start_time, end_time=end_time, memory_size=entanglement_number, target_fidelity=0.8)

    # SKELETON SECTION FOR ENCODING
    encoding_time = start_time + 5e11  # Schedule encoding after Bell pairs are ready
    
    def apply_encoding():
        """Apply encoding to memories 7-13 on each node"""
        print("\nApplying [[7,1,3]] encoding...")
        
        qm = tl.quantum_manager
        src_array = src_node.components[src_node.memo_arr_name]
        # dest_array = dest_node.components[dest_node.memo_arr_name]
        
        # Get keys for memories 7-13 on each node
        src_encode_keys = [src_array[i].qstate_key for i in range(7, 14)]
        dest_encode_keys = [dest_array[i].qstate_key for i in range(7, 14)]
        
        # Create circuit and call encode_713 from qec_protocols
        circuit = stim.Circuit()
        circuit = encode_713_stim(circuit, src_encode_keys)
        circuit = encode_713_stim(circuit, dest_encode_keys)
        
        print("Encoding applied to memories 7-13 on both nodes")
    
    process = Process(None, "encode", [])
    process.func = apply_encoding
    event = Event(encoding_time, process)
    tl.schedule(event)
    # END OF SKELETON SECTION
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


if __name__ == "__main__":


    print("\n=== KET STATE 20 Physical Entagled Pairs===")
    two_node_physical_twenty_memories_ketState_old(verbose=True)
    print("\n=== STABILIZER 20 Physical Entagled Pairs===")
    two_node_physical_twenty_memories_stabilizer_old(verbose=True)

    # print("\n=== STABILIZER 1 Logical Entagled Pair===")
    # two_node_physical_twenty_memories_stabilizer(verbose=True)

    # print("\n=== APP: KET STATE 20 Physical Entagled Pairs===")
    # two_node_physical_twenty_memories_ketState_with_app(verbose=True)
    # print("\n=== APP: STABILIZER 20 Physical Entagled Pairs===")
    # two_node_physical_twenty_memories_stabilizer_with_app(verbose=True)