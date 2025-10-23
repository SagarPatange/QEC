"""
Test script for Request Logical Pair Application - Alice's results only.

This script demonstrates how to use the RequestLogicalPairApp to:
1. Generate 7 Bell pairs using Barrett-Kok entanglement
2. Encode Alice's data qubits to logical |0⟩_L
3. Encode Bob's data qubits to logical |+⟩_L

Uses stabilizer formalism and 2nd generation quantum routers.
Only displays Alice's fidelity calculations (not Bob's duplicate calculations).
"""

import numpy as np
from sequence.kernel.quantum_manager import QuantumManager
from sequence.constants import STABILIZER_FORMALISM
from sequence.topology.router_net_topo import RouterNetTopo2G, RouterNetTopo
from sequence.entanglement_management.generation import EntanglementGenerationA, EntanglementGenerationB
from RequestLogicalPairApp import RequestLogicalPairApp
from sequence.utils import log


def test_request_logical_pair_with_qec(verbose=True):
    """Test Bell pair generation + QEC encoding with stabilizer formalism.
    
    Creates two app instances (Alice and Bob), generates 7 Bell pairs,
    then encodes Alice's data qubits to |0⟩_L and Bob's to |+⟩_L.
    
    Args:
        verbose: Whether to print detailed output
        
    Returns:
        Tuple of (alice_app, bob_app, alice_results, bob_results)
    """
    # Set up Stabilizer Formalism
    QuantumManager.set_global_manager_formalism(STABILIZER_FORMALISM)
    from sequence.constants import BARRET_KOK
    EntanglementGenerationA.set_global_type(BARRET_KOK)
    EntanglementGenerationB.set_global_type(BARRET_KOK)

    # Load Network Configuration
    network_config = 'config/line_2_2nd_gen_stabilizer.json'
    network_topo = RouterNetTopo2G(network_config)
    tl = network_topo.get_timeline()

    # Set up Logging
    log_filename = 'log/request_logical_pair_qec'
    log.set_logger(__name__, tl, log_filename)
    log.set_logger_level('INFO')

    # Track modules that generate logs
    log.track_module('bsm')
    log.track_module('RequestLogicalPairApp')

    # Find Network Nodes
    routers = network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER)
    alice = routers[0]  # router_0
    bob = routers[1]    # router_1

    # Create TWO app instances - one for each node
    alice_app = RequestLogicalPairApp(alice)
    bob_app = RequestLogicalPairApp(bob)

    # Simulation parameters
    start_time = 1e12    # 1 millisecond
    end_time = 10e12     # 10 milliseconds
    target_fidelity = 0.8
    memory_size = 7


    # Only Alice calls start() - Bob reacts automatically via get_other_reservation()
    alice_app.start(
        remote_node_name=bob.name,
        start_time=start_time,
        end_time=end_time,
        memory_size=memory_size,
        target_fidelity=target_fidelity,
        logical_state='0',                # Alice → |0⟩_L
        encoding_enabled=True
    )

    # Run Simulation
    tl.init()
    tl.run()

    # Get results from both nodes
    alice_results = alice_app.get_results()
    bob_results = bob_app.get_results()
    
    # ONLY PRINT ALICE'S RESULTS (no Bob)
    print("\n" + "="*70)
    print("ALICE - Bell Pair Generation Results")
    print("="*70)
    print(f"Role: Initiator")
    print(f"Logical state target: |{alice_results['encoding']['logical_state']}>_L")
    print(f"Encoding success: {alice_results['encoding']['success']}")
    if alice_results['encoding']['encoding_time']:
        print(f"Encoding time: {alice_results['encoding']['encoding_time']:.6f}s")
    print(f"Bell pairs generated: {len(alice_results['bell_pairs'])}")

    # Show individual Bell pairs AND their fidelities
    if alice_results['bell_pairs']:
        print(f"\nBell Pair Details:")
        for bp in alice_results['bell_pairs']:
            fidelity_str = f"{bp['fidelity']:.6f}" if bp['fidelity'] is not None else "None (error)"
            print(f"  Pair {bp['pair_id']}: Memory {bp['memory_index']}, Fidelity = {fidelity_str}")
        
        # Filter out None fidelities for statistics
        fidelities = [bp['fidelity'] for bp in alice_results['bell_pairs'] if bp['fidelity'] is not None]
        if fidelities:
            print(f"\nFidelity Statistics:")
            print(f"  Average: {np.mean(fidelities):.6f}")
            print(f"  Min: {np.min(fidelities):.6f}")
            print(f"  Max: {np.max(fidelities):.6f}")
            print(f"  Std Dev: {np.std(fidelities):.6f}")
        else:
            print(f"\nWARNING: No fidelities calculated (all None)")
    
    print("="*70)
    
    # Note: Bob's results are available but not printed
    print(f"\nNote: Bob also generated {len(bob_results['bell_pairs'])} Bell pairs (same physical pairs)")
    print(f"      Bob's fidelity calculations are not shown (would be duplicates)")
    
    return alice_app, bob_app, alice_results, bob_results


def test_request_logical_pair_simple():
    """Simplified test for quick verification."""
    print("\nRunning simplified Request Logical Pair test with QEC...\n")
    
    # Set formalism
    QuantumManager.set_global_manager_formalism(STABILIZER_FORMALISM)
    from sequence.constants import BARRET_KOK
    EntanglementGenerationA.set_global_type(BARRET_KOK)
    EntanglementGenerationB.set_global_type(BARRET_KOK)
    
    # Load network
    network_config = 'config/line_2_2nd_gen_stabilizer.json'
    network_topo = RouterNetTopo2G(network_config)
    tl = network_topo.get_timeline()
    
    # Find nodes
    routers = network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER)
    alice = routers[0]
    bob = routers[1]
    
    # Create and run apps
    alice_app = RequestLogicalPairApp(alice)
    bob_app = RequestLogicalPairApp(bob)

    remote_node_name=bob.name
    start_time=1e12
    end_time=10e12
    memory_size=7
    target_fidelity=0.8
    logical_state='0'
    encoding_enabled=True
    
    alice_app.start(
        remote_node_name=remote_node_name,
        start_time=start_time,
        end_time=end_time,
        memory_size=memory_size,
        target_fidelity=target_fidelity,
        logical_state=logical_state,
        encoding_enabled=encoding_enabled
    )
    
    tl.init()
    tl.run()
    
    # Return only Alice's results
    return alice_app.get_results()


if __name__ == '__main__':
    # Run the comprehensive test
    alice_app, bob_app, alice_results, bob_results = test_request_logical_pair_with_qec()
    
    print("\nTest completed successfully!")