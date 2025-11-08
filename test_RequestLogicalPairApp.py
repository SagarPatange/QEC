"""
Test script for Request Logical Pair Application - Alice's results only.

This script demonstrates how to use the RequestLogicalPairApp to:
1. Generate 7 Bell pairs using Barrett-Kok entanglement
2. Encode Alice's data qubits to logical |0>_L
3. Encode Bob's data qubits to logical |+>_L

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
from sequence.constants import BARRET_KOK



def test_request_logical_pair_with_qec(verbose=True, enable_teleported_cnot=False):    
    """Test Bell pair generation + QEC encoding with stabilizer formalism.
    
    Creates two app instances (Alice and Bob), generates 7 Bell pairs,
    then encodes Alice's data qubits to |0>_L and Bob's to |+>_L.
    
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

    # Set verbose flags - only Alice prints detailed results
    alice_app.verbose = True
    bob_app.verbose = False

    # Simulation parameters
    start_time = 1e12    # 1 millisecond
    end_time = 10e12     # 10 milliseconds
    target_fidelity = 0.8
    memory_size = 7


    # Only Alice calls start() - Bob reacts automatically via get_other_reservation()
    alice_app.start(
        remote_node_name=bob.name,
        remote_node=bob,  
        start_time=1e12,
        end_time=10e12,
        memory_size=7,
        target_fidelity=0.8,
        logical_state='+',
        encoding_enabled=True,
        teleported_cnot_enabled=True,
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
            print(f"\nPhysical Bell Pair Fidelity Statistics:")
            print(f"  Average: {np.mean(fidelities):.6f}")
            print(f"  Min: {np.min(fidelities):.6f}")
            print(f"  Max: {np.max(fidelities):.6f}")
            print(f"  Std Dev: {np.std(fidelities):.6f}")
        else:
            print(f"\nWARNING: No fidelities calculated (all None)")
    
    # Show logical Bell pair fidelity if teleported CNOT was enabled
    if enable_teleported_cnot and 'logical_bell_pair' in alice_results:
        logical_data = alice_results['logical_bell_pair']
        print(f"\nLogical Bell Pair Fidelity (Post Teleported-CNOT):")
        print(f"  Fidelity: {logical_data['fidelity']:.6f}")
        print(f"  <XX> = {logical_data['correlations']['XX']:+.6f}")
        print(f"  <YY> = {logical_data['correlations']['YY']:+.6f}")
        print(f"  <ZZ> = {logical_data['correlations']['ZZ']:+.6f}")
        
        # Compare to physical fidelities
        if fidelities:
            avg_physical = np.mean(fidelities)
            improvement = logical_data['fidelity'] - avg_physical
            print(f"\n  Comparison:")
            print(f"    Average physical fidelity: {avg_physical:.6f}")
            print(f"    Logical fidelity improvement: {improvement:+.6f}")
    
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

    # Set verbose flags
    alice_app.verbose = True
    bob_app.verbose = False

    remote_node_name=bob.name
    start_time=1e12
    end_time=10e12
    memory_size=7
    target_fidelity=0.8
    logical_state='0'
    encoding_enabled=True
    
    alice_app.start(
        remote_node_name=bob.name,
        remote_node=bob,  # THIS LINE MUST BE ADDED
        start_time=1e12,
        end_time=10e12,
        memory_size=7,
        target_fidelity=0.8,
        logical_state='+',
        encoding_enabled=True,
        teleported_cnot_enabled=True,
    )
    
    tl.init()
    tl.run()
    
    # Return only Alice's results
    return alice_app.get_results()


def test_with_teleported_cnot():
    """Test complete protocol stack: Bell pairs -> Encoding -> Teleported CNOT."""
    # Set formalism
    QuantumManager.set_global_manager_formalism(STABILIZER_FORMALISM)
    from sequence.constants import BARRET_KOK
    EntanglementGenerationA.set_global_type(BARRET_KOK)
    EntanglementGenerationB.set_global_type(BARRET_KOK)
    
    # Load network
    network_config = 'config/line_2_2nd_gen_stabilizer.json'
    network_topo = RouterNetTopo2G(network_config)
    tl = network_topo.get_timeline()
    
    # Set up logging
    log_filename = 'log/request_logical_pair_tcnot'
    log.set_logger(__name__, tl, log_filename)
    log.set_logger_level('INFO')
    log.track_module('RequestLogicalPairApp')
    log.track_module('teleported_cnot')
    
    # Find nodes
    routers = network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER)
    alice = routers[0]
    bob = routers[1]

    # Debug: Check components
    print(f"DEBUG - Alice components: {list(alice.components.keys())}")
    print(f"DEBUG - Bob components: {list(bob.components.keys())}")

    # Create apps
    alice_app = RequestLogicalPairApp(alice)
    bob_app = RequestLogicalPairApp(bob)

    # Disable verbose output
    alice_app.verbose = False
    bob_app.verbose = False
    
    # Start with teleported CNOT enabled
    alice_app.start(
        remote_node_name=bob.name,
        remote_node=bob,
        start_time=1e12,
        end_time=10e12,
        memory_size=7,
        target_fidelity=0.8,
        logical_state='+',
        encoding_enabled=True,
        teleported_cnot_enabled=True,
        depolarization_enabled=True,
        coherence_time_ms=5000,  # Try different values: 50, 100, 500, 1000, 5000
    )
    
    # Run simulation
    tl.init()
    tl.run()
    
    # Get results
    alice_results = alice_app.get_results()
    bob_results = bob_app.get_results()

    # Print minimal results
    def print_minimal_results(alice_results, bob_results):
        """Print clean, minimal test results."""
        alice_fids = [bp['fidelity'] for bp in alice_results['bell_pairs'] if bp['fidelity'] is not None]
        bob_fids = [bp['fidelity'] for bp in bob_results['bell_pairs'] if bp['fidelity'] is not None]
        product_fid = alice_results.get('product_state_fidelity', {})
        alice_x = product_fid.get('alice_x_prob')
        bob_z = product_fid.get('bob_z_prob')
        logical = alice_results.get('logical_bell_pair', {})

        print("\nTEST 2: Teleported CNOT Protocol")
        print("=" * 60)
        print("\nPHYSICAL BELL PAIRS:")
        print(f"Alice: {alice_fids}  Avg: {np.mean(alice_fids):.3f}")
        print(f"Bob:   {bob_fids}  Avg: {np.mean(bob_fids):.3f}")

        if alice_x is not None and bob_z is not None:
            print("\nLOGICAL STATES (Before Teleported CNOT):")
            print(f"Alice |+>_L: P(X_L=+1) = {alice_x:.3f}")
            print(f"Bob   |0>_L: P(Z_L=+1) = {bob_z:.3f}")

        if logical.get('fidelity') is not None:
            print("\nLOGICAL BELL PAIR (After Teleported CNOT):")
            print(f"<XX> = {logical['correlations']['XX']:+.3f}")
            print(f"<YY> = {logical['correlations']['YY']:+.3f}")
            print(f"<ZZ> = {logical['correlations']['ZZ']:+.3f}")
            print(f"Fidelity: {logical['fidelity']:.3f}")

        print("\n[PASS] TEST 2")
        print("=" * 60)

    print_minimal_results(alice_results, bob_results)

    return alice_app, bob_app, alice_results, bob_results


if __name__ == '__main__':
    # print("\n" + "="*70)
    # print("RUNNING AUTOMATED TEST SUITE")
    # print("="*70)
    
    # # TEST 1: Bell Pairs + Encoding (no teleported CNOT)
    # print("\n" + "="*70)
    # print("TEST 1: Bell Pair Generation + QEC Encoding (No Teleported CNOT)")
    # print("="*70)
    # alice_app_1, bob_app_1, alice_results_1, bob_results_1 = test_request_logical_pair_with_qec(
    #     verbose=True, 
    #     enable_teleported_cnot=False
    # )
    # print("\n[PASS] TEST 1 COMPLETED\n")

    # TEST 2: Bell Pairs + Encoding + Teleported CNOT
    print("\n" + "="*70)
    print("TEST 2: Bell Pair Generation + QEC Encoding + Teleported CNOT")
    print("="*70)
    alice_app_2, bob_app_2, alice_results_2, bob_results_2 = test_with_teleported_cnot()
    print("\n[PASS] TEST 2 COMPLETED\n")
    
    # # Final summary
    # print("\n" + "="*70)
    # print("ALL TESTS COMPLETED SUCCESSFULLY!")
    # print("="*70)
    # print(f"Test 1 (Encoding only): {len(alice_results_1['bell_pairs'])} Bell pairs generated")
    # print(f"Test 2 (Encoding + TCNOT): {len(alice_results_2['bell_pairs'])} Bell pairs generated")
    
    # Show logical fidelity summary if available
    if 'logical_bell_pair' in alice_results_2:
        logical_fid = alice_results_2['logical_bell_pair']['fidelity']
        print(f"Test 2 Logical Bell Pair Fidelity: {logical_fid:.6f}")
    
    print("="*70 + "\n")