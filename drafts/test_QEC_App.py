"""test_QEC_App.py
Test file for QEC Application with QuantumRouter2ndGeneration nodes.
Tests encoding, syndrome measurement, error correction, and logical operations.

This test file uses RouterNetTopo2G to load configs with "generation": 2 flag,
which creates QuantumRouter2ndGeneration nodes with:
- Communication memories (memo_arr) for distributed operations
- Data memories (data_memo_arr) for logical qubits  
- Ancilla memories (ancilla_memo_arr) for syndrome measurements

To generate the required config file:
python config_generator_line.py 2 20 1 0.0002 1 -d config -o line_2_gen2.json -s 10 --gen2
"""

import itertools
import numpy as np
import pytest
from sequence.topology.router_net_topo import RouterNetTopo2G
from drafts.QEC_app import QECApp
from sequence.constants import MILLISECOND, MICROSECOND
from sequence.kernel.quantum_utils import verify_same_state_vector
import sequence.utils.log as log  # ADD LOGGING


def single_qec_trial(psi, seeds: dict = None):
    """Run a single QEC trial with the given quantum state.

    Similar to single_trial in test_teleport.py but for QEC.

    Args:
        psi: The quantum state to protect
        seeds: Dictionary of random seeds for reproducibility

    Returns:
        The decoded state after error correction
    """
    # Load network topology (use 2G router network topology)
    topo = RouterNetTopo2G("config/line_2_logical_stabilizer_2G.json")
    tl = topo.get_timeline()
    
    # SET UP LOGGING
    log_filename = 'log/qec_single_trial'
    log.set_logger(__name__, tl, log_filename)
    log.set_logger_level('DEBUG')  # Use DEBUG for detailed info
    modules = ['timeline', 'qec_app', 'qec_protocol', 'network_manager', 
               'resource_manager', 'memory']
    for module in modules:
        log.track_module(module)
    
    # Extract nodes - get QuantumRouter2ndGeneration nodes
    routers = topo.get_nodes_by_type(RouterNetTopo2G.QUANTUM_ROUTER)
    alice = routers[0]
    bob = routers[1] if len(routers) > 1 else None
    
    # Set seeds for reproducibility
    alice.set_seed(seeds["alice"])
    if bob:
        bob.set_seed(seeds["bob"])
    
    # Prepare the state in Alice's data memory
    if hasattr(alice, 'data_memo_arr_name'):
        data_memo_arr = alice.get_component_by_name(alice.data_memo_arr_name)
    else:
        data_memo_arr = alice.get_component_by_name(alice.memo_arr_name)
    
    # Place initial state in first memory
    data_memo_arr[0].update_state(psi)
    
    # Create QEC app
    qec_app = QECApp(alice)
    
    # Start QEC protection
    logical_id = qec_app.start(
        physical_indices=list(range(7)),
        initial_state=psi,
        syndrome_interval=10 * MICROSECOND
    )
    
    # Run simulation to let syndrome measurements happen
    tl.init()
    tl.run_until(50 * MILLISECOND)
    
    # Stop QEC and decode
    decoded_state = qec_app.stop(logical_id)
    
    return np.array(decoded_state)


def test_basic_encoding_decoding():
    """Test that encoding and immediate decoding preserves the state."""
    
    # Set up network
    topo = RouterNetTopo2G("config/line_2_logical_stabilizer_2G.json")
    tl = topo.get_timeline()
    
    # SET UP LOGGING
    log_filename = 'log/qec_basic_encoding_decoding'
    log.set_logger(__name__, tl, log_filename)
    log.set_logger_level('INFO')
    modules = ['timeline', 'qec_app', 'qec_protocol', 'memory']
    for module in modules:
        log.track_module(module)
    
    routers = topo.get_nodes_by_type(RouterNetTopo2G.QUANTUM_ROUTER)
    alice = routers[0]
    
    # Fixed seed for reproducibility
    alice.set_seed(12345)
    
    # Test state |0⟩
    qec_app = QECApp(alice)
    
    # Start QEC with |0⟩ state
    logical_id = qec_app.start(
        physical_indices=None,  # Auto-allocate
        initial_state=np.array([1, 0]),
        syndrome_interval=100 * MILLISECOND  # Long interval so no measurements
    )
    
    # Immediately decode without running simulation
    tl.init()
    decoded_state = qec_app.stop(logical_id)
    
    # Verify state is preserved
    expected = np.array([1, 0])
    assert verify_same_state_vector(decoded_state, expected), \
        f"Decoded state {decoded_state} != expected {expected}"
    
    print("[PASS] Basic encoding/decoding test passed")


def test_logical_gates():
    """Test logical gate operations."""
    
    topo = RouterNetTopo2G("config/line_2_logical_stabilizer_2G.json")
    tl = topo.get_timeline()
    
    # SET UP LOGGING
    log_filename = 'log/qec_logical_gates'
    log.set_logger(__name__, tl, log_filename)
    log.set_logger_level('INFO')
    modules = ['timeline', 'qec_app', 'qec_protocol']
    for module in modules:
        log.track_module(module)
    
    routers = topo.get_nodes_by_type(RouterNetTopo2G.QUANTUM_ROUTER)
    alice = routers[0]
    alice.set_seed(67890)
    
    qec_app = QECApp(alice)
    
    # Create logical qubit in |0⟩
    logical_id = qec_app.start(
        physical_indices=None,  # Auto-allocate
        initial_state=np.array([1, 0]),
        syndrome_interval=100 * MILLISECOND
    )
    
    # Apply logical X (should flip to |1⟩)
    qec_app.perform_logical_gate(logical_id, 'X')
    
    tl.init()
    decoded_state = qec_app.stop(logical_id)
    
    expected = np.array([0, 1])  # |1⟩ state
    assert verify_same_state_vector(decoded_state, expected), \
        f"After X gate: {decoded_state} != expected {expected}"
    
    print("[PASS] Logical X gate test passed")
    
    # Test logical Z on |+⟩ state
    qec_app2 = QECApp(alice)
    plus_state = np.array([1, 1]) / np.sqrt(2)
    
    logical_id2 = qec_app2.start(
        physical_indices=list(range(7, 14)) if not qec_app2.is_2nd_gen else None,
        initial_state=plus_state,
        syndrome_interval=100 * MILLISECOND
    )
    
    # Apply logical Z (should flip to |−⟩)
    qec_app2.perform_logical_gate(logical_id2, 'Z')
    
    decoded_state2 = qec_app2.stop(logical_id2)
    
    minus_state = np.array([1, -1]) / np.sqrt(2)
    assert verify_same_state_vector(decoded_state2, minus_state), \
        f"After Z gate: {decoded_state2} != expected {minus_state}"
    
    print("[PASS] Logical Z gate test passed")


def test_syndrome_measurements():
    """Test that syndrome measurements are happening."""
    
    topo = RouterNetTopo2G("config/line_2_logical_stabilizer_2G.json")
    tl = topo.get_timeline()
    
    # SET UP LOGGING - Use DEBUG to see syndrome measurements
    log_filename = 'log/qec_syndrome_measurements'
    log.set_logger(__name__, tl, log_filename)
    log.set_logger_level('DEBUG')  # DEBUG level to see all syndrome measurements
    modules = ['timeline', 'qec_app', 'qec_protocol']
    for module in modules:
        log.track_module(module)
    
    routers = topo.get_nodes_by_type(RouterNetTopo2G.QUANTUM_ROUTER)
    alice = routers[0]
    alice.set_seed(11111)
    
    qec_app = QECApp(alice)
    
    # Start QEC with short syndrome interval
    logical_id = qec_app.start(
        physical_indices=None,  # Auto-allocate
        initial_state=np.array([1, 0]),
        syndrome_interval=5 * MICROSECOND  # Short interval
    )
    
    # Run for enough time to get multiple measurements
    tl.init()
    tl.run_until(50 * MICROSECOND)
    
    # Check that syndrome measurements happened
    history = qec_app.get_syndrome_history(logical_id)
    assert len(history) > 0, "No syndrome measurements recorded"
    
    # Should have roughly 10 measurements (50μs / 5μs)
    assert len(history) >= 8, f"Too few measurements: {len(history)}"
    
    print(f"[PASS] Syndrome measurement test passed ({len(history)} measurements)")
    
    # Clean up
    qec_app.stop(logical_id)


def test_concurrent_logical_qubits():
    """Test multiple logical qubits simultaneously."""
    
    topo = RouterNetTopo2G("config/line_2_logical_stabilizer_2G.json")
    tl = topo.get_timeline()
    
    # SET UP LOGGING
    log_filename = 'log/qec_concurrent_qubits'
    log.set_logger(__name__, tl, log_filename)
    log.set_logger_level('INFO')
    modules = ['timeline', 'qec_app', 'qec_protocol', 'memory']
    for module in modules:
        log.track_module(module)
    
    routers = topo.get_nodes_by_type(RouterNetTopo2G.QUANTUM_ROUTER)
    alice = routers[0]
    alice.set_seed(33333)
    
    # Prepare two different states
    psi1 = np.array([1, 0])              # |0⟩
    psi2 = np.array([1, 1]) / np.sqrt(2) # |+⟩
    
    # Create single QEC app managing multiple logical qubits
    qec_app = QECApp(alice)
    
    # If we have a 2nd gen router, use its special method
    if qec_app.is_2nd_gen:
        logical_id1 = qec_app.start_with_2nd_gen(
            initial_state=psi1,
            syndrome_interval=10 * MICROSECOND
        )
        
        logical_id2 = qec_app.start(
            physical_indices=list(range(7)),  # Use comm memories
            initial_state=psi2,
            syndrome_interval=10 * MICROSECOND,
            use_data_memories=False  # Force comm memories
        )
    else:
        logical_id1 = qec_app.start(
            physical_indices=list(range(7)),
            initial_state=psi1,
            syndrome_interval=10 * MICROSECOND
        )
        
        logical_id2 = qec_app.start(
            physical_indices=list(range(7, 14)),
            initial_state=psi2,
            syndrome_interval=10 * MICROSECOND
        )
    
    # Run simulation
    tl.init()
    tl.run_until(30 * MICROSECOND)
    
    # Check both logical qubits
    decoded1 = qec_app.stop(logical_id1)
    decoded2 = qec_app.stop(logical_id2)
    
    assert verify_same_state_vector(decoded1, psi1), \
        f"Logical 1 failed: {decoded1} != {psi1}"
    assert verify_same_state_vector(decoded2, psi2), \
        f"Logical 2 failed: {decoded2} != {psi2}"
    
    print("[PASS] Concurrent logical qubits test passed")


def test_2nd_gen_router_features():
    """Test QEC with QuantumRouter2ndGeneration's specialized memory arrays."""
    
    topo = RouterNetTopo2G("config/line_2_logical_stabilizer_2G.json")
    tl = topo.get_timeline()
    
    # SET UP LOGGING
    log_filename = 'log/qec_2nd_gen_features'
    log.set_logger(__name__, tl, log_filename)
    log.set_logger_level('DEBUG')  # DEBUG to see memory array details
    modules = ['timeline', 'qec_app', 'qec_protocol', 'memory']
    for module in modules:
        log.track_module(module)
    
    routers = topo.get_nodes_by_type(RouterNetTopo2G.QUANTUM_ROUTER)
    alice = routers[0]
    alice.set_seed(55555)
    
    qec_app = QECApp(alice)
    
    # Check if we have a 2nd gen router
    if qec_app.is_2nd_gen:
        print("[PASS] Detected QuantumRouter2ndGeneration")
        
        # Test using the 2nd gen specific method
        test_state = np.array([1, 1]) / np.sqrt(2)  # |+⟩
        
        logical_id = qec_app.start_with_2nd_gen(
            initial_state=test_state,
            syndrome_interval=10 * MICROSECOND
        )
        
        assert logical_id is not None, "Failed to create logical qubit with 2nd gen method"
        
        # Check memory arrays are being used correctly
        memory_arrays = qec_app.get_memory_arrays()
        assert 'data' in memory_arrays, "Data memory array not found"
        assert 'ancilla' in memory_arrays, "Ancilla memory array not found"
        print("[PASS] All memory arrays (comm, data, ancilla) available")
        
        # Run and verify
        tl.init()
        tl.run_until(30 * MICROSECOND)
        
        decoded = qec_app.stop(logical_id)
        assert verify_same_state_vector(decoded, test_state), \
            f"2nd gen QEC failed: {decoded} != {test_state}"
        
        print("[PASS] 2nd generation router features test passed")
    else:
        print("⚠ Standard QuantumRouter detected - skipping 2nd gen tests")
        
        logical_id = qec_app.start(
            physical_indices=None,
            initial_state=np.array([1, 0]),
            use_data_memories=False
        )
        
        assert logical_id is not None, "Failed with standard router"
        qec_app.stop(logical_id)
        print("[PASS] Standard router fallback works")


def test_qec_with_errors():
    """Test QEC with injected errors (placeholder for future)."""
    
    topo = RouterNetTopo2G("config/line_2_logical_stabilizer_2G.json")
    tl = topo.get_timeline()
    
    # SET UP LOGGING
    log_filename = 'log/qec_with_errors'
    log.set_logger(__name__, tl, log_filename)
    log.set_logger_level('DEBUG')
    modules = ['timeline', 'qec_app', 'qec_protocol']
    for module in modules:
        log.track_module(module)
    
    routers = topo.get_nodes_by_type(RouterNetTopo2G.QUANTUM_ROUTER)
    alice = routers[0]
    alice.set_seed(44444)
    
    qec_app = QECApp(alice)
    
    logical_id = qec_app.start(
        physical_indices=None,
        initial_state=np.array([1, 0]),
        syndrome_interval=5 * MICROSECOND
    )
    
    # TODO: Inject an error here
    # qm = alice.timeline.quantum_manager
    # qm.run_gate(qm.get(2), "X")  # Bit flip on qubit 2
    
    tl.init()
    tl.run_until(20 * MICROSECOND)
    
    decoded = qec_app.stop(logical_id)
    expected = np.array([1, 0])
    
    assert decoded is not None, "Decoding failed"
    
    print("[PASS] Error injection test passed (placeholder)")


# Keep existing helper functions
def _random_state(rng: np.random.Generator):
    """Generate a random single-qubit pure state."""
    a = rng.normal() + 1j * rng.normal()
    b = rng.normal() + 1j * rng.normal()
    v = np.array([a, b], dtype=complex)
    v = v / np.linalg.norm(v)
    return v


_rng = np.random.default_rng(42)
_test_states = [
    np.array([1, 0]),
    np.array([0, 1]),
    np.array([1, 1]) / np.sqrt(2),
    np.array([1, -1]) / np.sqrt(2),
    _random_state(_rng),
    _random_state(_rng)
]

_test_seeds = [
    {"alice": 12345, "bob": 67890},
    {"alice": 11111, "bob": 22222},
    {"alice": 99999, "bob": 88888}
]


@pytest.mark.parametrize(
    "psi,seeds",
    list(itertools.product(_test_states[:3], _test_seeds[:2]))
)
def test_qec_preserves_state(psi, seeds):
    """Test that QEC preserves quantum states correctly."""
    decoded = single_qec_trial(psi, seeds)
    
    assert verify_same_state_vector(decoded, psi), \
        f"QEC failed: decoded {decoded} != original {psi}"


def run_single_test(test_func, test_name):
    """Run a single test function with isolated execution."""
    try:
        test_func()
        print(f"[PASS] {test_name} passed")
        return True
    except Exception as e:
        print(f"[FAIL] {test_name} failed: {e}")
        import traceback
        traceback.print_exc()  # Print full traceback for debugging
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Running QEC Application Tests with Logging")
    print("=" * 60)
    print("Log files will be saved to log/ directory")
    print("=" * 60)

    # Run non-parametrized tests individually
    tests = [
        (test_basic_encoding_decoding, "Basic encoding/decoding"),
        (test_logical_gates, "Logical gates"),
        (test_syndrome_measurements, "Syndrome measurements"),
        (test_concurrent_logical_qubits, "Concurrent logical qubits"),
        (test_2nd_gen_router_features, "2nd generation router features"),
        (test_qec_with_errors, "Error handling"),
    ]

    print("\n1. Individual Tests:")
    passed_tests = 0
    for test_func, test_name in tests:
        if run_single_test(test_func, test_name):
            passed_tests += 1

    print(f"\n2. Parametrized Tests:")
    parametrized_passed = 0
    for i, (psi, seeds) in enumerate(itertools.product(_test_states[:3], _test_seeds[:2])):
        try:
            decoded = single_qec_trial(psi, seeds)
            passed = verify_same_state_vector(decoded, psi)
            status = "[PASS]" if passed else "[FAIL]"
            print(f"  Test {i+1}/6: {status}")
            if passed:
                parametrized_passed += 1
            elif not passed:
                print(f"    Failed: decoded {decoded} != original {psi}")
        except Exception as e:
            print(f"  Test {i+1}/6: [FAIL] (Error: {e})")

    print("\n" + "=" * 60)
    print(f"Tests completed: {passed_tests}/{len(tests)} individual tests passed")
    print(f"                 {parametrized_passed}/6 parametrized tests passed")
    print(f"\nCheck log files in log/ directory for detailed execution traces")
    print("=" * 60)