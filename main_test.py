print('importing ...')
from concurrent.futures import ProcessPoolExecutor
import json
import time
import numpy as np
from pathlib import Path
from tempfile import NamedTemporaryFile
from sequence.topology.router_net_topo import RouterNetTopo
from router_net_topo_2G import RouterNetTopo2G
import sequence.utils.log as log
from sequence.kernel.quantum_manager import QuantumManager
from sequence.constants import STABILIZER_FORMALISM
from sequence.entanglement_management.generation import EntanglementGenerationA, EntanglementGenerationB, BarretKokA

from request_app import RequestAppThroughput
from RequestLogicalPairApp import RequestLogicalPairApp
from sequence.topology.node import QuantumRouter2ndGeneration

def resolve_config_path(config_file: str) -> str:
    config_path = Path(config_file)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_file
    return str(config_path)


def build_n_node_params() -> dict[str, object]:
    """Build default parameters for the N-node app run.

    Args:
        None.

    Returns:
        dict[str, object]: Default run parameters.
    """
    return {
        "log_level": "WARNING",
        "start_time_s": 1.0,
        "run_duration_ms": 1000.0,
        "round_spacing_ms": 1.0,
        "correction_mode": "cec",
        "target_fidelity": 0.8,
        "num_logical_pairs": 1000,
        "link_distance_km": 1,
        "gate_fidelity": 1,
        "two_qubit_gate_fidelity": 0.995,
        "measurement_fidelity": 1,
        "state_preparation_fidelity": 1,
        "gate_error_channel": "depolarize",
        "pauli_1q_weights": None,
        "pauli_2q_weights": None,
        "idle_t1_sec": 1e12,
        "idle_t2_sec": 1e12,
        "ft_prep_mode": "minimal",
        "idle_pauli_x": 0.05,
        "idle_pauli_y": 0.05,
        "idle_pauli_z": 0.90,
        "physical_bell_pair_fidelity": 1,
    }


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
    tl.quantum_manager.two_qubit_gate_fid = 0.99

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
    correction_mode = "cec"

    for router in network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER):
        if not isinstance(router, QuantumRouter2ndGeneration):
            raise TypeError(f"Node {router.name} must be QuantumRouter2ndGeneration")

        router.correction_mode = correction_mode
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
            target_fidelity=target_fidelity)

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
    tl.quantum_manager.two_qubit_gate_fid = 0.99

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
    correction_mode = "cec"

    for router in network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER):
        if not isinstance(router, QuantumRouter2ndGeneration):
            raise TypeError(f"Node {router.name} must be QuantumRouter2ndGeneration")

        router.correction_mode = correction_mode
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
            target_fidelity=target_fidelity)

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
        print(f"{left} <-> {right} | initial_phys={initial_phys:.6f} | prep={left_app.prep_fidelity:.6f} | ft={left_app.ft_prep_mode} | logical={logical:.6f}")

    final_app = apps[node_names[0]]
    if final_app.current_run["final_end_to_end_fidelity"] is not None:
        metrics["end_to_end_logical"] = float(final_app.current_run["final_end_to_end_fidelity"])
        metrics["end_to_end_logical_raw"] = float(final_app.current_run["final_end_to_end_fidelity_raw"])
        metrics["end_to_end_logical_corrected"] = float(final_app.current_run["final_end_to_end_fidelity_corrected"])
        print("\nEnd-to-End")
        print(
            f"{node_names[0]} <-> {node_names[-1]} | "
            f"raw={final_app.current_run['final_end_to_end_fidelity_raw']:.6f} | "
            f"corrected={final_app.current_run['final_end_to_end_fidelity_corrected']:.6f}"
        )

    if initial_values:
        metrics["avg_initial_phys"] = float(np.mean(initial_values))
    if logical_values:
        metrics["avg_link_logical"] = float(np.mean(logical_values))

    print(f"\n--- {num_links}-link pipeline complete ---")

    return {"apps": apps, "metrics": metrics}


def n_node_logical_pair_with_app(verbose: bool = False, config_file: str = "config/standard_configs/line_2_2G.json",
                                 css_code: str = "[[7,1,3]]", log_filename: str = "log/n_node_logical_pair",
                                 params: dict[str, object] | None = None) -> dict[str, object]:
    """Create end-to-end logical entanglement across an N-node linear chain.

    Args:
        verbose: Whether to print verbose diagnostics.
        config_file: Topology config file path.
        css_code: CSS code label for the run.
        log_filename: Log file prefix.
        params: Run-parameter overrides.

    Returns:
        dict[str, object]: App objects and collected run metrics.
    """
    merged_params = build_n_node_params()
    if params is not None:
        merged_params.update(params)
    params = merged_params

    resolved_config = resolve_config_path(config_file)
    with open(resolved_config, "r", encoding="utf-8") as file:
        config = json.load(file)

    idle_pauli_x = params["idle_pauli_x"]
    idle_pauli_y = params["idle_pauli_y"]
    idle_pauli_z = params["idle_pauli_z"]
    pauli_1q_weights = params["pauli_1q_weights"]
    pauli_2q_weights = params["pauli_2q_weights"]

    if (idle_pauli_x is None) != (idle_pauli_y is None) or (idle_pauli_x is None) != (idle_pauli_z is None):
        raise RuntimeError("idle_pauli_x, idle_pauli_y, idle_pauli_z must be set together")
    if pauli_1q_weights is not None and len(pauli_1q_weights) != 3:
        raise RuntimeError("pauli_1q_weights must have 3 entries")
    if pauli_2q_weights is not None and len(pauli_2q_weights) != 15:
        raise RuntimeError("pauli_2q_weights must have 15 entries")

    config["templates"]["qec"]["memory"]["fidelity"] = float(params["physical_bell_pair_fidelity"])

    for node in config["nodes"]:
        if node.get("type") != "QuantumRouter":
            continue
        if params["gate_fidelity"] is not None:
            node["gate_fidelity"] = float(params["gate_fidelity"])
        if params["two_qubit_gate_fidelity"] is not None:
            node["two_qubit_gate_fidelity"] = float(params["two_qubit_gate_fidelity"])
        if params["measurement_fidelity"] is not None:
            node["measurement_fidelity"] = float(params["measurement_fidelity"])
        if params["state_preparation_fidelity"] is not None:
            node["state_preparation_fidelity"] = float(params["state_preparation_fidelity"])
        if params["gate_error_channel"] is not None:
            node["gate_error_channel"] = str(params["gate_error_channel"])
        if pauli_1q_weights is not None:
            node["pauli_1q_weights"] = list(pauli_1q_weights)
        if pauli_2q_weights is not None:
            node["pauli_2q_weights"] = list(pauli_2q_weights)
        if params["idle_t1_sec"] is not None:
            node["idle_t1_sec"] = float(params["idle_t1_sec"])
        if params["idle_t2_sec"] is not None:
            node["idle_t2_sec"] = float(params["idle_t2_sec"])
        if params["ft_prep_mode"] is not None:
            node["ft_prep_mode"] = str(params["ft_prep_mode"])
        node["correction_mode"] = str(params["correction_mode"])
        if idle_pauli_x is not None:
            node["idle_pauli_weights"] = {
                "x": float(idle_pauli_x),
                "y": float(idle_pauli_y),
                "z": float(idle_pauli_z),
            }

    if params["link_distance_km"] is not None:
        half_distance_m = float(params["link_distance_km"]) * 1000.0 / 2.0
        for qchannel in config["qchannels"]:
            qchannel["distance"] = half_distance_m

    temp_config = NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(config, temp_config, indent=4)
    temp_config.write("\n")
    temp_config.close()

    network_config = temp_config.name

    network_topo = RouterNetTopo2G(network_config)
    tl = network_topo.get_timeline()

    routers = network_topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER)
    tl.quantum_manager.gate_fid = getattr(routers[0], 'gate_fid', 1.0)
    tl.quantum_manager.two_qubit_gate_fid = getattr(routers[0], 'two_qubit_gate_fid', 1.0)
    tl.quantum_manager.measurement_fid = getattr(routers[0], 'meas_fid', 1.0)
    tl.quantum_manager.state_preparation_fid = getattr(routers[0], 'state_preparation_fid', 1.0)
    if params["gate_error_channel"] is not None:
        tl.quantum_manager.gate_error_channel = str(params["gate_error_channel"])
    if pauli_1q_weights is not None:
        tl.quantum_manager.pauli_1q_weights = tuple(float(w) for w in pauli_1q_weights)
    if pauli_2q_weights is not None:
        tl.quantum_manager.pauli_2q_weights = tuple(float(w) for w in pauli_2q_weights)

    log.set_logger(__name__, tl, log_filename)
    log.set_logger_level("WARNING")
    # modules = ['timeline', 'network_manager', 'resource_manager', 'rule_manager', 'generation', 'purification', 'swapping', 'bsm', 'barret_kok', 'RequestLogicalPairApp', 'TeleportedCNOT', 'QREProtocol']
    # modules = ['barret_kok']
    modules = ['RequestLogicalPairApp']

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

    start_time_ps = int(float(params["start_time_s"]) * 1e12)
    window_duration_ps = int(float(params["run_duration_ms"]) * 1e9)
    round_spacing_ps = int(float(params["round_spacing_ms"]) * 1e9)
    target_fidelity = float(params["target_fidelity"])
    num_logical_pairs = int(params["num_logical_pairs"])
    correction_mode = str(params["correction_mode"])

    apps = {}
    for node_name in node_names:
        routers_by_name[node_name].round_spacing_ps = round_spacing_ps
        if idle_pauli_x is not None:
            routers_by_name[node_name].idle_pauli_weights = {
                "x": float(idle_pauli_x),
                "y": float(idle_pauli_y),
                "z": float(idle_pauli_z),
            }
        if params["idle_t1_sec"] is not None:
            routers_by_name[node_name].idle_t1_sec = float(params["idle_t1_sec"])
        if params["idle_t2_sec"] is not None:
            routers_by_name[node_name].idle_t2_sec = float(params["idle_t2_sec"])
        routers_by_name[node_name].correction_mode = correction_mode
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
            fidelity=target_fidelity,
            num_logical_pairs=num_logical_pairs)

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
            "avg_end_to_end_logical_raw": float("nan"),
            "avg_end_to_end_logical_corrected": float("nan"),
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

                metrics["end_to_end_rows"].append({
                    "run_id": int(run_id),
                    "fidelity": float(run_stats["final_end_to_end_fidelity"]) if run_stats["final_end_to_end_fidelity"] is not None else float("nan"),
                    "fidelity_raw": float(run_stats["final_end_to_end_fidelity_raw"]) if run_stats["final_end_to_end_fidelity_raw"] is not None else float("nan"),
                    "fidelity_corrected": float(run_stats["final_end_to_end_fidelity_corrected"]) if run_stats["final_end_to_end_fidelity_corrected"] is not None else float("nan"),
                    "latency_ps": latency_ps if latency_ps is not None else float("nan"),
                    "latency_s": latency_s,
                    "throughput_pairs_per_s": throughput_pairs_per_s,
                })
        metrics["num_logical_pairs_completed"] = len(metrics["end_to_end_rows"])
        # Aggregate link-level averages.
        initial_values = [r["initial_phys"] for r in metrics["link_rows"] if not np.isnan(r["initial_phys"])]
        logical_values = [r["logical"] for r in metrics["link_rows"] if not np.isnan(r["logical"])]
        if initial_values:
            metrics["avg_initial_phys"] = float(np.mean(initial_values))
        if logical_values:
            metrics["avg_link_logical"] = float(np.mean(logical_values))
        e2e_values = [r["fidelity"] for r in metrics["end_to_end_rows"] if not np.isnan(r["fidelity"])]
        e2e_raw_values = [r["fidelity_raw"] for r in metrics["end_to_end_rows"] if not np.isnan(r["fidelity_raw"])]
        e2e_corrected_values = [r["fidelity_corrected"] for r in metrics["end_to_end_rows"] if not np.isnan(r["fidelity_corrected"])]
        latency_values = [r["latency_ps"] for r in metrics["end_to_end_rows"] if not np.isnan(r["latency_ps"])]
        throughput_values = [r["throughput_pairs_per_s"] for r in metrics["end_to_end_rows"] if not np.isnan(r["throughput_pairs_per_s"])]
        if e2e_values:
            metrics["avg_end_to_end_logical"] = float(np.mean(e2e_values))
        if e2e_raw_values:
            metrics["avg_end_to_end_logical_raw"] = float(np.mean(e2e_raw_values))
        if e2e_corrected_values:
            metrics["avg_end_to_end_logical_corrected"] = float(np.mean(e2e_corrected_values))
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
        print(f"params: n={final_app.n} ft_mode={final_app.ft_prep_mode} idle_weights={final_app.idle_pauli_weights} T1={final_app.idle_t1_sec:.3e}s T2={final_app.idle_t2_sec:.3e}s")
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
            header = f"{'Run':>4} {'Raw Fid':>10} {'Corr Fid':>10} {'Latency (ps)':>14} {'Latency (ms)':>14} {'Throughput':>14}"
            print(header)
            print("-" * len(header))
            for row in metrics["end_to_end_rows"]:
                print(f"{row['run_id']:>4d} {row['fidelity_raw']:>10.4f} {row['fidelity_corrected']:>10.4f} {row['latency_ps']:>14.0f} {row['latency_ps'] * 1e-9:>14.6f} {row['throughput_pairs_per_s']:>14.6e}")

            print(f"{'Avg':>4} {metrics['avg_end_to_end_logical_raw']:>10.4f} {metrics['avg_end_to_end_logical_corrected']:>10.4f} {metrics['avg_latency_ps']:>14.0f} {metrics['avg_latency_ps'] * 1e-9:>14.6f} {metrics['avg_throughput_pairs_per_s']:>14.6e}")
        if not np.isnan(metrics["avg_end_to_end_logical"]):
            print(f"Avg end-to-end ({node_names[0]} <-> {node_names[-1]}): raw={metrics['avg_end_to_end_logical_raw']:.4f} | corrected={metrics['avg_end_to_end_logical_corrected']:.4f}")
        if not np.isnan(metrics["avg_latency_ps"]):
            print(f"Avg latency: {metrics['avg_latency_ps']:.0f} ps ({metrics['avg_latency_s']:.6e} s, {metrics['avg_latency_ps'] * 1e-9:.6f} ms)")
            print(f"Avg throughput: {metrics['avg_throughput_pairs_per_s']:.6e} pairs/s")
    
    metrics = _collect_metrics()
    _print_metrics_summary(metrics)
    print(f"=== {num_links}-link pipeline complete ===")
    return {"apps": apps, "metrics": metrics}


def split_pair_counts(total_pairs: int, workers: int) -> list[int]:
    """Split a total logical-pair budget across workers.

    Args:
        total_pairs: Total number of logical pairs to simulate.
        workers: Number of worker processes.

    Returns:
        list[int]: Per-worker logical-pair counts.
    """
    base = total_pairs // workers
    extra = total_pairs % workers
    return [base + (1 if i < extra else 0) for i in range(workers)]


def run_n_node_worker(worker_index: int, pair_count: int, base_log_filename: str) -> dict[str, object]:
    """Run one independent N-node simulation worker.

    Args:
        worker_index: Worker index for unique log naming.
        pair_count: Number of logical pairs assigned to this worker.
        base_log_filename: Base log filename shared by all workers.

    Returns:
        dict[str, object]: Metrics dictionary for the worker.
    """
    params = build_n_node_params()
    params["num_logical_pairs"] = pair_count

    result = n_node_logical_pair_with_app(
        verbose=False,
        config_file="config/standard_configs/line_2_2G.json",
        css_code="[[7,1,3]]",
        log_filename=f"{base_log_filename}_worker_{worker_index}",
        params=params,
    )
    return result["metrics"]


def run_parallel_n_node_trials(total_pairs: int, workers: int, log_filename: str) -> list[dict[str, object]]:
    """Run multiple independent N-node simulations in parallel.

    Args:
        total_pairs: Total logical-pair budget across all workers.
        workers: Number of worker processes.
        log_filename: Base log filename shared by all workers.

    Returns:
        list[dict[str, object]]: Per-worker metrics.
    """
    pair_counts = split_pair_counts(total_pairs, workers)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(run_n_node_worker, worker_index, pair_count, log_filename)
            for worker_index, pair_count in enumerate(pair_counts)
            if pair_count > 0
        ]
        return [future.result() for future in futures]


def merge_worker_logs(base_log_filename: str, workers: int) -> str:
    """Merge worker log files into one combined log file.

    Args:
        base_log_filename: Destination log filename.
        workers: Number of worker log files to merge.

    Returns:
        str: Combined log filename.
    """
    destination = Path(base_log_filename)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", encoding="utf-8") as output_file:
        for worker_index in range(workers):
            worker_log = Path(f"{base_log_filename}_worker_{worker_index}")
            if not worker_log.exists():
                continue
            with worker_log.open("r", encoding="utf-8") as input_file:
                contents = input_file.read()
            output_file.write(contents)
            if contents and not contents.endswith("\n"):
                output_file.write("\n")

    return str(destination)


def summarize_parallel_worker_metrics(worker_metrics: list[dict[str, object]]) -> dict[str, float | int]:
    """Aggregate per-worker metrics into one combined summary.

    Args:
        worker_metrics: Metrics returned by parallel workers.

    Returns:
        dict[str, float | int]: Combined summary values.
    """
    completed_pairs = sum(int(m["num_logical_pairs_completed"]) for m in worker_metrics)
    requested_pairs = sum(int(m["num_logical_pairs_requested"]) for m in worker_metrics)

    raw_values = [
        float(row["fidelity_raw"])
        for metrics in worker_metrics
        for row in metrics["end_to_end_rows"]
        if not np.isnan(row["fidelity_raw"])
    ]
    corrected_values = [
        float(row["fidelity_corrected"])
        for metrics in worker_metrics
        for row in metrics["end_to_end_rows"]
        if not np.isnan(row["fidelity_corrected"])
    ]
    latency_values = [
        float(row["latency_ps"])
        for metrics in worker_metrics
        for row in metrics["end_to_end_rows"]
        if not np.isnan(row["latency_ps"])
    ]
    throughput_values = [
        float(row["throughput_pairs_per_s"])
        for metrics in worker_metrics
        for row in metrics["end_to_end_rows"]
        if not np.isnan(row["throughput_pairs_per_s"])
    ]

    return {
        "num_workers": len(worker_metrics),
        "num_logical_pairs_requested": requested_pairs,
        "num_logical_pairs_completed": completed_pairs,
        "avg_end_to_end_logical_raw": float(np.mean(raw_values)) if raw_values else float("nan"),
        "avg_end_to_end_logical_corrected": float(np.mean(corrected_values)) if corrected_values else float("nan"),
        "avg_latency_ps": float(np.mean(latency_values)) if latency_values else float("nan"),
        "avg_latency_s": float(np.mean(latency_values)) * 1e-12 if latency_values else float("nan"),
        "avg_throughput_pairs_per_s": float(np.mean(throughput_values)) if throughput_values else float("nan"),
    }


def print_parallel_run_summary(summary: dict[str, float | int]) -> None:
    """Print one combined summary for all parallel workers.

    Args:
        summary: Aggregated parallel-run summary.

    Returns:
        None.
    """
    print("\n=== Combined Parallel Run Summary ===")
    print(f"workers={summary['num_workers']}")
    print(
        f"logical_pairs: requested={summary['num_logical_pairs_requested']} "
        f"completed={summary['num_logical_pairs_completed']}"
    )
    print(
        f"Avg end-to-end: raw={summary['avg_end_to_end_logical_raw']:.4f} | "
        f"corrected={summary['avg_end_to_end_logical_corrected']:.4f}"
    )
    print(
        f"Avg latency: {summary['avg_latency_ps']:.0f} ps "
        f"({summary['avg_latency_s']:.6e} s, {summary['avg_latency_ps'] * 1e-9:.6f} ms)"
    )
    print(f"Avg throughput: {summary['avg_throughput_pairs_per_s']:.6e} pairs/s")

if __name__ == "__main__":

    start = time.time()
    print("main start ...")
    # ----- N-node runs -----
    total_pairs = 1080
    workers = 12
    base_log_filename = "log/n_node_logical_pair"
    worker_metrics = run_parallel_n_node_trials(
        total_pairs=total_pairs,
        workers=workers,
        log_filename=base_log_filename,
    )
    merged_log = merge_worker_logs(base_log_filename, workers)
    combined_summary = summarize_parallel_worker_metrics(worker_metrics)
    print_parallel_run_summary(combined_summary)
    print(f"merged log: {merged_log}")

    # Add more N-node runs as needed:
    # n_node_logical_pair_with_app(verbose=True)


    end = time.time()
    print(f"\nMain execution time: {end - start:.2f} seconds")
