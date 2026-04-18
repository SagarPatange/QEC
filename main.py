import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict
from time import perf_counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEQUENCE_ROOT = os.path.join(os.path.dirname(BASE_DIR), "SeQUeNCe")
TMP_DIR = os.path.join(BASE_DIR, ".tmp")
MPL_CONFIG_DIR = os.path.join(BASE_DIR, ".mplconfig")

os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(MPL_CONFIG_DIR, exist_ok=True)

os.environ["TMPDIR"] = TMP_DIR
os.environ["MPLCONFIGDIR"] = MPL_CONFIG_DIR

if SEQUENCE_ROOT not in sys.path:
    sys.path.insert(0, SEQUENCE_ROOT)

import sequence.utils.log as log
from sequence.constants import MILLISECOND, SECOND, TABLEAU_FORMALISM
from router_net_topo_2G import RouterNetTopo2G
from RequestLogicalPairApp import RequestLogicalPairApp


def main() -> None:
    """Run one logical-pair simulation from CLI arguments.

    Args:
        None.

    Returns:
        None.
    """
    total_tick = perf_counter()
    config_tick = perf_counter()

    parser = argparse.ArgumentParser(description="Parameters for logical-pair simulation")
    parser.add_argument("--config_file", type=str, default="config/standard_configs/line_6_2G.json")
    parser.add_argument("--css_code", type=str, default="[[7,1,3]]")
    parser.add_argument("--log_directory", type=str, default="log")
    parser.add_argument("--debug_log_directory", type=str)
    parser.add_argument("--log_level", type=str, default="DEBUG")
    parser.add_argument("--start_time_s", type=float, default=1.0)
    parser.add_argument("--run_duration_ms", type=float, default=1e5)
    parser.add_argument("--round_spacing_ms", type=float, default=1.0)
    parser.add_argument("--correction_mode", type=str, choices=["none", "cec", "qec", "qec+cec"], default="cec")
    parser.add_argument("--target_fidelity", type=float, default=0.8)
    parser.add_argument("--num_logical_pairs", type=int, default=30)
    parser.add_argument("--link_distance_km", type=float)
    parser.add_argument("--gate_fidelity", type=float)
    parser.add_argument("--two_qubit_gate_fidelity", type=float)
    parser.add_argument("--measurement_fidelity", type=float)
    parser.add_argument("--state_preparation_fidelity", type=float, default=0.9999)
    parser.add_argument("--gate_error_channel", type=str, choices=["depolarize", "pauli"])
    parser.add_argument("--pauli_1q_weights", type=float, nargs=3)
    parser.add_argument("--pauli_2q_weights", type=float, nargs=15)
    parser.add_argument("--idle_t1_sec", type=float)
    parser.add_argument("--idle_t2_sec", type=float)
    parser.add_argument("--ft_prep_mode", type=str, choices=["none", "minimal", "standard"])
    parser.add_argument("--idle_pauli_x", type=float)
    parser.add_argument("--idle_pauli_y", type=float)
    parser.add_argument("--idle_pauli_z", type=float)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    config_file = args.config_file
    if not os.path.isabs(config_file):
        config_file = os.path.join(os.path.dirname(__file__), config_file)
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")

    # Keep grouped CLI overrides consistent before mutating the loaded config.
    if (args.idle_pauli_x is None) != (args.idle_pauli_y is None) or (args.idle_pauli_x is None) != (args.idle_pauli_z is None):
        raise RuntimeError("idle_pauli_x, idle_pauli_y, idle_pauli_z must be set together")
    if args.pauli_1q_weights is not None and len(args.pauli_1q_weights) != 3:
        raise RuntimeError("pauli_1q_weights must have 3 entries")
    if args.pauli_2q_weights is not None and len(args.pauli_2q_weights) != 15:
        raise RuntimeError("pauli_2q_weights must have 15 entries")

    with open(config_file, "r", encoding="utf-8") as file:
        config = json.load(file)

    # Force the run onto the tableau backend regardless of the config default.
    config["formalism"] = TABLEAU_FORMALISM

    # Apply per-router hardware and protocol overrides from the CLI.
    for node in config["nodes"]:
        if node.get("type") != "QuantumRouter":
            continue
        if args.gate_fidelity is not None:
            node["gate_fidelity"] = float(args.gate_fidelity)
        if args.two_qubit_gate_fidelity is not None:
            node["two_qubit_gate_fidelity"] = float(args.two_qubit_gate_fidelity)
        if args.measurement_fidelity is not None:
            node["measurement_fidelity"] = float(args.measurement_fidelity)
        if args.state_preparation_fidelity is not None:
            node["state_preparation_fidelity"] = float(args.state_preparation_fidelity)
        if args.idle_t1_sec is not None:
            node["idle_t1_sec"] = float(args.idle_t1_sec)
        if args.idle_t2_sec is not None:
            node["idle_t2_sec"] = float(args.idle_t2_sec)
        if args.ft_prep_mode is not None:
            node["ft_prep_mode"] = args.ft_prep_mode
        if args.correction_mode is not None:
            node["correction_mode"] = args.correction_mode
        if args.idle_pauli_x is not None:
            node["idle_pauli_weights"] = {"x": float(args.idle_pauli_x), "y": float(args.idle_pauli_y), "z": float(args.idle_pauli_z)}

    # Update all optical link distances together when sweeping link length.
    if args.link_distance_km is not None:
        half_distance_m = float(args.link_distance_km) * 1000.0 / 2.0
        for qchannel in config["qchannels"]:
            qchannel["distance"] = half_distance_m

    # Extend the timeline stop time to cover all requested rounds plus spacing.
    start_time_ps = int(args.start_time_s * SECOND)
    run_duration_ps = int(args.run_duration_ms * MILLISECOND)
    round_spacing_ps = int(args.round_spacing_ms * MILLISECOND)
    config["stop_time"] = start_time_ps + args.num_logical_pairs * (run_duration_ps + round_spacing_ps) + run_duration_ps

    # Materialize a temporary config after CLI overrides so topology loading sees one resolved file.
    temp_config = tempfile.NamedTemporaryFile(mode="w", suffix=".json", dir=TMP_DIR, delete=False, encoding="utf-8")
    json.dump(config, temp_config, indent=4)
    temp_config.write("\n")
    temp_config.close()
    config_elapsed = perf_counter() - config_tick

    os.makedirs(args.log_directory, exist_ok=True)
    if args.debug_log_directory is not None:
        os.makedirs(args.debug_log_directory, exist_ok=True)

    config_tag = os.path.splitext(os.path.basename(args.config_file))[0]
    dist_tag = "cfg" if args.link_distance_km is None else str(args.link_distance_km)
    gate_tag = "cfg" if args.gate_fidelity is None else str(args.gate_fidelity)
    twoq_tag = "cfg" if args.two_qubit_gate_fidelity is None else str(args.two_qubit_gate_fidelity)
    prep_tag = "cfg" if args.state_preparation_fidelity is None else str(args.state_preparation_fidelity)
    t1_tag = "cfg" if args.idle_t1_sec is None else str(args.idle_t1_sec)
    t2_tag = "cfg" if args.idle_t2_sec is None else str(args.idle_t2_sec)
    ft_tag = "cfg" if args.ft_prep_mode is None else args.ft_prep_mode
    pauli_tag = "cfg" if args.idle_pauli_x is None else f"{args.idle_pauli_x}_{args.idle_pauli_y}_{args.idle_pauli_z}"
    correction_tag = args.correction_mode
    debug_log_path = None if args.debug_log_directory is None else (
        f"{args.debug_log_directory}/{config_tag},code={args.css_code},dist={dist_tag},gate={gate_tag},"
        f"twoq={twoq_tag},prep={prep_tag},T1={t1_tag},T2={t2_tag},ft={ft_tag},pauli={pauli_tag},ccorr={correction_tag}.debug"
    )

    def write_debug_line(line: str) -> None:
        """Append one debug timing line when debug logging is enabled.

        Args:
            line: Debug line to append.

        Returns:
            None.
        """
        if debug_log_path is None:
            return
        with open(debug_log_path, "a", encoding="utf-8") as file:
            file.write(line + "\n")

    topo_tick = perf_counter()
    network_topo = RouterNetTopo2G(temp_config.name)
    tl = network_topo.get_timeline()
    topo_elapsed = perf_counter() - topo_tick

    log_filename = f"{args.log_directory}/{config_tag},code={args.css_code},dist={dist_tag},gate={gate_tag},twoq={twoq_tag},prep={prep_tag},T1={t1_tag},T2={t2_tag},ft={ft_tag},pauli={pauli_tag},ccorr={correction_tag}"
    log.set_logger(__name__, tl, log_filename)
    log.set_logger_level(args.log_level)
    modules = ["main"]
    for module in modules:
        log.track_module(module)

    name_to_apps = {}
    routers = network_topo.get_nodes_by_type(RouterNetTopo2G.QUANTUM_ROUTER)
    routers.sort(key=lambda router: int(router.name.split("_")[-1]))
    node_names = [router.name for router in routers]
    # Mirror router-level hardware parameters onto the shared tableau quantum manager.
    tl.quantum_manager.gate_fid = getattr(routers[0], "gate_fid", 1.0)
    tl.quantum_manager.two_qubit_gate_fid = getattr(routers[0], "two_qubit_gate_fid", 1.0)
    tl.quantum_manager.measurement_fid = getattr(routers[0], "meas_fid", 1.0)
    tl.quantum_manager.state_preparation_fid = getattr(routers[0], "state_preparation_fid", 1.0)
    if args.gate_error_channel is not None:
        tl.quantum_manager.gate_error_channel = args.gate_error_channel
    if args.pauli_1q_weights is not None:
        tl.quantum_manager.pauli_1q_weights = tuple(float(w) for w in args.pauli_1q_weights)
    if args.pauli_2q_weights is not None:
        tl.quantum_manager.pauli_2q_weights = tuple(float(w) for w in args.pauli_2q_weights)

    for router in routers:
        router.round_spacing_ps = round_spacing_ps
        router.correction_mode = args.correction_mode
        app = RequestLogicalPairApp(router, css_code=args.css_code, path_node_names=node_names)
        name_to_apps[router.name] = app

    # Launch logical-pair requests from each router to its right-hand neighbor along the path.
    schedule_tick = perf_counter()
    for i in range(len(node_names) - 1):
        name_to_apps[node_names[i]].start(
            responder=node_names[i + 1],
            start_t=start_time_ps,
            end_t=start_time_ps + run_duration_ps,
            fidelity=args.target_fidelity,
            num_logical_pairs=args.num_logical_pairs)
    schedule_elapsed = perf_counter() - schedule_tick

    timing_line = (
        f"[timing] setup config={config_elapsed:.3f}s topo={topo_elapsed:.3f}s "
        f"schedule={schedule_elapsed:.3f}s stop_time_ps={config['stop_time']} "
        f"queued_events={len(tl.events)}"
    )
    print(timing_line, flush=True)
    write_debug_line(timing_line)

    init_tick = perf_counter()
    tl.init()
    init_elapsed = perf_counter() - init_tick
    timing_line = (
        f"[timing] after init init={init_elapsed:.3f}s "
        f"queued_events={len(tl.events)} scheduled_events={tl.schedule_counter}"
    )
    print(timing_line, flush=True)
    write_debug_line(timing_line)

    run_tick = perf_counter()
    tl.run()
    run_elapsed = perf_counter() - run_tick
    timing_line = (
        f"[timing] after run run={run_elapsed:.3f}s "
        f"executed_events={tl.run_counter} remaining_events={len(tl.events)}"
    )
    print(timing_line, flush=True)
    write_debug_line(timing_line)
    completed_line = f"completed_runs={len(name_to_apps[node_names[0]].run_stats)}"
    print(completed_line)
    write_debug_line(completed_line)

    fidelity_dict = defaultdict(float)
    time_to_serve_dict = defaultdict(float)

    final_app = name_to_apps[node_names[0]]
    # Pull per-run latency and final fidelity into flat maps for logging/printing.
    for run_id, run_stats in sorted(final_app.run_stats.items()):
        fidelity = run_stats["final_end_to_end_fidelity"]
        latency_ps = run_stats["latency_ps"]
        if fidelity is None or latency_ps is None:
            continue
        fidelity_dict[run_id] = float(fidelity)
        time_to_serve_dict[run_id] = float(latency_ps)

    finalize_tick = perf_counter()
    for run_id, time_to_serve in sorted(time_to_serve_dict.items()):
        fidelity = fidelity_dict[run_id]
        log.logger.info(f"run_id={run_id}, time to serve={time_to_serve / MILLISECOND}, fidelity={fidelity:.6f}")
        if args.verbose:
            print(f"run_id={run_id}, time_to_serve_ms={time_to_serve / MILLISECOND:.6f}, fidelity={fidelity:.6f}")
    finalize_elapsed = perf_counter() - finalize_tick

    total_elapsed = perf_counter() - total_tick
    timing_line = f"[timing] finalize={finalize_elapsed:.3f}s total={total_elapsed:.3f}s"
    print(timing_line, flush=True)
    write_debug_line(timing_line)

    # Remove the resolved temporary config file after the run completes.
    os.remove(temp_config.name)


if __name__ == "__main__":
    main()
