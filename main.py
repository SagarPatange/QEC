import argparse
import os
from collections import defaultdict

import sequence.utils.log as log
from sequence.constants import MILLISECOND, SECOND

from router_net_topo_2G import RouterNetTopo2G
from RequestLogicalPairApp import RequestLogicalPairApp


def main() -> None:
    """Run one logical-pair simulation from CLI arguments.

    Args:
        None.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description="Parameters for logical-pair simulation")
    parser.add_argument("--config_file", type=str, default="config/line_5_2G_near_term.json")
    parser.add_argument("--css_code", type=str, default="[[7,1,3]]")
    parser.add_argument("--log_directory", type=str, default="log")
    parser.add_argument("--log_level", type=str, default="DEBUG")
    parser.add_argument("--start_time_s", type=float, default=1.0)
    parser.add_argument("--window_time_ms", type=float, default=1.0)
    parser.add_argument("--target_fidelity", type=float, default=0.8)
    parser.add_argument("--num_logical_pairs", type=int, default=30)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    config_file = args.config_file
    if not os.path.isabs(config_file):
        config_file = os.path.join(os.path.dirname(__file__), config_file)

    if not os.path.exists(args.log_directory):
        os.mkdir(args.log_directory)

    network_topo = RouterNetTopo2G(config_file)
    tl = network_topo.get_timeline()

    log_filename = f"{args.log_directory}/{os.path.splitext(os.path.basename(args.config_file))[0]},code={args.css_code}"
    log.set_logger(__name__, tl, log_filename)
    log.set_logger_level(args.log_level)
    modules = ["main"]
    for module in modules:
        log.track_module(module)

    name_to_apps = {}
    routers = network_topo.get_nodes_by_type(RouterNetTopo2G.QUANTUM_ROUTER)
    routers.sort(key=lambda router: int(router.name.split("_")[-1]))
    node_names = [router.name for router in routers]

    for router in routers:
        app = RequestLogicalPairApp(router, css_code=args.css_code, path_node_names=node_names)
        name_to_apps[router.name] = app

    start_time_ps = int(args.start_time_s * SECOND)
    window_duration_ps = int(args.window_time_ms * MILLISECOND)

    for i in range(len(node_names) - 1):
        name_to_apps[node_names[i]].start(
            responder=node_names[i + 1],
            start_t=start_time_ps,
            end_t=start_time_ps + window_duration_ps,
            fidelity=args.target_fidelity,
            num_logical_pairs=args.num_logical_pairs,
        )

    tl.init()
    tl.run()

    fidelity_dict = defaultdict(float)
    time_to_serve_dict = defaultdict(float)

    final_app = name_to_apps[node_names[0]]
    for run_id, run_stats in sorted(final_app.run_stats.items()):
        fidelity = run_stats["final_end_to_end_fidelity"]
        latency_ps = run_stats["latency_ps"]
        if fidelity is None or latency_ps is None:
            continue
        fidelity_dict[run_id] = float(fidelity)
        time_to_serve_dict[run_id] = float(latency_ps)

    for run_id, time_to_serve in sorted(time_to_serve_dict.items()):
        fidelity = fidelity_dict[run_id]
        log.logger.info(f"run_id={run_id}, time to serve={time_to_serve / MILLISECOND}, fidelity={fidelity:.6f}")
        if args.verbose:
            print(f"run_id={run_id}, time_to_serve_ms={time_to_serve / MILLISECOND:.6f}, fidelity={fidelity:.6f}")


if __name__ == "__main__":
    main()
