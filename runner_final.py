import json
import sys
import time
from pathlib import Path
from subprocess import PIPE, Popen



BASE_ARGS = [
        "--css_code", "[[7,1,3]]",
        "--target_fidelity", "0.8",
        "--num_logical_pairs", "5000",
        "--link_distance_km", "20",
        "--gate_fidelity", "1",
        "--measurement_fidelity", "1",
        "--initialization_fidelity", "1",
        "--gate_error_channel", "depolarize",
        "--idle_error_channel", "pauli",
        "--idle_t1_sec", "1e12",
        "--idle_t2_sec", "1e12",
        "--ft_prep_mode", "minimal",
        "--idle_pauli_x", "0.05",
        "--idle_pauli_y", "0.05",
        "--idle_pauli_z", "0.9",
        "--run_duration_ms", "1000",
        "--round_spacing_ms", "1",
        "--two_qubit_gate_fidelity", "1",
        "--correction_mode", "cec",
        "--physical_bell_pair_fidelity", "1",
]

Z_PARAM_GRID_FILE = "config/generated_configs/z_plot_param_grid.json"

CONFIG_FILE_BY_TOPOLOGY = {
        "line_2": "config/standard_configs/line_2_2G.json",
        "line_3": "config/standard_configs/line_3_2G.json",
        "line_6": "config/standard_configs/line_6_2G.json",
        "line_11": "config/standard_configs/line_11_2G.json",
        "line_21": "config/standard_configs/line_21_2G.json",
        "line_26": "config/standard_configs/line_26_2G.json",
        "line_51": "config/standard_configs/line_51_2G.json",
        "line_101": "config/standard_configs/line_101_2G.json",
}



def read_process_lines(process: Popen) -> list[str]:
    """Read buffered stdout and stderr lines from one completed subprocess.

    Args:
        process: Completed subprocess.

    Returns:
        list[str]: Non-empty decoded output lines.
    """
    lines: list[str] = []
    if process.stdout is not None:
        stdout_text = process.stdout.read().decode()
        lines.extend(line for line in stdout_text.splitlines() if line)
    if process.stderr is not None:
        stderr_text = process.stderr.read().decode()
        lines.extend(line for line in stderr_text.splitlines() if line)
    return lines

def run_tasks(tasks: list[list[str]], parallel: int = 10) -> None:
    """Run subprocess tasks with bounded parallelism.

    Args:
        tasks: Commands to execute.
        parallel: Maximum number of concurrent subprocesses.

    Returns:
        None.
    """
    base_dir = Path(__file__).resolve().parent
    ps: list[tuple[Popen, list[str], float]] = []
    completed: list[tuple[list[str], list[str]]] = []
    while len(tasks) > 0 or len(ps) > 0:
        if len(ps) < parallel and len(tasks) > 0:
            task = tasks.pop(0)
            print(task, f"{len(tasks)} still in queue")
            process = Popen(task, stdout=PIPE, stderr=PIPE, cwd=base_dir)
            ps.append((process, task, time.time()))
        else:
            time.sleep(0.05)
            new_ps: list[tuple[Popen, list[str], float]] = []
            for process, task, start_time in ps:
                if process.poll() is None:
                    new_ps.append((process, task, start_time))
                else:
                    completed.append((read_process_lines(process), task))
            ps = new_ps
    for output_lines, _task in completed:
        for line in output_lines:
            print(line)

def load_z_param_row(topology: str, z_value: float) -> dict[str, float | str]:
    """Load one z-grid row for a given topology and z value.

    Args:
        topology: Topology name such as line_2, line_3, or line_6.
        z_value: Target z value to match from the generated grid.

    Returns:
        dict[str, float | str]: Matching z-grid row.
    """
    base_dir = Path(__file__).resolve().parent
    grid_path = base_dir / Z_PARAM_GRID_FILE
    with grid_path.open("r", encoding="utf-8") as file:
        param_rows = json.load(file)

    for param_row in param_rows:
        if str(param_row["topology"]) != topology:
            continue
        if abs(float(param_row["z"]) - z_value) < 1e-12:
            return param_row

    raise RuntimeError(f"Missing z-grid row for topology={topology}, z={z_value}")

def build_z_args(topology: str, z_value: float) -> list[str]:
    """Build main.py args from one z-grid row.

    Args:
        topology: Topology name such as line_2, line_3, or line_6.
        z_value: Target z value to match from the generated grid.

    Returns:
        list[str]: CLI args corresponding to the chosen z-grid row.
    """
    param_row = load_z_param_row(topology, z_value)
    return [
        "--gate_fidelity", str(param_row["gate_fidelity"]),
        "--two_qubit_gate_fidelity", str(param_row["two_qubit_gate_fidelity"]),
        "--measurement_fidelity", str(param_row["measurement_fidelity"]),
        "--initialization_fidelity", str(param_row["initialization_fidelity"]),
        "--physical_bell_pair_fidelity", str(param_row["physical_bell_pair_fidelity"]),
        "--idle_t2_sec", str(param_row["idle_t2_sec"]),
    ]

def topology_from_config_file(config_file: str) -> str:
    """Extract the topology name from a standard config filename.

    Args:
        config_file: Config path such as config/standard_configs/line_51_2G.json.

    Returns:
        str: Topology name such as line_51.
    """
    return Path(config_file).stem.replace("_2G", "")

def main_graph1_distance_sweep() -> None:
    """Run the graph 1 distance sweep using the z=0.9 parameter set.

    Args:
        None.

    Returns:
        None.
    """
    tasks: list[list[str]] = []
    base_dir = Path(__file__).resolve().parent
    command = [sys.executable, str(base_dir / "main.py")]
    correction_modes = ["cec"]
    config_files = [
            "config/standard_configs/line_51_2G.json",
            "config/standard_configs/line_101_2G.json",
    ]
    z_value = 0.9

    for config_file in config_files:
        topology = topology_from_config_file(config_file)
        base_args = BASE_ARGS + [
                "--config_file", config_file,
                "--log_directory", "log/runner_final/graph1_distance_sweep",
                "--num_logical_pairs", "500",
        ] + build_z_args(topology, z_value)
        for correction_mode in correction_modes:
            args = ["--correction_mode", correction_mode]
            tasks.append(command + base_args + args)

    run_tasks(tasks, parallel=2)

def main_graph2_link_count_sweep() -> None:
    """Run the graph 2 link-count sweep using the z=0.9 parameter set.

    Args:
        None.

    Returns:
        None.
    """
    tasks: list[list[str]] = []
    base_dir = Path(__file__).resolve().parent
    command = [sys.executable, str(base_dir / "main.py")]
    correction_modes = ["cec"]
    total_end_to_end_distance_km = 100.0
    config_files = [
            "config/standard_configs/line_2_2G.json",
            "config/standard_configs/line_3_2G.json",
            "config/standard_configs/line_6_2G.json",
            "config/standard_configs/line_11_2G.json",
            "config/standard_configs/line_21_2G.json",
            "config/standard_configs/line_26_2G.json",
            "config/standard_configs/line_51_2G.json",
            "config/standard_configs/line_101_2G.json",
    ]
    z_value = 0.9

    for config_file in config_files:
        topology = topology_from_config_file(config_file)
        node_count = int(topology.split("_")[1])
        num_links = node_count - 1
        inter_node_distance_km = total_end_to_end_distance_km / num_links
        base_args = BASE_ARGS + [
                "--config_file", config_file,
                "--link_distance_km", str(inter_node_distance_km),
                "--log_directory", "log/runner_final/graph2_link_count_sweep",
        ] + build_z_args(topology, z_value)
        for correction_mode in correction_modes:
            args = ["--correction_mode", correction_mode]
            tasks.append(command + base_args + args)

    run_tasks(tasks, parallel=2)

def main_graph3_inter_node_distance_sweep() -> None:
    """Run the graph 3 inter-node-distance sweep using the z=0.9 parameter set.

    Args:
        None.

    Returns:
        None.
    """
    tasks: list[list[str]] = []
    base_dir = Path(__file__).resolve().parent
    command = [sys.executable, str(base_dir / "main.py")]
    correction_modes = ["cec"]
    inter_node_distances_km = ["1", "5", "10", "20", "30", "40", "50", "60", "70", "80", "90", "100"]
    config_file = "config/standard_configs/line_6_2G.json"
    z_value = 0.9

    for inter_node_distance_km in inter_node_distances_km:
        base_args = BASE_ARGS + [
                "--config_file", config_file,
                "--link_distance_km", inter_node_distance_km,
                "--run_duration_ms", "20000.0",
                "--log_directory", "log/runner_final/graph3_inter_node_distance_sweep",
                "--num_logical_pairs", "500",
        ] + build_z_args("line_6", z_value)
        for correction_mode in correction_modes:
            args = ["--correction_mode", correction_mode]
            tasks.append(command + base_args + args)

    run_tasks(tasks, parallel=3)

if __name__ == "__main__":
    # main_graph1_distance_sweep()
    # main_graph2_link_count_sweep()
    main_graph3_inter_node_distance_sweep()