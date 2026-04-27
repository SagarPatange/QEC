import sys
import time
from pathlib import Path
from subprocess import PIPE, Popen


BASE_ARGS = [
    "--config_file", "config/standard_configs/line_6_2G.json",
    "--css_code", "[[7,1,3]]",
    "--target_fidelity", "0.8",
    "--num_logical_pairs", "1000",
    "--link_distance_km", "20.0",
    "--gate_fidelity", "0.9996",
    "--two_qubit_gate_fidelity", "0.997",
    "--measurement_fidelity", "0.995",
    "--state_preparation_fidelity", "0.9999",
    "--gate_error_channel", "pauli",
    "--idle_t1_sec", "100.0",
    "--idle_t2_sec", "10.0",
    "--ft_prep_mode", "minimal",
    "--idle_pauli_x", "0.05",
    "--idle_pauli_y", "0.05",
    "--idle_pauli_z", "0.9",
    "--run_duration_ms", "100000.0",
    "--round_spacing_ms", "1.0",
]


def get_output(process: Popen) -> None:
    """Print subprocess stdout and stderr.

    Args:
        process: Completed subprocess.

    Returns:
        None.
    """
    stderr = process.stderr.readlines()
    if stderr:
        for line in stderr:
            print(line.decode().rstrip())
    stdout = process.stdout.readlines()
    if stdout:
        for line in stdout:
            print(line.decode().rstrip())


def run_tasks(tasks: list[list[str]], parallel: int = 10) -> None:
    """Run subprocess tasks with bounded parallelism.

    Args:
        tasks: Commands to execute.
        parallel: Maximum number of concurrent subprocesses.

    Returns:
        None.
    """
    base_dir = Path(__file__).resolve().parent
    ps = []
    while len(tasks) > 0 or len(ps) > 0:
        if len(ps) < parallel and len(tasks) > 0:
            task = tasks.pop(0)
            print(task, f"{len(tasks)} still in queue")
            ps.append(Popen(task, stdout=PIPE, stderr=PIPE, cwd=base_dir))
        else:
            time.sleep(0.05)
            new_ps = []
            for process in ps:
                if process.poll() is None:
                    new_ps.append(process)
                else:
                    get_output(process)
            ps = new_ps

# Graph 1
def main_graph1_twoqubit_gate_sweep() -> None:
    """Compare correction modes across a sweep of two-qubit gate fidelities.

    Args:
        None.

    Returns:
        None.
    """
    tasks = []
    base_dir = Path(__file__).resolve().parent
    command = [sys.executable, str(base_dir / "main.py")]
    base_args = BASE_ARGS + ["--log_directory", "log/runner-ideal/graph1_twoqubit_gate_sweep"]

    two_qubit_gate_fidelities = ["0.995", "0.996", "0.997", "0.998", "0.999", "0.9995", "1.0"]

    correction_modes = ["none", "cec", "qec"]
    for two_qubit_gate_fidelity in two_qubit_gate_fidelities:
        # gate_fidelity = min(1.0, (0.9996 / 0.997) * float(two_qubit_gate_fidelity))
        # measurement_fidelity = min(1.0, (0.995 / 0.997) * float(two_qubit_gate_fidelity))
        for correction_mode in correction_modes:
            args = [
                "--two_qubit_gate_fidelity", two_qubit_gate_fidelity,
                # "--gate_fidelity", f"{gate_fidelity:.6f}",
                # "--measurement_fidelity", f"{measurement_fidelity:.6f}",
                "--correction_mode", correction_mode,
            ]
            tasks.append(command + base_args + args)

    run_tasks(tasks, parallel=21)


# Graph 2
def main_graph2_data_coherence_sweep() -> None:
    """Run a T2 sweep at fixed topology and hardware settings.

    Args:
        None.

    Returns:
        None.
    """
    tasks = []
    base_dir = Path(__file__).resolve().parent
    command = [sys.executable, str(base_dir / "main.py")]
    base_args = BASE_ARGS + ["--log_directory", "log/runner-ideal/graph2_t2_sweep"]

    idle_t2_times = [0.01, 0.1, 1, 5, 10, 50, 100, 199.999]  # Sweep only T2 in seconds while holding T1 fixed.

    correction_modes = ["none", "cec", "qec"]
    for idle_t2_sec in idle_t2_times:
        for correction_mode in correction_modes:
            args = ["--idle_t2_sec", str(idle_t2_sec), "--correction_mode", correction_mode]
            tasks.append(command + base_args + args)

    run_tasks(tasks, parallel=24)


# Graph 3 (slow)
def main_graph3_distance_sweep() -> None:
    """Run one distance sweep for both latency and fidelity plots.

    Args:
        None.

    Returns:
        None.
    """
    tasks = []
    base_dir = Path(__file__).resolve().parent
    command = [sys.executable, str(base_dir / "main.py")]
    correction_modes = ["none", "cec", "qec"]
    config_duration_pairs = [("config/standard_configs/line_2_2G.json"), ("config/standard_configs/line_6_2G.json"), 
                             ("config/standard_configs/line_11_2G.json"), ("config/standard_configs/line_26_2G.json")]
    # config_duration_pairs = [("config/standard_configs/line_51_2G.json"), ("config/standard_configs/line_101_2G.json")]

    for config_file in config_duration_pairs:
        base_args = BASE_ARGS + ["--config_file", config_file, "--log_directory", "log/runner-ideal/graph3_distance_sweep", "--num_logical_pairs", "1000"]
        for correction_mode in correction_modes:
            args = ["--correction_mode", correction_mode]
            tasks.append(command + base_args + args)

    run_tasks(tasks, parallel=18)


# Graph 4 (slow)
def main_graph4_link_count_sweep() -> None:
    """Run a fixed-total-distance sweep while varying repeater count.

    Args:
        None.

    Returns:
        None.
    """
    tasks = []
    base_dir = Path(__file__).resolve().parent
    command = [sys.executable, str(base_dir / "main.py")]
    correction_modes = ["none", "cec", "qec"]
    total_end_to_end_distance_km = 100.0
    config_duration_pairs = [("config/standard_configs/line_2_2G.json"), ("config/standard_configs/line_3_2G.json"),
                             ("config/standard_configs/line_6_2G.json"), ("config/standard_configs/line_11_2G.json"), 
                             ("config/standard_configs/line_21_2G.json")]

    config_duration_pairs = [("config/standard_configs/line_51_2G.json"),("config/standard_configs/line_101_2G.json")]


    for config_file in config_duration_pairs:
        node_count = int(Path(config_file).stem.split("_")[1])
        num_links = node_count - 1
        inter_node_distance_km = total_end_to_end_distance_km / num_links
        base_args = BASE_ARGS + ["--config_file", config_file, "--link_distance_km", str(inter_node_distance_km), "--log_directory", "log/runner-ideal/graph4_link_count_sweep"]
        for correction_mode in correction_modes:
            args = ["--correction_mode", correction_mode]
            tasks.append(command + base_args + args)

    run_tasks(tasks, parallel=21)


# Graph 5 (slow)
def main_graph5_inter_node_distance_sweep() -> None:
    """Run a 6-node sweep while varying only the inter-node distance.

    Args:
        None.

    Returns:
        None.
    """
    tasks = []
    base_dir = Path(__file__).resolve().parent
    command = [sys.executable, str(base_dir / "main.py")]
    base_args = BASE_ARGS + ["--run_duration_ms", "20000.0", "--log_directory", "log/runner-ideal/graph5_inter_node_distance_sweep", "--num_logical_pairs", "1000"]

    inter_node_distances_km = ["1", "5", "10", "20", "30", "40", "50", "60", "70"]
    inter_node_distances_km = ["80", "90", "100"]

    correction_modes = ["none", "cec", "qec"]
    for inter_node_distance_km in inter_node_distances_km:
        for correction_mode in correction_modes:
            args = ["--link_distance_km", inter_node_distance_km, "--correction_mode", correction_mode]
            tasks.append(command + base_args + args)

    run_tasks(tasks, parallel=42)


# Graph 6
def main_graph6_physical_bell_pair_fidelity_sweep() -> None:
    """Run a 6-node sweep while varying only the physical Bell-pair fidelity.

    Args:
        None.

    Returns:
        None.
    """
    tasks = []
    base_dir = Path(__file__).resolve().parent
    command = [sys.executable, str(base_dir / "main.py")]

    log_directory = "log/runner-ideal/graph6_physical_bell_pair_fidelity_sweep"
    base_args = BASE_ARGS + [ "--log_directory", log_directory]

    physical_bell_pair_fidelities = ["0.95", "0.96", "0.97", "0.98", "0.99", "0.995", "1.0"]
    correction_modes = ["none", "cec", "qec"]
    for physical_bell_pair_fidelity in physical_bell_pair_fidelities:
        for correction_mode in correction_modes:
            args = ["--physical_bell_pair_fidelity", physical_bell_pair_fidelity, "--correction_mode", correction_mode]
            tasks.append(command + base_args + args)

    run_tasks(tasks, parallel=21)


# Graph 7
def main_graph7_twoqubit_gate_fidelity_sweep_nice_params() -> None:
    """Run a 5-link two-qubit gate-fidelity sweep with favorable parameters.

    Args:
        None.

    Returns:
        None.
    """
    tasks = []
    base_dir = Path(__file__).resolve().parent
    command = [sys.executable, str(base_dir / "main.py")]

    config_file = "config/standard_configs/line_6_2G.json"
    inter_node_distance_km = 20
    num_logical_pairs = "1000"
    run_duration_ms = "10000.0"
    log_directory = "log/runner-ideal/graph7_twoqubit_gate_fidelity_sweep_nice_params"
    physical_bell_pair_fidelity = "0.9999"
    gate_fidelity = "0.9999"
    measurement_fidelity = "0.9999"
    state_preparation_fidelity = "0.99999"
    idle_t1_sec = "100.0"
    idle_t2_sec = "100.0"
    base_args = [
        "--config_file", config_file,
        "--css_code", "[[7,1,3]]",
        "--target_fidelity", "0.8",
        "--num_logical_pairs", num_logical_pairs,
        "--link_distance_km", str(inter_node_distance_km),
        "--gate_fidelity", gate_fidelity,
        "--measurement_fidelity", measurement_fidelity,
        "--state_preparation_fidelity", state_preparation_fidelity,
        "--physical_bell_pair_fidelity", physical_bell_pair_fidelity,
        "--gate_error_channel", "pauli",
        "--idle_t1_sec", idle_t1_sec,
        "--idle_t2_sec", idle_t2_sec,
        "--ft_prep_mode", "minimal",
        "--idle_pauli_x", "0.05",
        "--idle_pauli_y", "0.05",
        "--idle_pauli_z", "0.9",
        "--run_duration_ms", run_duration_ms,
        "--round_spacing_ms", "1.0",
        "--log_directory", log_directory]

    two_qubit_gate_fidelities = ["0.995", "0.996", "0.997", "0.998", "0.999", "0.9995", "0.9999", "1"]
    correction_modes = ["none", "cec", "qec"]
    for two_qubit_gate_fidelity in two_qubit_gate_fidelities:
        for correction_mode in correction_modes:
            args = ["--two_qubit_gate_fidelity", two_qubit_gate_fidelity, "--correction_mode", correction_mode]
            tasks.append(command + base_args + args)

    run_tasks(tasks, parallel=24)


def main_graph8_ideal_params_distance_sweep() -> None:
    """Run a fixed-20-km/link distance sweep with near-ideal hardware.

    Args:
        None.

    Returns:
        None.
    """
    tasks = []
    base_dir = Path(__file__).resolve().parent
    command = [sys.executable, str(base_dir / "main.py")]

    config_files = [
        "config/standard_configs/line_2_2G.json",
        "config/standard_configs/line_3_2G.json",
        "config/standard_configs/line_6_2G.json",
        "config/standard_configs/line_11_2G.json",
        "config/standard_configs/line_21_2G.json",
        "config/standard_configs/line_51_2G.json"
    ]

    config_files = [
        "config/standard_configs/line_101_2G.json",
        "config/standard_configs/line_151_2G.json",
        "config/standard_configs/line_201_2G.json",
    ]

    inter_node_distance_km = "20.0"
    log_directory = "log/runner-ideal/graph8_ideal_params_distance_sweep"

    base_args = [
        "--css_code", "[[7,1,3]]",
        "--target_fidelity", "0.8",
        "--num_logical_pairs", "1000",
        "--link_distance_km", inter_node_distance_km,
        "--gate_fidelity", "0.9999",
        "--two_qubit_gate_fidelity", "0.9999",
        "--measurement_fidelity", "0.9999",
        "--state_preparation_fidelity", "0.99999",
        "--physical_bell_pair_fidelity", "0.999",
        "--gate_error_channel", "pauli",
        "--idle_t1_sec", "100.0",
        "--idle_t2_sec", "100.0",
        "--ft_prep_mode", "minimal",
        "--idle_pauli_x", "0.05",
        "--idle_pauli_y", "0.05",
        "--idle_pauli_z", "0.9",
        "--run_duration_ms", "10000.0",
        "--round_spacing_ms", "1.0",
        "--log_directory", log_directory,
    ]

    correction_modes = ["none", "cec", "qec"]
    for config_file in config_files:
        for correction_mode in correction_modes:
            args = [
                "--config_file", config_file,
                "--correction_mode", correction_mode,
            ]
            tasks.append(command + base_args + args)

    run_tasks(tasks, parallel=9)


if __name__ == "__main__":

    # Graphs are plotted in plot.ipynb, which reads from the log directories specified in each graph's function. To generate new data, uncomment the desired graph function calls below and run this script.

    # Graph 1
    main_graph1_twoqubit_gate_sweep() 

    # Graph 2
    # main_graph2_data_coherence_sweep()

    # Graph 3 (slow)
    # main_graph3_distance_sweep()

    # Graph 4 (slow)
    # main_graph4_link_count_sweep()

    # Graph 5 (slow)
    # main_graph5_inter_node_distance_sweep()

    # Graph 6
    # main_graph6_physical_bell_pair_fidelity_sweep()

    # Graph 7
    # main_graph7_twoqubit_gate_fidelity_sweep_nice_params()

    # Graph 8 (slow)
    # main_graph8_ideal_params_distance_sweep()

