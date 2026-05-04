import json
import sys
import time
import secrets
import select
from pathlib import Path
from subprocess import PIPE, STDOUT, Popen

# LOG_ROOT = "log/runner_May3rd"
LOG_ROOT = "log/runner_May4th"

CONFIG_FILES = ["config/standard_configs/line_2_2G.json",
                "config/standard_configs/line_3_2G.json",
                "config/standard_configs/line_6_2G.json"]


# ideal parameters
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
        "--run_duration_ms", "100000",
        "--round_spacing_ms", "1",
        "--two_qubit_gate_fidelity", "1",
        "--correction_mode", "cec",
        "--physical_bell_pair_fidelity", "1",
]


def drain_process_lines(process: Popen) -> None:
    """Print all currently available subprocess output lines.

    Args:
        process: Running or completed subprocess.

    Returns:
        None.
    """
    if process.stdout is not None:
        while True:
            ready, _, _ = select.select([process.stdout], [], [], 0)
            if not ready:
                break
            line = process.stdout.readline()
            if line == "":
                break
            line = line.rstrip()
            if line:
                print(line, flush=True)


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
    while len(tasks) > 0 or len(ps) > 0:
        if len(ps) < parallel and len(tasks) > 0:
            task = tasks.pop(0)
            print(task, f"{len(tasks)} still in queue")
            process = Popen(task, stdout=PIPE, stderr=STDOUT, cwd=base_dir, text=True, bufsize=1)
            ps.append((process, task, time.time()))
        else:
            time.sleep(0.05)
            new_ps: list[tuple[Popen, list[str], float]] = []
            for process, task, start_time in ps:
                drain_process_lines(process)
                if process.poll() is None:
                    new_ps.append((process, task, start_time))
                else:
                    drain_process_lines(process)
            ps = new_ps


def test_graph_two_qubit_gate_sweep() -> None:
    """Run the two-qubit-only fidelity sweep.

    Args:
        None.

    Returns:
        None.
    """
    tasks = []
    base_dir = Path(__file__).resolve().parent
    command = [sys.executable, str(base_dir / "main.py")]
    log_directory = f"{LOG_ROOT}/two_qubit_gate_sweep"

    # 20 data points
    two_qubit_gate_fidelities = ["0.99", "0.991", "0.992", "0.993", "0.994", "0.995", "0.996", "0.997", "0.998", "0.999",
                                 "0.9991", "0.9992", "0.9993", "0.9994", "0.9995", "0.9996", "0.9997", "0.9998", "0.9999", "1"]

    for two_qubit_gate_fidelity in two_qubit_gate_fidelities:
        for config_file in CONFIG_FILES:
            args = [
                "--two_qubit_gate_fidelity", two_qubit_gate_fidelity,
                "--config_file", config_file,
                "--log_directory", log_directory,
                "--seed_offset", str(secrets.randbelow(2**31 - 1)),
            ]
            tasks.append(command + BASE_ARGS + args)

    parallel = len(two_qubit_gate_fidelities) * 3
    run_tasks(tasks, parallel=parallel)


def test_graph_one_qubit_gate_sweep() -> None:
    """Run the one-qubit-only fidelity sweep.

    Args:
        None.

    Returns:
        None.
    """
    tasks = []
    base_dir = Path(__file__).resolve().parent
    command = [sys.executable, str(base_dir / "main.py")]
    log_directory = f"{LOG_ROOT}/one_qubit_gate_sweep"

    # 20 data points
    one_qubit_gate_fidelities = ["0.99", "0.991", "0.992", "0.993", "0.994", "0.995", "0.996", "0.997", "0.998", "0.999",
                                 "0.9991", "0.9992", "0.9993", "0.9994", "0.9995", "0.9996", "0.9997", "0.9998", "0.9999", "1"]

    for one_qubit_gate_fidelity in one_qubit_gate_fidelities:
        for config_file in CONFIG_FILES:
            args = [
                "--gate_fidelity", one_qubit_gate_fidelity,
                "--config_file", config_file,
                "--log_directory", log_directory,
                "--seed_offset", str(secrets.randbelow(2**31 - 1)),
            ]
            tasks.append(command + BASE_ARGS + args)

    parallel = len(one_qubit_gate_fidelities) * 3
    run_tasks(tasks, parallel=parallel)


def test_graph_measurement_fidelity_sweep() -> None:
    """Run the measurement-only fidelity sweep.

    Args:
        None.

    Returns:
        None.
    """
    tasks = []
    base_dir = Path(__file__).resolve().parent
    command = [sys.executable, str(base_dir / "main.py")]
    log_directory = f"{LOG_ROOT}/graph_measurement_fidelity_sweep"

    # 20 data points
    measurement_fidelities = ["0.99", "0.991", "0.992", "0.993", "0.994", "0.995", "0.996", "0.997", "0.998", "0.999",
                              "0.9991", "0.9992", "0.9993", "0.9994", "0.9995", "0.9996", "0.9997", "0.9998", "0.9999", "1.0"]

    for measurement_fidelity in measurement_fidelities:
        for config_file in CONFIG_FILES:
            args = [
                "--measurement_fidelity", measurement_fidelity,
                "--config_file", config_file,
                "--log_directory", log_directory,
                "--seed_offset", str(secrets.randbelow(2**31 - 1)),
            ]
            tasks.append(command + BASE_ARGS + args)

    parallel = len(measurement_fidelities) * 3
    run_tasks(tasks, parallel=parallel)


def test_graph_initialization_fidelity_sweep() -> None:
    """Run the initialization-only fidelity sweep.

    Args:
        None.

    Returns:
        None.
    """
    tasks = []
    base_dir = Path(__file__).resolve().parent
    command = [sys.executable, str(base_dir / "main.py")]
    log_directory = f"{LOG_ROOT}/graph_initialization_fidelity_sweep"

    # 20 data points
    initialization_fidelities = ["0.99", "0.991", "0.992", "0.993", "0.994", "0.995", "0.996", "0.997", "0.998", "0.999",
                                 "0.9991", "0.9992", "0.9993", "0.9994", "0.9995", "0.9996", "0.9997", "0.9998", "0.9999", "1.0"]

    for initialization_fidelity in initialization_fidelities:
        for config_file in CONFIG_FILES:
            args = [
                "--initialization_fidelity", initialization_fidelity,
                "--config_file", config_file,
                "--log_directory", log_directory,
                "--seed_offset", str(secrets.randbelow(2**31 - 1)),
            ]
            tasks.append(command + BASE_ARGS + args)

    parallel = min(len(tasks), 40)
    run_tasks(tasks, parallel=parallel)


def test_t2_sweep() -> None:
    """Run the T2 sweep.

    Args:
        None.

    Returns:
        None.
    """
    tasks = []
    base_dir = Path(__file__).resolve().parent
    command = [sys.executable, str(base_dir / "main.py")]

    # 18 data points
    idle_t2_sec_list = ["0.01", "0.02", "0.05", "0.075", "0.1", "0.15", "0.2", "0.35", "0.5", "0.75", 
                        "1", "1.5", "2", "5", "10", "20", "50", "100"]
    log_directory = f"{LOG_ROOT}/t2_sweep"

    for idle_t2_sec in idle_t2_sec_list:
        for config_file in CONFIG_FILES:
            args = [
                "--idle_t2_sec", idle_t2_sec,
                "--config_file", config_file,
                "--log_directory", log_directory,
                "--link_distance_km", "20",
                "--seed_offset", str(secrets.randbelow(2**31 - 1)),
            ]
            tasks.append(command + BASE_ARGS + args)

    parallel = min(len(tasks), 14)
    run_tasks(tasks, parallel=parallel)


def test_graph_physical_bell_pair_fidelity_sweep() -> None:
    """Run a 6-node sweep while varying only the physical Bell-pair fidelity.

    Args:
        None.

    Returns:
        None.
    """
    tasks = []
    base_dir = Path(__file__).resolve().parent
    command = [sys.executable, str(base_dir / "main.py")]
    log_directory = f"{LOG_ROOT}/graph_physical_bell_pair_fidelity_sweep"

    # 22 data points
    physical_bell_pair_fidelities = ["0.9", "0.905", "0.91", "0.915", "0.92", "0.925", "0.93", "0.935", "0.94", "0.945",
                                     "0.95", "0.955", "0.96", "0.965", "0.97", "0.975", "0.98", "0.985", "0.99", "0.995", "0.999", "1.0"]

    for physical_bell_pair_fidelity in physical_bell_pair_fidelities:
        for config_file in CONFIG_FILES:
            args = ["--physical_bell_pair_fidelity", physical_bell_pair_fidelity, 
                    "--config_file", config_file, 
                    "--log_directory", log_directory,
                    "--seed_offset", str(secrets.randbelow(2**31 - 1))]
            tasks.append(command + BASE_ARGS + args)

    parallel = len(physical_bell_pair_fidelities) * 3
    run_tasks(tasks, parallel=parallel)


def test_t2_inter_node_distance_sweep() -> None:
    """Run a joint T2 and inter-node-distance sweep.

    Args:
        None.

    Returns:
        None.
    """
    tasks = []
    base_dir = Path(__file__).resolve().parent
    command = [sys.executable, str(base_dir / "main.py")]
    log_directory = f"{LOG_ROOT}/t2_inter_node_distance_sweep"

    # idle_t2_sec_list = [
    #     "1e-1", "1e0", "5e0", "1e1", "5e1", "1e2", "5e2", "1e3", "5e3", "1e4",
    # ]
    # inter_node_distances_km = [
    #     "1", "5", "10", "25", "50", "100", "250", "500",
    # ]
    remaining_t2_to_distances = {
        "5e2": ["50", "100", "250", "500"],
        "1e3": ["1", "5", "10", "25", "50", "100", "250", "500"],
        "5e3": ["1", "5", "10", "25", "50", "100", "250", "500"],
        "1e4": ["1", "5", "10", "25", "50", "100", "250", "500"],
    }
    config_file = "config/standard_configs/line_2_2G.json"

    for idle_t2_sec, inter_node_distances_km in remaining_t2_to_distances.items():
        for inter_node_distance_km in inter_node_distances_km:
            args = [
                "--num_logical_pairs", "1000",
                "--idle_t1_sec", "1e12",
                "--idle_t2_sec", idle_t2_sec,
                "--link_distance_km", inter_node_distance_km,
                "--config_file", config_file,
                "--log_directory", log_directory,
                "--seed_offset", str(secrets.randbelow(2**31 - 1)),
            ]
            tasks.append(command + BASE_ARGS + args)

    parallel = min(len(tasks),10)
    run_tasks(tasks, parallel=parallel)




if __name__ == "__main__":

    test_t2_inter_node_distance_sweep()
    # test_t2_sweep()
    # test_graph_physical_bell_pair_fidelity_sweep()
    # test_graph_two_qubit_gate_sweep()
    # test_graph_one_qubit_gate_sweep()
    # test_graph_measurement_fidelity_sweep()
    # test_graph_initialization_fidelity_sweep()



"""

plot 1: 6 thresholds
(a) (b) (c)
(d) (e) (f)

plot 2:
the z (6 node 5 link, 20km length)
x axis the z
y axis is the fidelity

plot 3:
x axis is the number of links (each link is 20km)
y axis is the z

pick a proper z, then do plot 4~6

plot 4:
fixed 100km, varying number of links

plot 5:
fix 20km length, then varying number of links

plot 6:
fix number of links, varying link length



do not need the 2026 paramers and near-ideal parameter.




"""
