import sys
import time
import secrets
import select
from pathlib import Path
from subprocess import PIPE, STDOUT, Popen

LOG_ROOT = "log/runner_April30th"

CONFIG_FILES = [
        "config/standard_configs/line_2_2G.json",
        # "config/standard_configs/line_3_2G.json",
        # "config/standard_configs/line_6_2G.json",
]

# ideal parameters
BASE_ARGS = [
        "--css_code", "[[7,1,3]]",
        "--target_fidelity", "0.8",
        "--num_logical_pairs", "400",
        "--link_distance_km", "0.001",
        "--gate_fidelity", "1",
        "--measurement_fidelity", "1",
        "--initialization_fidelity", "1",
        "--gate_error_channel", "pauli",
        "--pauli_1q_weights", "0.05", "0.05", "0.9",
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

    run_tasks(tasks, parallel=15)


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

    # one_qubit_gate_fidelities = ["0.99"]
    one_qubit_gate_fidelities = ["0.99", "0.991", "0.992", "0.993", "0.994", "0.995", "0.996", "0.997", "0.998", "0.999",
                                 "0.9991", "0.9992", "0.9993", "0.9994", "0.9995", "0.9996", "0.9997", "0.9998", "0.9999", "1"]
    # one_qubit_gate_fidelities = ["0.99", "0.991", "0.992", "0.993", "0.994", "0.995", "0.996", "0.997", "0.998", "0.999","1"]


    for one_qubit_gate_fidelity in one_qubit_gate_fidelities:
        for config_file in CONFIG_FILES:
            args = [
                "--gate_fidelity", one_qubit_gate_fidelity,
                "--config_file", config_file,
                "--log_directory", log_directory,
                "--seed_offset", str(secrets.randbelow(2**31 - 1)),
            ]
            tasks.append(command + BASE_ARGS + args)

    run_tasks(tasks, parallel=10)


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

    run_tasks(tasks, parallel=10)


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

    initialization_fidelities = ["0.996", "0.9965", "0.997", "0.9975", "0.998", "0.9985", "0.999",
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

    run_tasks(tasks, parallel=15)


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
    idle_t2_sec_list = ["0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "1", "2", "5", "10", "20", "50", "100"]
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

    run_tasks(tasks, parallel=9)


def test_graph_distance_sweep_two_qubit_gate_997() -> None:
    """Run one distance sweep for both latency and fidelity plots.

    Args:
        None.

    Returns:
        None.
    """
    tasks = []
    base_dir = Path(__file__).resolve().parent
    command = [sys.executable, str(base_dir / "main.py")]
    config_duration_pairs = [("config/standard_configs/line_2_2G.json"), 
                             ("config/standard_configs/line_3_2G.json"), 
                             ("config/standard_configs/line_6_2G.json")]
    two_qubit_gate_fidelity = "0.997"
    correction_mode = "cec"

    for config_file in config_duration_pairs:
        args = ["--config_file", config_file, "--log_directory", f"{LOG_ROOT}/graph_distance_sweep_2q_gate_{two_qubit_gate_fidelity}",
                "--two_qubit_gate_fidelity", two_qubit_gate_fidelity, "--correction_mode", correction_mode,
                "--seed_offset", str(secrets.randbelow(2**31 - 1))]
        tasks.append(command + BASE_ARGS + args)

    run_tasks(tasks, parallel=18)


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
    physical_bell_pair_fidelities = ["0.96", "0.97", "0.98", "0.99", "0.995", "0.999", "1.0"]

    for physical_bell_pair_fidelity in physical_bell_pair_fidelities:
            args = ["--physical_bell_pair_fidelity", physical_bell_pair_fidelity, "--log_directory", log_directory,
                    "--seed_offset", str(secrets.randbelow(2**31 - 1))]
            tasks.append(command + BASE_ARGS + args)

    run_tasks(tasks, parallel=21)



def test_graph_distance_sweep_physical_bellpair_99() -> None:
    """Run one distance sweep for both latency and fidelity plots.

    Args:
        None.

    Returns:
        None.
    """
    tasks = []
    base_dir = Path(__file__).resolve().parent
    command = [sys.executable, str(base_dir / "main.py")]
    config_duration_pairs = [("config/standard_configs/line_2_2G.json"), 
                             ("config/standard_configs/line_3_2G.json"), 
                             ("config/standard_configs/line_6_2G.json")]
    physical_bell_pair_fidelity = "0.99"

    for config_file in config_duration_pairs:
        args = ["--config_file", config_file, "--log_directory", f"{LOG_ROOT}/graph_distance_sweep_physical_bellpair_{physical_bell_pair_fidelity}",
                "--physical_bell_pair_fidelity", physical_bell_pair_fidelity,
                "--seed_offset", str(secrets.randbelow(2**31 - 1))]
        tasks.append(command + BASE_ARGS + args)

    run_tasks(tasks, parallel=18)



if __name__ == "__main__":

    # Test sweeps

    # test_graph_measurement_fidelity_sweep()

    test_graph_one_qubit_gate_sweep()

    # test_graph_initialization_fidelity_sweep()

    # test_graph_two_qubit_gate_sweep()

    # test_graph_physical_bell_pair_fidelity_sweep()

    # test_graph_distance_sweep_physical_bellpair_99()
