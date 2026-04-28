import sys
import time
from pathlib import Path
from subprocess import PIPE, Popen


# ideal parameters
BASE_ARGS = [
        "--config_file", "config/standard_configs/line_2_2G.json",
        "--css_code", "[[7,1,3]]",
        "--target_fidelity", "0.8",
        "--num_logical_pairs", "10000",
        "--link_distance_km", "1",
        "--gate_fidelity", "1",
        "--measurement_fidelity", "1",
        "--state_preparation_fidelity", "1",
        "--gate_error_channel", "pauli",
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


def test_graph_two_qubit_gate_sweep() -> None:
    """Run a minimal Graph 3 sweep over two-qubit gate fidelity.

    Args:
        None.

    Returns:
        None.
    """
    tasks = []
    base_dir = Path(__file__).resolve().parent
    command = [sys.executable, str(base_dir / "main.py")]

    two_qubit_gate_fidelities = ["0.995", "0.996", "0.997", "0.998", "0.999", "0.9995", "1"]

    for two_qubit_gate_fidelity in two_qubit_gate_fidelities:
        args = ["--two_qubit_gate_fidelity", two_qubit_gate_fidelity, "--log_directory", "log/runner-test/graph_two_qubit_gate_sweep"]
        tasks.append(command + BASE_ARGS + args)

    run_tasks(tasks, parallel=28)


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
        args = ["--config_file", config_file, "--log_directory", f"log/runner-test/graph_distance_sweep_2q_gate_{two_qubit_gate_fidelity}", 
                "--two_qubit_gate_fidelity", two_qubit_gate_fidelity, "--correction_mode", correction_mode]
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

    log_directory = "log/runner-test/graph_physical_bell_pair_fidelity_sweep"
    physical_bell_pair_fidelities = ["0.96", "0.97", "0.98", "0.99", "0.995", "0.999", "1.0"]

    for physical_bell_pair_fidelity in physical_bell_pair_fidelities:
            args = ["--physical_bell_pair_fidelity", physical_bell_pair_fidelity, "--log_directory", log_directory]
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
        args = ["--config_file", config_file, "--log_directory", f"log/runner-test/graph_distance_sweep_physical_bellpair_{physical_bell_pair_fidelity}", 
                "--physical_bell_pair_fidelity", physical_bell_pair_fidelity]
        tasks.append(command + BASE_ARGS + args)

    run_tasks(tasks, parallel=18)



if __name__ == "__main__":

    # Test sweeps

    # test_graph_two_qubit_gate_sweep()

    test_graph_distance_sweep_two_qubit_gate_997()

    # test_graph_physical_bell_pair_fidelity_sweep()

    # test_graph_distance_sweep_physical_bellpair_99()

