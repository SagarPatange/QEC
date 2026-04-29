import sys
import time
from pathlib import Path
from subprocess import PIPE, Popen


# ideal parameters
BASE_ARGS = [
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

CONFIG_FILES = [
    "config/standard_configs/line_2_2G.json",
    "config/standard_configs/line_3_2G.json",
    "config/standard_configs/line_6_2G.json"
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


def t2_sweep():
    tasks = []
    command = ["python", "main.py"]
    # 13 data points between 0.01 and 100, spaced logarithmically
    idle_t2_sec_list = ["0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "1", "2", "5", "10", "20", "50", "100"]
    log_directory = "log/runner-threshold/t2_sweep"
    for idle_t2_sec in idle_t2_sec_list:
        for config_file in CONFIG_FILES:
            args = ["--idle_t2_sec", idle_t2_sec,
                    "--config_file", config_file,
                    "--log_directory", log_directory]
            tasks.append(command + BASE_ARGS + args)
    run_tasks(tasks, parallel=20)


def one_qubit_gate_sweep():
    tasks = []
    command = ["python", "main.py"]
    # 17 data points between 0.996 and 1, spaced more closely near 1
    one_qubit_gate_fidelities = ["0.996", "0.9965", "0.997", "0.9975", "0.998", "0.9985", "0.999", 
                                 "0.9991", "0.9992", "0.9993", "0.9994", "0.9995", "0.9996", "0.9997", "0.9998", "0.9999", "1"]
    log_directory = "log/runner-threshold/one_qubit_gate_sweep"
    for one_qubit_gate_fidelity in one_qubit_gate_fidelities:
        for config_file in CONFIG_FILES:
            args = ["--gate_fidelity", one_qubit_gate_fidelity,
                    "--config_file", config_file,
                    "--log_directory", log_directory]
            tasks.append(command + BASE_ARGS + args)
    run_tasks(tasks, parallel=20)


def two_qubit_gate_sweep():
    tasks = []
    command = ["python", "main.py"]
    # 17 data points between 0.996 and 1, spaced more closely near 1
    two_qubit_gate_fidelities = ["0.996", "0.9965", "0.997", "0.9975", "0.998", "0.9985", "0.999", 
                                 "0.9991", "0.9992", "0.9993", "0.9994", "0.9995", "0.9996", "0.9997", "0.9998", "0.9999", "1"]
    log_directory = "log/runner-threshold/two_qubit_gate_sweep"
    for two_qubit_gate_fidelity in two_qubit_gate_fidelities:
        for config_file in CONFIG_FILES:
            args = ["--two_qubit_gate_fidelity", two_qubit_gate_fidelity,
                    "--config_file", config_file,
                    "--log_directory", log_directory]
            tasks.append(command + BASE_ARGS + args)
    run_tasks(tasks, parallel=20)


def physical_bell_pair_sweep():
    tasks = []
    command = ["python", "main.py"]
    # 13 data points between 0.9 and 1
    physical_bell_pair_fidelities = ["0.9", "0.91", "0.92", "0.93", "0.94", "0.95", "0.96", "0.97", "0.98", "0.99", "0.995", "0.999", "1.0"]
    log_directory = "log/runner-threshold/graph_physical_bell_pair_fidelity_sweep"
    for physical_bell_pair_fidelity in physical_bell_pair_fidelities:
        for config_file in CONFIG_FILES:
            args = ["--physical_bell_pair_fidelity", physical_bell_pair_fidelity,
                    "--config_file", config_file,
                    "--log_directory", log_directory]
            tasks.append(command + BASE_ARGS + args)
    run_tasks(tasks, parallel=15)


def measurement_sweep():
    tasks = []
    command = ["python", "main.py"]
    # 17 data points between 0.996 and 1, spaced more closely near 1
    measurement_fidelities = ["0.996", "0.9965", "0.997", "0.9975", "0.998", "0.9985", "0.999", 
                              "0.9991", "0.9992", "0.9993", "0.9994", "0.9995", "0.9996", "0.9997", "0.9998", "0.9999", "1.0"]
    log_directory = "log/runner-threshold/graph_measurement_fidelity_sweep"
    for measurement_fidelity in measurement_fidelities:
        for config_file in CONFIG_FILES:
            args = ["--measurement_fidelity", measurement_fidelity,
                    "--config_file", config_file,
                    "--log_directory", log_directory]
            tasks.append(command + BASE_ARGS + args)
    run_tasks(tasks, parallel=15)


def state_preparation_sweep():
    tasks = []
    command = ["python", "main.py"]
    # 17 data points between 0.996 and 1, spaced more closely near 1
    state_preparation_fidelities = ["0.996", "0.9965", "0.997", "0.9975", "0.998", "0.9985", "0.999", 
                                     "0.9991", "0.9992", "0.9993", "0.9994", "0.9995", "0.9996", "0.9997", "0.9998", "0.9999", "1.0"]
    log_directory = "log/runner-threshold/graph_state_preparation_fidelity_sweep"
    for state_preparation_fidelity in state_preparation_fidelities:
        for config_file in CONFIG_FILES:
            args = ["--state_preparation_fidelity", state_preparation_fidelity,
                    "--config_file", config_file,
                    "--log_directory", log_directory]
            tasks.append(command + BASE_ARGS + args)
    run_tasks(tasks, parallel=15)


if __name__ == "__main__":

    """
    Find the threshold of a parameter while other parameters are set to perfect
    A large figure with 6 subplots, arranged in a 2x3 grid:
    (a) (b) (c)
    (d) (e) (f)
    Each subplot has three lines, one for each of the three configurations (line_2_2G, line_3_2G, line_6_2G).
    """

    # t2_sweep()                  # subfigure (a) sweep over T2 time
    # one_qubit_gate_sweep()      # subfigure (b) sweep over one-qubit gate fidelity
    # two_qubit_gate_sweep()      # subfigure (c) sweep over two-qubit gate fidelity
    # physical_bell_pair_sweep()  # subfigure (d) sweep over physical Bell pair fidelity
    # measurement_sweep()         # subfigure (e) sweep over measurement fidelity
    state_preparation_sweep()   # subfigure (f) sweep over state preparation fidelity
