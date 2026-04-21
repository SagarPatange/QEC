import sys
import time
from pathlib import Path
from subprocess import PIPE, Popen


BASE_ARGS = [
    "--config_file", "config/standard_configs/line_11_2G.json",
    "--css_code", "[[7,1,3]]",
    "--target_fidelity", "0.8",
    "--num_logical_pairs", "200",
    "--link_distance_km", "10.0",
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


def main_node_count_sweep() -> None:
    """Run a node-count sweep at fixed distance and hardware settings.

    Args:
        None.

    Returns:
        None.
    """
    tasks = []
    base_dir = Path(__file__).resolve().parent
    command = [sys.executable, str(base_dir / "main.py")]
    base_args = ["--css_code", "[[7,1,3]]", "--target_fidelity", "0.8", "--num_logical_pairs", "30", "--link_distance_km", "10.0", "--gate_fidelity", "1.0", "--two_qubit_gate_fidelity", "1.0", "--idle_t1_sec", "1000000000000.0", "--idle_t2_sec", "1000000000000.0", "--ft_prep_mode", "minimal", "--idle_pauli_x", "0.05", "--idle_pauli_y", "0.05", "--idle_pauli_z", "0.9", "--run_duration_ms", "100000.0", "--round_spacing_ms", "1.0", "--log_directory", "log/runner/node_count_sweep"]

    config_files = ["config/standard_configs/line_2_2G.json", "config/standard_configs/line_3_2G.json", "config/standard_configs/line_6_2G.json", "config/standard_configs/line_11_2G.json", "config/standard_configs/line_21_2G.json"]  # Sweep topology size with fixed physical-layer settings.
    for config_file in config_files:
        args = ["--config_file", config_file]
        tasks.append(command + base_args + args)

    run_tasks(tasks, parallel=10)


def main_distance_sweep() -> None:
    """Run a distance sweep at fixed node count and hardware settings.

    Args:
        None.

    Returns:
        None.
    """
    tasks = []
    base_dir = Path(__file__).resolve().parent
    command = [sys.executable, str(base_dir / "main.py")]
    base_args = ["--config_file", "config/standard_configs/line_6_2G.json", "--css_code", "[[7,1,3]]", "--target_fidelity", "0.8", "--num_logical_pairs", "30", "--gate_fidelity", "1.0", "--two_qubit_gate_fidelity", "1.0", "--idle_t1_sec", "1000000000000.0", "--idle_t2_sec", "1000000000000.0", "--ft_prep_mode", "minimal", "--idle_pauli_x", "0.05", "--idle_pauli_y", "0.05", "--idle_pauli_z", "0.9", "--round_spacing_ms", "1.0", "--log_directory", "log/runner/distance_sweep"]

    distance_duration_pairs = [(1.0, 1000.0), (10.0, 10000.0), (50.0, 50000.0), (100.0, 100000.0)]  # Pair each distance with a run duration large enough for that regime.
    for link_distance_km, run_duration_ms in distance_duration_pairs:
        args = ["--link_distance_km", str(link_distance_km), "--run_duration_ms", str(run_duration_ms)]
        tasks.append(command + base_args + args)

    run_tasks(tasks, parallel=10)


def main_two_qubit_gate_sweep() -> None:
    """Run a two-qubit gate-fidelity sweep at fixed topology and distance.

    Args:
        None.

    Returns:
        None.
    """
    tasks = []
    base_dir = Path(__file__).resolve().parent
    command = [sys.executable, str(base_dir / "main.py")]
    base_args = ["--config_file", "config/standard_configs/line_6_2G.json", "--css_code", "[[7,1,3]]", "--target_fidelity", "0.8", "--num_logical_pairs", "100", "--link_distance_km", "10.0", "--gate_fidelity", "1.0", "--idle_t1_sec", "1000000000000.0", "--idle_t2_sec", "1000000000000.0", "--ft_prep_mode", "minimal", "--idle_pauli_x", "0.05", "--idle_pauli_y", "0.05", "--idle_pauli_z", "0.9", "--run_duration_ms", "100000.0", "--round_spacing_ms", "1.0", "--log_directory", "log/runner/two_qubit_gate_sweep"]

    two_qubit_gate_fidelities = [1.0, 0.9999, 0.9995, 0.999, 0.998, 0.995]  # Sweep only the two-qubit gate fidelity.
    for two_qubit_gate_fidelity in two_qubit_gate_fidelities:
        args = ["--two_qubit_gate_fidelity", str(two_qubit_gate_fidelity)]
        tasks.append(command + base_args + args)

    run_tasks(tasks, parallel=10)


def main_ft_mode_sweep() -> None:
    """Run an FT-preparation-mode sweep at fixed topology and hardware settings.

    Args:
        None.

    Returns:
        None.
    """
    tasks = []
    base_dir = Path(__file__).resolve().parent
    command = [sys.executable, str(base_dir / "main.py")]
    base_args = ["--config_file", "config/standard_configs/line_6_2G.json", "--css_code", "[[7,1,3]]", "--target_fidelity", "0.8", "--num_logical_pairs", "30", "--link_distance_km", "10.0", "--gate_fidelity", "1.0", "--two_qubit_gate_fidelity", "1.0", "--idle_t1_sec", "1000000000000.0", "--idle_t2_sec", "1000000000000.0", "--idle_pauli_x", "0.05", "--idle_pauli_y", "0.05", "--idle_pauli_z", "0.9", "--run_duration_ms", "100000.0", "--round_spacing_ms", "1.0", "--log_directory", "log/runner/ft_mode_sweep"]

    ft_prep_modes = ["none", "minimal", "standard"]  # Sweep only FT preparation mode.
    for ft_prep_mode in ft_prep_modes:
        args = ["--ft_prep_mode", ft_prep_mode]
        tasks.append(command + base_args + args)

    run_tasks(tasks, parallel=10)


def main_css_code_sweep() -> None:
    """Run a CSS-code sweep at fixed topology and hardware settings.

    Args:
        None.

    Returns:
        None.
    """
    tasks = []
    base_dir = Path(__file__).resolve().parent
    command = [sys.executable, str(base_dir / "main.py")]
    base_args = ["--config_file", "config/standard_configs/line_6_2G.json", "--target_fidelity", "0.8", "--num_logical_pairs", "30", "--link_distance_km", "10.0", "--gate_fidelity", "1.0", "--two_qubit_gate_fidelity", "1.0", "--idle_t1_sec", "1000000000000.0", "--idle_t2_sec", "1000000000000.0", "--ft_prep_mode", "minimal", "--idle_pauli_x", "0.05", "--idle_pauli_y", "0.05", "--idle_pauli_z", "0.9", "--run_duration_ms", "100000.0", "--round_spacing_ms", "1.0", "--log_directory", "log/runner/css_code_sweep"]

    css_codes = ["[[7,1,3]]"]  # Add more CSS codes here when they are available.
    for css_code in css_codes:
        args = ["--css_code", css_code]
        tasks.append(command + base_args + args)

    run_tasks(tasks, parallel=10)


def main_distance_twoq_code_sweep() -> None:
    """Run a sweep over link distance, two-qubit gate fidelity, and CSS code.

    Args:
        None.

    Returns:
        None.
    """
    tasks = []
    base_dir = Path(__file__).resolve().parent
    command = [sys.executable, str(base_dir / "main.py")]
    base_args = ["--config_file", "config/standard_configs/line_6_2G.json", "--target_fidelity", "0.8", "--num_logical_pairs", "200", "--gate_fidelity", "1.0", "--idle_t1_sec", "1000000000000.0", "--idle_t2_sec", "1000000000000.0", "--ft_prep_mode", "minimal", "--idle_pauli_x", "0.05", "--idle_pauli_y", "0.05", "--idle_pauli_z", "0.9", "--round_spacing_ms", "1.0", "--log_directory", "log/runner/distance_twoq_code_sweep"]

    distance_duration_pairs = [(1.0, 1000.0), (10.0, 10000.0), (50.0, 50000.0), (100.0, 100000.0)]
    two_qubit_gate_fidelities = [1.0, 0.9999, 0.9995, 0.999, 0.998, 0.995]
    css_codes = ["[[7,1,3]]"]  # Add more codes here when available.
    for link_distance_km, run_duration_ms in distance_duration_pairs:
        for two_qubit_gate_fidelity in two_qubit_gate_fidelities:
            for css_code in css_codes:
                args = ["--link_distance_km", str(link_distance_km), "--run_duration_ms", str(run_duration_ms), "--two_qubit_gate_fidelity", str(two_qubit_gate_fidelity), "--css_code", css_code]
                tasks.append(command + base_args + args)

    run_tasks(tasks, parallel=10)


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
    base_args = BASE_ARGS + ["--config_file", "config/standard_configs/line_11_2G.json", "--num_logical_pairs", "1000", "--log_directory", "log/runner/graph1_twoqubit_gate_sweep"]

    two_qubit_gate_fidelities = ["0.995", "0.996", "0.997", "0.998", "0.999", "0.9995", "1.0"]
    # two_qubit_gate_fidelities = ["0.996", "0.999", "0.9995", "1.0"]

    correction_modes = ["none", "cec", "qec"]
    for two_qubit_gate_fidelity in two_qubit_gate_fidelities:
        gate_fidelity = min(1.0, (0.9996 / 0.997) * float(two_qubit_gate_fidelity))
        measurement_fidelity = min(1.0, (0.995 / 0.997) * float(two_qubit_gate_fidelity))
        for correction_mode in correction_modes:
            args = [
                "--two_qubit_gate_fidelity", two_qubit_gate_fidelity,
                "--gate_fidelity", f"{gate_fidelity:.6f}",
                "--measurement_fidelity", f"{measurement_fidelity:.6f}",
                "--correction_mode", correction_mode,
            ]
            tasks.append(command + base_args + args)

    run_tasks(tasks, parallel=10)


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
    base_args = BASE_ARGS + ["--config_file", "config/standard_configs/line_11_2G.json", "--num_logical_pairs", "1000", "--run_duration_ms", "700000.0", "--log_directory", "log/runner/graph2_t2_sweep"]

    idle_t2_times = [0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]  # Sweep only T2 in seconds while holding T1 fixed.

    correction_modes = ["none", "cec", "qec"]
    for idle_t2_sec in idle_t2_times:
        for correction_mode in correction_modes:
            args = ["--idle_t2_sec", str(idle_t2_sec), "--correction_mode", correction_mode]
            tasks.append(command + base_args + args)

    run_tasks(tasks, parallel=10)


# Graph 3
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
    # Tuples are (config_file, run_duration_ms).
    config_duration_pairs = [("config/standard_configs/line_2_2G.json", 10000.0), ("config/standard_configs/line_6_2G.json", 50000.0), ("config/standard_configs/line_11_2G.json", 100000.0), ("config/standard_configs/line_26_2G.json", 250000.0), ("config/standard_configs/line_51_2G.json", 500000.0), ("config/standard_configs/line_101_2G.json", 1000000.0)]

    for config_file, run_duration_ms in config_duration_pairs:
        base_args = BASE_ARGS + ["--config_file", config_file, "--num_logical_pairs", "1000", "--run_duration_ms", str(run_duration_ms), "--log_directory", "log/runner/graph3_distance_sweep"]
        for correction_mode in correction_modes:
            args = ["--correction_mode", correction_mode]
            tasks.append(command + base_args + args)

    run_tasks(tasks, parallel=10)


# Graph 4
def main_graph4_link_count_sweep() -> None:
    """Run a link-count sweep at fixed 10 km per elementary link.

    Args:
        None.

    Returns:
        None.
    """
    tasks = []
    base_dir = Path(__file__).resolve().parent
    command = [sys.executable, str(base_dir / "main.py")]
    correction_modes = ["none", "cec", "qec"]
    # Tuples are (config_file, run_duration_ms).
    config_duration_pairs = [("config/standard_configs/line_2_2G.json", 10000.0), ("config/standard_configs/line_3_2G.json", 20000.0), ("config/standard_configs/line_6_2G.json", 50000.0), ("config/standard_configs/line_11_2G.json", 100000.0), ("config/standard_configs/line_21_2G.json", 200000.0), ("config/standard_configs/line_51_2G.json", 500000.0), ("config/standard_configs/line_101_2G.json", 1000000.0)]

    for config_file, run_duration_ms in config_duration_pairs:
        base_args = BASE_ARGS + ["--config_file", config_file, "--num_logical_pairs", "1000", "--run_duration_ms", str(run_duration_ms), "--log_directory", "log/runner/graph4_link_count_sweep"]
        for correction_mode in correction_modes:
            args = ["--correction_mode", correction_mode]
            tasks.append(command + base_args + args)

    run_tasks(tasks, parallel=10)


def main_two_qubit_qec_isolation() -> None:
    """Sweep only two-qubit gate fidelity under otherwise ideal local conditions.

    Args:
        None.

    Returns:
        None.
    """
    tasks = []
    base_dir = Path(__file__).resolve().parent
    command = [sys.executable, str(base_dir / "main.py")]
    base_args = ["--config_file", "config/standard_configs/line_6_2G.json", "--css_code", "[[7,1,3]]", "--target_fidelity", "0.8", "--num_logical_pairs", "200", "--link_distance_km", "10.0", "--gate_fidelity", "1.0", "--measurement_fidelity", "1.0", "--idle_t1_sec", "1e12", "--idle_t2_sec", "1e12", "--ft_prep_mode", "minimal", "--run_duration_ms", "100000.0", "--round_spacing_ms", "1.0", "--log_directory", "log/runner/two_qubit_qec_isolation"]

    two_qubit_gate_fidelities = ["1.0", "0.9999", "0.9995", "0.999", "0.998", "0.995"]  # Sweep only the two-qubit gate fidelity.
    correction_modes = ["none", "cec","qec"]  # Compare the uncorrected baseline against pre-swap QEC.
    for two_qubit_gate_fidelity in two_qubit_gate_fidelities:
        for correction_mode in correction_modes:
            args = ["--two_qubit_gate_fidelity", two_qubit_gate_fidelity, "--correction_mode", correction_mode]
            tasks.append(command + base_args + args)

    run_tasks(tasks, parallel=10)


if __name__ == "__main__":

    # Qubit Overhead Table
    # Uses plot.ipynb, not runner.py.

    # Graph 1
    # main_graph1_twoqubit_gate_sweep()

    # Graph 2
    main_graph2_data_coherence_sweep()

    # Graph 3
    # main_graph3_distance_sweep()

    # Graph 4
    # main_graph4_link_count_sweep()

    # Existing sweeps
    # main_two_qubit_gate_sweep()
    # main_distance_sweep()
    # main_distance_twoq_code_sweep()
    # main_node_count_sweep()
    # main_css_code_sweep()
