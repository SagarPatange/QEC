import time
import sys
from pathlib import Path
from subprocess import PIPE, Popen


def set_config_file(args: list[str], config_file: str) -> list[str]:
    return args + ["--config_file", config_file]


def set_css_code(args: list[str], css_code: str) -> list[str]:
    return args + ["--css_code", css_code]


def set_log_directory(args: list[str], log_directory: str) -> list[str]:
    return args + ["--log_directory", log_directory]


def set_start_time_s(args: list[str], start_time_s: float) -> list[str]:
    return args + ["--start_time_s", str(start_time_s)]


def set_window_time_ms(args: list[str], window_time_ms: float) -> list[str]:
    return args + ["--window_time_ms", str(window_time_ms)]


def set_target_fidelity(args: list[str], target_fidelity: float) -> list[str]:
    return args + ["--target_fidelity", str(target_fidelity)]


def set_num_logical_pairs(args: list[str], num_logical_pairs: int) -> list[str]:
    return args + ["--num_logical_pairs", str(num_logical_pairs)]


def get_output(process: Popen) -> None:
    stderr = process.stderr.readlines()
    if stderr:
        for line in stderr:
            print(line.decode().rstrip())
    stdout = process.stdout.readlines()
    if stdout:
        for line in stdout:
            print(line.decode().rstrip())


def main() -> None:
    """Run batches of logical-pair experiments.

    Args:
        None.

    Returns:
        None.
    """
    tasks = []
    base_dir = Path(__file__).resolve().parent
    command = [sys.executable, str(base_dir / "main.py")]
    base_args = ["--log_directory", "log/runner"]

    config_files = ["config/standard_configs/line_2_2G.json", "config/standard_configs/line_3_2G.json", "config/standard_configs/line_5_2G.json", "config/standard_configs/line_10_2G.json", "config/standard_configs/line_20_2G.json"]
    css_codes = ["[[7,1,3]]"]
    target_fidelities = [0.8]
    num_logical_pairs_list = [30]
    link_distances_km = [1.0, 10.0]
    gate_fidelities = [1.0]
    two_qubit_gate_fidelities = [1.0]
    idle_data_coherence_times = [1e12]
    idle_comm_coherence_times = [1e12]
    ft_prep_modes = ["minimal"]
    idle_pauli_weight_sets = [(0.05, 0.05, 0.9)]
    seeds = [0]

    for config_file in config_files:
        for css_code in css_codes:
            for target_fidelity in target_fidelities:
                for num_logical_pairs in num_logical_pairs_list:
                    for link_distance_km in link_distances_km:
                        for gate_fidelity in gate_fidelities:
                            for two_qubit_gate_fidelity in two_qubit_gate_fidelities:
                                for idle_data_coherence_time_sec in idle_data_coherence_times:
                                    for idle_comm_coherence_time_sec in idle_comm_coherence_times:
                                        for ft_prep_mode in ft_prep_modes:
                                            for idle_pauli_x, idle_pauli_y, idle_pauli_z in idle_pauli_weight_sets:
                                                for seed in seeds:
                                                    args = list(base_args)
                                                    args = set_config_file(args, config_file)
                                                    args = set_css_code(args, css_code)
                                                    args = set_target_fidelity(args, target_fidelity)
                                                    args = set_num_logical_pairs(args, num_logical_pairs)
                                                    args = set_log_directory(args, f"log/runner/seed_{seed}")
                                                    args += ["--link_distance_km", str(link_distance_km)]
                                                    args += ["--gate_fidelity", str(gate_fidelity)]
                                                    args += ["--two_qubit_gate_fidelity", str(two_qubit_gate_fidelity)]
                                                    args += ["--idle_data_coherence_time_sec", str(idle_data_coherence_time_sec)]
                                                    args += ["--idle_comm_coherence_time_sec", str(idle_comm_coherence_time_sec)]
                                                    args += ["--ft_prep_mode", ft_prep_mode]
                                                    args += ["--idle_pauli_x", str(idle_pauli_x), "--idle_pauli_y", str(idle_pauli_y), "--idle_pauli_z", str(idle_pauli_z)]
                                                    tasks.append(command + args)

    parallel = 12
    processes = []

    while len(tasks) > 0 or len(processes) > 0:
        if len(processes) < parallel and len(tasks) > 0:
            task = tasks.pop(0)
            print(task, f"{len(tasks)} still in queue")
            processes.append(Popen(task, stdout=PIPE, stderr=PIPE, cwd=base_dir))
        else:
            time.sleep(0.05)

        new_processes = []
        for process in processes:
            if process.poll() is None:
                new_processes.append(process)
            else:
                get_output(process)
        processes = new_processes


if __name__ == "__main__":
    main()
