import json
import secrets
import select
import sys
import time
from pathlib import Path
from subprocess import PIPE, STDOUT, Popen


LOG_ROOT = "log/runner_z"

CONFIG_FILE_BY_TOPOLOGY = {
    "line_2": "config/standard_configs/line_2_2G.json",
    "line_3": "config/standard_configs/line_3_2G.json",
    "line_6": "config/standard_configs/line_6_2G.json",
}

Z_PARAM_GRID_FILE = "config/generated_configs/z_plot_param_grid.json"

Z_BASE_ARGS = [
    "--css_code", "[[7,1,3]]",
    "--target_fidelity", "0.8",
    "--num_logical_pairs", "10000",
    "--link_distance_km", "20",
    "--gate_error_channel", "depolarize",
    "--idle_error_channel", "pauli",
    "--idle_t1_sec", "100",
    "--ft_prep_mode", "minimal",
    "--idle_pauli_x", "0.05",
    "--idle_pauli_y", "0.05",
    "--idle_pauli_z", "0.9",
    "--run_duration_ms", "1000",
    "--round_spacing_ms", "1",
    "--correction_mode", "cec",
]


def drain_process_lines(process: Popen) -> None:
    """Print all currently available subprocess output lines.

    Args:
        process: Running or completed subprocess.

    Returns:
        None.
    """
    if process.stdout is None:
        return

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
    processes: list[tuple[Popen, list[str], float]] = []

    while tasks or processes:
        if len(processes) < parallel and tasks:
            task = tasks.pop(0)
            print(task, f"{len(tasks)} still in queue")
            process = Popen(
                task,
                stdout=PIPE,
                stderr=STDOUT,
                cwd=base_dir,
                text=True,
                bufsize=1,
            )
            processes.append((process, task, time.time()))
            continue

        time.sleep(0.05)
        remaining: list[tuple[Popen, list[str], float]] = []
        for process, task, start_time in processes:
            drain_process_lines(process)
            if process.poll() is None:
                remaining.append((process, task, start_time))
            else:
                drain_process_lines(process)
        processes = remaining


def load_z_param_rows(grid_path: Path) -> list[dict[str, float | str]]:
    """Load z-grid rows from the generated JSON file.

    Args:
        grid_path: Path to the generated z grid JSON file.

    Returns:
        Grid rows as a list of dictionaries.
    """
    if not grid_path.exists():
        raise FileNotFoundError(f"Missing grid file: {grid_path}")

    with grid_path.open("r", encoding="utf-8") as file:
        param_rows = json.load(file)

    if not isinstance(param_rows, list):
        raise RuntimeError("z grid JSON must contain a list of parameter rows")

    return param_rows


def run_z_param_grid() -> None:
    """Run the combined z-parameter grid.

    Args:
        None.

    Returns:
        None.
    """
    tasks: list[list[str]] = []
    base_dir = Path(__file__).resolve().parent
    command = [sys.executable, str(base_dir / "main.py")]
    log_directory = f"{LOG_ROOT}/z_param_grid"
    grid_path = base_dir / Z_PARAM_GRID_FILE
    param_rows = load_z_param_rows(grid_path)

    for param_row in param_rows:
        topology = str(param_row["topology"])
        config_file = CONFIG_FILE_BY_TOPOLOGY[topology]
        args = [
            "--gate_fidelity", str(param_row["gate_fidelity"]),
            "--two_qubit_gate_fidelity", str(param_row["two_qubit_gate_fidelity"]),
            "--measurement_fidelity", str(param_row["measurement_fidelity"]),
            "--initialization_fidelity", str(param_row["initialization_fidelity"]),
            "--physical_bell_pair_fidelity", str(param_row["physical_bell_pair_fidelity"]),
            "--idle_t2_sec", str(param_row["idle_t2_sec"]),
            "--config_file", config_file,
            "--log_directory", log_directory,
            "--seed_offset", str(secrets.randbelow(2**31 - 1)),
        ]
        tasks.append(command + Z_BASE_ARGS + args)

    parallel = len(tasks)
    run_tasks(tasks, parallel=parallel)


if __name__ == "__main__":
    run_z_param_grid()
