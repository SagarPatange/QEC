"""Sequential cheating runner -- drop-in replacement for main_test.py.

Same per-pair physics and same `config_file` calling convention as main_test.py:

  * `build_n_node_params()` returns the same default-parameter dictionary.
  * `n_node_logical_pair_cheating(...)` mirrors
    `n_node_logical_pair_with_app(...)` -- accepts the same kwargs (`verbose`,
    `config_file`, `css_code`, `params`) and returns a results dict with a
    `"metrics"` payload shaped like the SeQUeNCe-backed version.

The only difference is what runs under the hood: the SeQUeNCe event-driven
simulation, classical messaging, BSM nodes, and per-circuit tableau merges
are all skipped. Per-shot work goes straight through `cheating.utilities`
on stim, with the *same* noise model, encoder, FT-prep, TCNOT, encoded
swapping, and endpoint recovery as RequestLogicalPairApp / TeleportedCNOT /
QREProtocol / css_codes use. Result statistics match the non-cheating
pipeline within Monte Carlo error of ~1/sqrt(N).

Topology is read from the config file (router count -> N), so any
`config/standard_configs/line_*_2G.json` file works:

    python cheating/main_test_cheating.py
    CHEATING_CONFIG_FILE=config/standard_configs/line_6_2G.json python cheating/main_test_cheating.py
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from utilities import simulate_logical_pairs_nnode  # noqa: E402  -- intentional after sys.path tweak


def resolve_config_path(config_file: str) -> str:
    """Resolve a config file path relative to the project root.

    main_test.py anchors `config_file` at the directory it lives in. This
    cheating module lives one level deeper (`cheating/`), so anchor at
    `_HERE.parent` to match the same on-disk layout.
    """
    config_path = Path(config_file)
    if not config_path.is_absolute():
        config_path = _HERE.parent / config_file
    return str(config_path)


def build_n_node_params() -> dict[str, object]:
    """Default run parameters; verbatim mirror of main_test.build_n_node_params (lines 26-58)."""
    return {
        "log_level": "WARNING",
        "start_time_s": 1.0,
        "run_duration_ms": 1000.0,
        "round_spacing_ms": 1.0,
        "correction_mode": "cec",
        "target_fidelity": 0.8,
        "num_logical_pairs": 1000,
        "link_distance_km": 1,
        "gate_fidelity": 1,
        "two_qubit_gate_fidelity": 0.995,
        "measurement_fidelity": 1,
        "state_preparation_fidelity": 1,
        "gate_error_channel": "depolarize",
        "pauli_1q_weights": None,
        "pauli_2q_weights": None,
        "idle_t1_sec": 1e12,
        "idle_t2_sec": 1e12,
        "ft_prep_mode": "minimal",
        "idle_pauli_x": 0.05,
        "idle_pauli_y": 0.05,
        "idle_pauli_z": 0.90,
        "physical_bell_pair_fidelity": 1,
    }


def count_routers(config_file: str) -> int:
    """Read the config file and return the number of QuantumRouter nodes (= N)."""
    resolved = resolve_config_path(config_file)
    with open(resolved, "r", encoding="utf-8") as fh:
        config = json.load(fh)
    routers = [n for n in config.get("nodes", []) if n.get("type") == "QuantumRouter"]
    if len(routers) < 2:
        raise ValueError(f"{config_file}: expected at least 2 QuantumRouter nodes, got {len(routers)}")
    return len(routers)


def _read_ft_max_retries_from_config(config_file: str) -> int:
    """Read ft_max_retries from the first QuantumRouter in the config (defaults to 3).

    The reference pipeline reads this per-node from the JSON config; the standard
    configs (line_*_2G.json) all use 3. Mirroring that here keeps default-behavior
    consistent between cheating and reference.
    """
    resolved = resolve_config_path(config_file)
    with open(resolved, "r", encoding="utf-8") as fh:
        config = json.load(fh)
    for node in config.get("nodes", []):
        if node.get("type") == "QuantumRouter" and "ft_max_retries" in node:
            return int(node["ft_max_retries"])
    return 3


def _params_for_simulator(params: dict[str, object], config_file: str) -> dict[str, object]:
    """Project the main_test param dict into the subset simulate_logical_pairs_nnode consumes."""
    return {
        "gate_fidelity": float(params["gate_fidelity"]),
        "two_qubit_gate_fidelity": float(params["two_qubit_gate_fidelity"]),
        "measurement_fidelity": float(params["measurement_fidelity"]),
        "state_preparation_fidelity": float(params["state_preparation_fidelity"]),
        "physical_bell_pair_fidelity": float(params["physical_bell_pair_fidelity"]),
        "gate_error_channel": str(params["gate_error_channel"]),
        "pauli_1q_weights": params["pauli_1q_weights"],
        "pauli_2q_weights": params["pauli_2q_weights"],
        "ft_prep_mode": str(params["ft_prep_mode"]),
        "ft_max_retries": _read_ft_max_retries_from_config(config_file),
        "correction_mode": str(params["correction_mode"]),
        "link_distance_km": float(params["link_distance_km"]),
    }


def n_node_logical_pair_cheating(
    verbose: bool = False,
    config_file: str = "config/standard_configs/line_2_2G.json",
    css_code: str = "[[7,1,3]]",
    params: dict[str, object] | None = None,
    seed: int | None = None,
) -> dict[str, object]:
    """Cheating drop-in for main_test.n_node_logical_pair_with_app.

    Args:
        verbose: Print a Run Summary block when True (matches main_test).
        config_file: Topology config; only the count of QuantumRouter nodes is
            consumed (defines N for the linear chain). Other config fields
            (channel distances, BSM nodes, etc.) are inherent to the SeQUeNCe
            pipeline and have no effect on the cheating pair statistics.
        css_code: CSS-code label for the metrics dict (Steane is hard-coded).
        params: Run-parameter overrides (see build_n_node_params for the full
            schema). Same keys as main_test.
        seed: Optional reproducibility seed; None -> non-deterministic.

    Returns:
        {"metrics": {...}} with the same key set as main_test's metrics dict.
        Latency, throughput, per-link logical fidelities, and physical fidelity
        are returned as NaN because they're properties of the bypassed
        SeQUeNCe event simulation, not of the per-pair quantum statistics.
    """
    merged = build_n_node_params()
    if params is not None:
        merged.update(params)
    params = merged

    num_nodes = count_routers(config_file)
    num_logical_pairs = int(params["num_logical_pairs"])

    out = simulate_logical_pairs_nnode(
        num_nodes=num_nodes,
        num_pairs=num_logical_pairs,
        params=_params_for_simulator(params, config_file),
        seed=seed,
    )

    completed = int(out["num_logical_pairs_completed"])
    requested = int(out["num_logical_pairs_requested"])
    avg = float(out["avg_end_to_end_logical_corrected"])

    end_to_end_rows = [
        {
            "run_id": i,
            "fidelity": float(f),
            "fidelity_raw": float("nan"),
            "fidelity_corrected": float(f),
            "latency_ps": float("nan"),
            "latency_s": float("nan"),
            "throughput_pairs_per_s": float("nan"),
        }
        for i, f in enumerate(out["per_shot_fidelities"])
        if np.isfinite(float(f))
    ]

    metrics = {
        "css_code": css_code,
        "config_file": config_file,
        "gate_fidelity": float(params["gate_fidelity"]),
        "two_qubit_gate_fidelity": float(params["two_qubit_gate_fidelity"]),
        "num_nodes": num_nodes,
        "num_links": num_nodes - 1,
        "num_logical_pairs_requested": requested,
        "num_logical_pairs_completed": completed,
        "link_rows": [],
        "end_to_end_rows": end_to_end_rows,
        "avg_initial_phys": float("nan"),
        "avg_link_logical": float("nan"),
        "avg_end_to_end_logical": avg,
        "avg_end_to_end_logical_raw": float("nan"),
        "avg_end_to_end_logical_corrected": avg,
        "avg_latency_ps": float("nan"),
        "avg_latency_s": float("nan"),
        "avg_throughput_pairs_per_s": float("nan"),
    }

    if verbose:
        _print_summary(metrics)
    return {"metrics": metrics}


def _print_summary(metrics: dict[str, object]) -> None:
    """Print a Run Summary block matching main_test._print_metrics_summary (lines 712-738)."""
    print("\n=== Run Summary (cheating) ===")
    print(
        f"code={metrics['css_code']} | nodes={metrics['num_nodes']} | "
        f"links={metrics['num_links']} | gate_fid={metrics['gate_fidelity']:.4f} | "
        f"twoq_fid={metrics['two_qubit_gate_fidelity']:.4f}"
    )
    print(
        f"logical_pairs: requested={metrics['num_logical_pairs_requested']} "
        f"completed={metrics['num_logical_pairs_completed']}"
    )
    if not np.isnan(metrics["avg_end_to_end_logical_corrected"]):
        print(
            f"Avg end-to-end logical fidelity (corrected): "
            f"{metrics['avg_end_to_end_logical_corrected']:.6f}"
        )
    print(
        "Note: latency / throughput / per-link logical fidelities are unavailable "
        "in cheating mode (no SeQUeNCe event simulation)."
    )


if __name__ == "__main__":
    start = time.time()
    print("main_test_cheating start ...")

    # Env overrides (intended for quick smoke tests; same convention as the parallel runner).
    #   CHEATING_CONFIG_FILE - topology config (default line_2_2G.json)
    #   CHEATING_NUM_PAIRS   - logical pairs to simulate (default 1000)
    #   CHEATING_BASE_SEED   - reproducibility seed (default time-based / nondeterministic)
    config_file = os.environ.get("CHEATING_CONFIG_FILE", "config/standard_configs/line_2_2G.json")
    num_pairs = int(os.environ.get("CHEATING_NUM_PAIRS", "1000"))
    seed_env = os.environ.get("CHEATING_BASE_SEED")
    seed = int(seed_env) if seed_env is not None else None

    n_node_logical_pair_cheating(
        verbose=True,
        config_file=config_file,
        params={"num_logical_pairs": num_pairs},
        seed=seed,
    )

    end = time.time()
    print(f"\nMain execution time: {end - start:.2f} seconds")
