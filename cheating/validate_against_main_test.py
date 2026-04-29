"""Cross-check the cheating simulator against main_test.py at one fidelity point.

Usage:
    cd <repo>
    python cheating/validate_against_main_test.py [-n 200] [-f 0.99] [-c config/...json]

Runs both pipelines for `-n` logical pairs at `two_qubit_gate_fidelity = -f`,
on the topology in `-c` (default line_2_2G.json; works for any line_N_2G.json
with N >= 2), prints both averages with 1-sigma binomial uncertainties, and
reports how many sigmas they differ. They should agree to within ~3 sigma;
differ by much more and the cheating impl has drifted from the reference.

Cost: main_test takes roughly 1.5-2 s per logical pair for line_2 and grows
with N. N=200 pairs at line_2 means ~5-7 minutes for the reference run; the
cheating run is finished in seconds. Pick smaller N (e.g. 50) for a fast
smoke check, especially for line_6 and longer chains.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

# Cheating utilities live in this same directory.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
# main_test.py lives one level up.
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _binomial_sigma(p: float, n: int) -> float:
    if n <= 0 or not (0.0 < p < 1.0):
        return 0.0
    return math.sqrt(p * (1.0 - p) / n)


def _count_routers(config_file: str) -> int:
    import json

    config_path = Path(config_file)
    if not config_path.is_absolute():
        config_path = _REPO / config_file
    with open(config_path, "r", encoding="utf-8") as fh:
        config = json.load(fh)
    return sum(1 for n in config.get("nodes", []) if n.get("type") == "QuantumRouter")


def _run_main_test(num_pairs: int, two_qubit_gate_fidelity: float, config_file: str,
                   run_duration_ms: float | None = None,
                   round_spacing_ms: float | None = None) -> tuple[float, int, float]:
    """Run main_test.n_node_logical_pair_with_app and return (avg, n, elapsed)."""
    from main_test import build_n_node_params, n_node_logical_pair_with_app

    params = build_n_node_params()
    params["num_logical_pairs"] = num_pairs
    params["two_qubit_gate_fidelity"] = float(two_qubit_gate_fidelity)
    params["log_level"] = "WARNING"
    if run_duration_ms is not None:
        params["run_duration_ms"] = float(run_duration_ms)
    if round_spacing_ms is not None:
        params["round_spacing_ms"] = float(round_spacing_ms)

    t0 = time.time()
    result = n_node_logical_pair_with_app(verbose=False, config_file=config_file, params=params)
    elapsed = time.time() - t0
    m = result["metrics"]
    return float(m["avg_end_to_end_logical_corrected"]), int(m["num_logical_pairs_completed"]), elapsed


def _run_cheating(num_pairs: int, two_qubit_gate_fidelity: float, seed: int, num_nodes: int) -> tuple[float, int, float]:
    """Run cheating.simulate_logical_pairs_nnode and return (avg, n, elapsed)."""
    from utilities import simulate_logical_pairs_nnode

    t0 = time.time()
    result = simulate_logical_pairs_nnode(
        num_nodes=num_nodes,
        num_pairs=num_pairs,
        params={"two_qubit_gate_fidelity": float(two_qubit_gate_fidelity)},
        seed=seed,
    )
    elapsed = time.time() - t0
    return (
        float(result["avg_end_to_end_logical_corrected"]),
        int(result["num_logical_pairs_completed"]),
        elapsed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--num-pairs", type=int, default=200,
                        help="logical pairs per pipeline (default 200)")
    parser.add_argument("-f", "--fidelity", type=float, default=0.99,
                        help="two_qubit_gate_fidelity (default 0.99)")
    parser.add_argument("-c", "--config-file", type=str,
                        default="config/standard_configs/line_2_2G.json",
                        help="topology config file (default line_2_2G.json)")
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument("--run-duration-ms", type=float, default=None,
                        help="override SeQUeNCe simulated run window (default: leave at build_n_node_params 1000.0); "
                             "raise this to let the reference complete more of the requested pairs.")
    parser.add_argument("--round-spacing-ms", type=float, default=None,
                        help="override SeQUeNCe per-pair scheduling spacing (default: 1.0). Must exceed the per-pair "
                             "latency or most attempts will pipeline-collide and abort. ~10ms is a safe choice.")
    parser.add_argument("--skip-reference", action="store_true",
                        help="run only cheating, skip main_test (for fast iteration)")
    args = parser.parse_args()

    num_nodes = _count_routers(args.config_file)
    print(f"validation: N={args.num_pairs} pairs at f_2={args.fidelity} "
          f"on {args.config_file} ({num_nodes} routers)")
    print("-" * 60)

    cheat_avg, cheat_n, cheat_t = _run_cheating(args.num_pairs, args.fidelity, args.seed, num_nodes)
    cheat_sigma = _binomial_sigma(cheat_avg, cheat_n)
    print(f"cheating:  F = {cheat_avg:.6f} +/- {cheat_sigma:.6f}  "
          f"(N={cheat_n}, {cheat_t:.1f}s = {cheat_t / cheat_n * 1000:.2f} ms/pair)")

    if args.skip_reference:
        print("(reference skipped via --skip-reference)")
        return

    print("running main_test reference (this is slow) ...")
    ref_avg, ref_n, ref_t = _run_main_test(
        args.num_pairs, args.fidelity, args.config_file,
        run_duration_ms=args.run_duration_ms,
        round_spacing_ms=args.round_spacing_ms,
    )
    ref_sigma = _binomial_sigma(ref_avg, ref_n)
    print(f"main_test: F = {ref_avg:.6f} +/- {ref_sigma:.6f}  "
          f"(N={ref_n}, {ref_t:.1f}s = {ref_t / max(ref_n, 1):.2f} s/pair)")

    delta = abs(cheat_avg - ref_avg)
    combined_sigma = math.sqrt(cheat_sigma ** 2 + ref_sigma ** 2)
    if combined_sigma > 0:
        sigmas = delta / combined_sigma
    else:
        sigmas = float("inf") if delta > 0 else 0.0
    speedup = ref_t / cheat_t if cheat_t > 0 else float("inf")
    print("-" * 60)
    print(f"|delta_F| = {delta:.6f}  ({sigmas:.2f} sigma; should be < ~3 if statistics agree)")
    print(f"speedup:    {speedup:.0f} x")


if __name__ == "__main__":
    main()
