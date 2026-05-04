"""Cheating parallel sweep -- a drop-in replacement for main_test_parallel.py.

Same parameter sweep, same plot, same JSON output schema, same parallel
multiprocessing structure (Windows 11-safe). The only difference is that
the per-pair quantum work is performed by `cheating.utilities` directly on
stim, with the entire SeQUeNCe network simulator skipped.

Why this is "cheating" but still correct:

  Looking at main_test.py, every pair flows through:
    physical Bell pair generation (Barret-Kok protocol with classical
    confirmation messages and channel propagation delay)
       -> Steane encoding + optional FT prep
       -> 3-phase teleported CNOT (with classical messaging between Alice
          and Bob between phases)
       -> noiseless endpoint stabilizer recovery
       -> Bell-state fidelity from logical XX, YY, ZZ correlators
  Of those, only the bolded *quantum* operations affect the fidelity number
  the parallel sweep plots. All the network choreography is overhead.

  Profile-grade rough breakdown of where main_test_parallel spends time:
    1. SeQUeNCe event-driven simulation (~80%): hundreds of timeline events
       per pair (Barret-Kok rounds, Reservation handshake, classical message
       send + receive, app callbacks, etc.).
    2. Per-circuit `qm.run_circuit` setup (~15%): each TCNOT phase / encoder
       step builds a fresh stim.TableauSimulator and merges sub-blocks.
    3. Tableau merging in `_calculate_pair_fidelity` (~5%): builds a temp
       simulator from per-block tableaus before peeking the correlators.

  The cheating runner removes (1) entirely, replaces (2) with a single
  TableauSimulator per shot driven by *precompiled* circuits, and removes
  (3) by keeping all qubits in one simulator from the start. The noise model
  (DEPOLARIZE2 with p_2 = 1.25 * (1 - f_2), etc.) and the Steane stabilizer
  / encoder / FT-check / TCNOT / fidelity formulas are bit-for-bit the same
  ones that quantum_manager_tableau.py / css_codes.py / TeleportedCNOT.py /
  RequestLogicalPairApp.py use. Same physics, same statistics; the result
  matches main_test_parallel within Monte Carlo error of ~1/sqrt(N).

Designed to run on Windows 11:
  * worker callable lives at module scope so it pickles for the spawn start method
  * `if __name__ == "__main__":` guard + `multiprocessing.freeze_support()`
  * worker count capped (MAX_WORKERS_WINDOWS) to bound resident memory
  * single persistent ProcessPoolExecutor reused across every fidelity
  * single-chunk exceptions caught instead of aborting the whole sweep
  * worker stdout redirected to devnull so chunk progress doesn't scramble
    the parent's tqdm bars
"""

# Suppress the cupy CUDA_PATH UserWarning that fires once per spawned worker
# (a many-worker x many-fidelity sweep otherwise produces N x M identical lines).
# Must run before cupy is imported anywhere in the process tree.
import os as _os
_os.environ.setdefault("CUDA_PATH", "")

print('importing ...')
import contextlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Make sure the cheating package is importable regardless of cwd. This file
# lives in `<repo>/cheating/`, so adding the file's parent directory to
# sys.path is enough to import `utilities` (sibling module).
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from utilities import (  # noqa: E402  -- intentional after sys.path tweak
    aggregate_worker_metrics,
    simulate_logical_pairs_nnode,
    split_pair_counts,
)


# Each spawned worker re-imports stim + numpy + this module. That's modest
# (no SeQUeNCe / no CUDA) but we still cap workers so a 32-thread machine
# doesn't blow up its pagefile.
MAX_WORKERS_WINDOWS = 8

# Pairs per chunk. Each chunk is one future submitted to the pool, so the inner
# tqdm bar advances once per chunk completion. Smaller -> more frequent ticks
# but more pipeline-precompile overhead per chunk.
TARGET_CHUNK_SIZE = 50


def run_cheating_chunk(
    chunk_index: int,
    pair_count: int,
    two_qubit_gate_fidelity: float,
    base_seed: int,
    num_nodes: int,
) -> dict[str, object]:
    """Worker entry point: simulate one chunk of logical pairs.

    Picklable (module-scope, simple-typed args) so the spawn start method
    can ship it to a fresh Python process on Windows.

    Args:
        chunk_index: Index of this chunk within the fidelity batch (used
            only to derive a unique RNG seed).
        pair_count: Number of logical pairs assigned to this chunk.
        two_qubit_gate_fidelity: Two-qubit gate fidelity for this batch.
        base_seed: Seed root shared across the whole sweep, perturbed per chunk.
        num_nodes: Length of the linear-chain topology (>=2). N=2 reproduces
            line_2_2G.json; N=6 reproduces line_6_2G.json; etc.

    Returns:
        Per-chunk metrics dict (same schema as a single
        `simulate_logical_pairs_nnode` call).
    """
    # Derive a chunk-unique seed so chunks don't replay the same RNG stream.
    seed = (int(base_seed) + 1) * 1_000_003 + int(chunk_index) * 7919

    # Silence per-chunk stdout so the parent's tqdm bars stay readable;
    # stderr is left intact so genuine errors still surface.
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        return simulate_logical_pairs_nnode(
            num_nodes=int(num_nodes),
            num_pairs=pair_count,
            params={"two_qubit_gate_fidelity": float(two_qubit_gate_fidelity)},
            seed=seed,
        )


def plot_F_corrected(
    fidelity_array: np.ndarray,
    F_corrected: np.ndarray,
    output_path: Path,
) -> None:
    """Save a plot of F_corrected vs two_qubit_gate_fidelity."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fidelity_array, F_corrected, marker="o", linestyle="-")
    ax.set_xlabel("two_qubit_gate_fidelity")
    ax.set_ylabel("F_corrected")
    ax.set_title("End-to-end logical fidelity vs two-qubit gate fidelity (cheating)")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Parallel parameter sweep, save JSON in repo root, plot log-log.

    Env var overrides (intended for quick smoke tests, never required):
        CHEATING_NUM_NODES        - chain length / number of routers (default 2 for line_2_2G; set to 6 for line_6_2G)
        CHEATING_NUM_FIDS         - number of fidelity grid points (default 18)
        CHEATING_NUM_PAIRS        - logical pairs per fidelity (default 100000)
        CHEATING_MAX_WORKERS      - cap worker count (default min(cpu, MAX_WORKERS_WINDOWS))
        CHEATING_TARGET_CHUNK     - pairs per chunk override
        CHEATING_BASE_SEED        - reproducibility seed root
    """
    start = time.time()
    print("main_test_parallel_cheating start ...")

    # Same sweep grid and per-fidelity workload as main_test_parallel.py.
    num_nodes = int(_os.environ.get("CHEATING_NUM_NODES", "2"))
    if num_nodes < 2:
        raise ValueError(f"CHEATING_NUM_NODES must be >= 2, got {num_nodes}")
    num_fids = int(_os.environ.get("CHEATING_NUM_FIDS", "18"))
    fidelity_array = 1.0 - np.power(10.0, np.linspace(-4.0, -1.5, num_fids))
    num_logical_pairs = int(_os.environ.get("CHEATING_NUM_PAIRS", "100000"))
    workers_cap = int(_os.environ.get("CHEATING_MAX_WORKERS", str(MAX_WORKERS_WINDOWS)))
    workers = max(1, min(os.cpu_count() or 1, workers_cap))
    # Time-based default so each run uses a fresh RNG stream; CHEATING_BASE_SEED
    # env var still overrides for reproducibility. The actual seed used is
    # printed below and persisted in the output JSON, so a run can be replayed
    # by setting CHEATING_BASE_SEED to that value.
    env_seed = _os.environ.get("CHEATING_BASE_SEED")
    base_seed = int(env_seed) if env_seed is not None else time.time_ns() & 0xFFFFFFFF

    repo_dir = _HERE.parent  # main_test_parallel.py lives here
    plot_dir = repo_dir / "plot_pngs"
    plot_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"num_nodes={num_nodes} | workers={workers} | fidelities={len(fidelity_array)} | "
        f"pairs/fidelity={num_logical_pairs} | base_seed={base_seed}"
    )

    target_chunk = int(_os.environ.get("CHEATING_TARGET_CHUNK", str(TARGET_CHUNK_SIZE)))
    num_chunks = max(workers, math.ceil(num_logical_pairs / target_chunk))
    pair_counts = split_pair_counts(num_logical_pairs, num_chunks)
    print(f"chunks/fidelity={len(pair_counts)} | pairs/chunk~{pair_counts[0]}")

    F_corrected_values = np.full(len(fidelity_array), np.nan)
    one_minus_F_corrected = np.full(len(fidelity_array), np.nan)

    # One persistent pool reused across every fidelity value: spawn re-imports
    # are paid once per worker, not once per fidelity.
    with ProcessPoolExecutor(max_workers=workers) as executor:
        outer_bar = tqdm(
            total=len(fidelity_array),
            desc="fidelity sweep",
            unit="pt",
            position=0,
            dynamic_ncols=True,
        )
        for index, fidelity in enumerate(fidelity_array):
            sweep_start = time.time()
            tqdm.write(f"\n=== Sweep point {index + 1}/{len(fidelity_array)} ===")
            tqdm.write(f"two_qubit_gate_fidelity = {fidelity:.6f}")
            chunk_seed_base = base_seed + index * 1_000_001
            futures = [
                executor.submit(
                    run_cheating_chunk,
                    chunk_index,
                    pair_count,
                    float(fidelity),
                    chunk_seed_base,
                    num_nodes,
                )
                for chunk_index, pair_count in enumerate(pair_counts)
                if pair_count > 0
            ]

            chunk_metrics: list[dict[str, object]] = []
            inner_bar = tqdm(
                total=len(futures),
                desc=f"  chunks @ fid={fidelity:.4g}",
                unit="chunk",
                position=1,
                leave=False,
                dynamic_ncols=True,
            )
            failed_chunks = 0
            for future in as_completed(futures):
                try:
                    chunk_metrics.append(future.result())
                except Exception:  # noqa: BLE001
                    failed_chunks += 1
                    tqdm.write(
                        f"  chunk failed at fid={fidelity:.6f} "
                        f"({failed_chunks}/{len(futures)} so far):"
                    )
                    tqdm.write(traceback.format_exc())
                inner_bar.update(1)
            inner_bar.close()

            if not chunk_metrics:
                tqdm.write(
                    f"  all chunks failed at fid={fidelity:.6f}; "
                    f"leaving F_corrected as NaN and continuing"
                )
                outer_bar.update(1)
                continue

            summary = aggregate_worker_metrics(chunk_metrics)
            f_corrected = summary["avg_end_to_end_logical_corrected"]
            F_corrected_values[index] = f_corrected
            one_minus_F_corrected[index] = 1.0 - f_corrected
            tqdm.write(
                f"two_qubit_gate_fidelity={fidelity:.6f} | "
                f"completed={summary['num_logical_pairs_completed']}/"
                f"{summary['num_logical_pairs_requested']} | "
                f"avg F_corrected={f_corrected:.6f} | "
                f"1 - F_corrected={1.0 - f_corrected:.6e} | "
                f"elapsed={time.time() - sweep_start:.1f}s"
            )
            outer_bar.set_postfix(
                F_corr=f"{f_corrected:.4f}",
                elapsed=f"{time.time() - sweep_start:.1f}s",
            )
            outer_bar.update(1)
        outer_bar.close()

    file_suffix = "" if num_nodes == 2 else f"_n{num_nodes}"
    plot_path = plot_dir / f"main_test_parallel_cheating_sweep{file_suffix}.png"
    plot_F_corrected(fidelity_array, F_corrected_values, plot_path)
    print(f"\nPlot saved to {plot_path}")

    # JSON dump alongside this script (cheating/main_test_parallel_cheating.py).
    data_path = _HERE / f"main_test_parallel_cheating_sweep{file_suffix}.json"
    payload = {
        "num_nodes": int(num_nodes),
        "num_logical_pairs_per_fidelity": int(num_logical_pairs),
        "workers": int(workers),
        "chunks_per_fidelity": int(len(pair_counts)),
        "base_seed": int(base_seed),
        "two_qubit_gate_fidelity": fidelity_array.tolist(),
        "F_corrected": F_corrected_values.tolist(),
        "one_minus_F_corrected": one_minus_F_corrected.tolist(),
    }
    with data_path.open("w", encoding="utf-8") as data_file:
        json.dump(payload, data_file, indent=2)
        data_file.write("\n")
    print(f"Data saved to {data_path}")

    end = time.time()
    print(f"\nMain execution time: {end - start:.2f} seconds")


if __name__ == "__main__":
    # freeze_support() is needed for frozen Windows executables; harmless
    # otherwise. Must be called before any pool is built.
    mp.freeze_support()
    main()
