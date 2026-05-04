# `cheating/` — fast, SeQUeNCe-free Steane-pipeline simulator

A drop-in replacement for the SeQUeNCe-based reference pipeline that produces
**statistically equivalent** end-to-end logical-Bell-pair fidelity numbers at
roughly **1000×** the speed. The cheating module bypasses the SeQUeNCe
event-driven scheduler, classical-message timing, BarretKok physical-Bell-pair
generation, and per-circuit tableau merges. What remains — the *quantum*
physics — runs as straight-line stim circuits with the same noise model,
encoder, FT-prep, teleported CNOT, encoded swap, and endpoint recovery as the
reference (`RequestLogicalPairApp.py`, `TeleportedCNOT.py`, `QREProtocol.py`,
`css_codes.py`, `quantum_manager_tableau.py`).

The cheating module supports linear chains of **any length N ≥ 2** by parsing
the `config/standard_configs/line_*_2G.json` topology files (only the
`QuantumRouter` count is needed; other config fields are inherent to the
SeQUeNCe pipeline and irrelevant for the per-pair quantum statistics).

---

## Folder contents

| File | Role |
|---|---|
| `utilities.py` | Core library. Steane code data, encoder/FT-prep, teleported CNOT phases, encoded swap, BarretKok BSM simulation, endpoint recovery, `simulate_logical_pairs_nnode` entry point. |
| `main_test_parallel_cheating.py` | **Parallel parameter sweep** over `two_qubit_gate_fidelity`. Drop-in for `../main_test_parallel.py`. Outputs sweep JSON + log-log PNG. |
| `main_test_cheating.py` | **Sequential single-run** drop-in for `../main_test.py`. Same `n_node_logical_pair_with_app(...)`-style call signature; reads the config file to determine `N`. |
| `validate_against_main_test.py` | Cross-check tool. Runs both pipelines for a small `N` and reports `|F_cheat - F_ref| / σ`. |
| `plot_cheating_sweep.ipynb` | Jupyter notebook for plotting `main_test_parallel_cheating_sweep[*].json`. |
| `main_test_parallel_cheating_sweep[*].json` | Outputs from prior sweeps (e.g. `_n6.json` for the 6-router topology). |
| `sweep_plot.pdf` | Saved sweep plot. |

---

## Quick start

All commands assume the **repository root** as the current directory (the
folder that contains `main_test.py` and `cheating/`).

### 1. Sequential run (`main_test_cheating.py`)

This mirrors the reference's `n_node_logical_pair_with_app(...)` and is the
easiest way to get a single-point fidelity number for a given config:

```bash
python cheating/main_test_cheating.py
```

Defaults: `config/standard_configs/line_2_2G.json`, 1000 logical pairs, fresh
RNG seed.

### 2. Parameter sweep over `two_qubit_gate_fidelity` (`main_test_parallel_cheating.py`)

This is the workhorse for plots (`F_corrected` vs `two_qubit_gate_fidelity`):

```bash
python cheating/main_test_parallel_cheating.py
```

Defaults: 2-router chain (`line_2_2G.json`), 18 fidelity grid points,
100 000 logical pairs per fidelity, up to `MAX_WORKERS_WINDOWS` (=8) workers.

Outputs:
- JSON: `cheating/main_test_parallel_cheating_sweep.json`
- PNG: `plot_pngs/main_test_parallel_cheating_sweep.png`

For longer chains the file names pick up an `_n{N}` suffix (e.g.
`main_test_parallel_cheating_sweep_n6.json`).

### 3. Cross-check against SeQUeNCe (`validate_against_main_test.py`)

```bash
python cheating/validate_against_main_test.py -n 50 -f 0.99 -c config/standard_configs/line_3_2G.json
```

Runs both pipelines and reports `|ΔF| / σ`. Pass criterion is typically
`< 3σ`. Reference run is slow (~1.5 s/pair at `line_2`, more at higher N);
use `--skip-reference` for cheating-only timing.

---

## How the configuration JSON is used

The cheating module reads only what it actually needs from the SeQUeNCe
config: **the chain length N** (number of `QuantumRouter` nodes) and the
**per-router `ft_max_retries`** (used by FT-prep). All other JSON fields —
qchannel distances, BSM nodes, classical-channel topology, `formalism`,
`stop_time`, etc. — are inherent to the SeQUeNCe event-driven simulator and
**have no effect on the per-pair quantum statistics** (see the docstring at
the top of `utilities.py` for why each shortcut preserves the statistical
outcome).

For per-pair physics, the cheating reads these fields from
`build_n_node_params()` (or whatever overrides you pass):
`gate_fidelity`, `two_qubit_gate_fidelity`, `measurement_fidelity`,
`state_preparation_fidelity`, `physical_bell_pair_fidelity`,
`gate_error_channel` (+ optional `pauli_1q_weights` / `pauli_2q_weights`),
`ft_prep_mode`, `ft_max_retries`, `correction_mode`, and **`link_distance_km`**.

`link_distance_km` deserves a callout: it determines the per-photon
detector-click probability inside the BarretKok BSM (longer distance →
more fiber attenuation → fewer photons reach the detector). At
`measurement_fidelity = 1.0` it affects only throughput (not `F_corrected`),
but at `measurement_fidelity < 1.0` it shifts the BarretKok ghost-vs-true
click ratio and so does change `F_corrected`. The cheating uses fixed
hardware-template constants matching the standard configs (memory
efficiency 0.9, detector efficiency 0.95, fibre attenuation 0.0002 dB/m).

### Selecting a config in `main_test_cheating.py`

Pass it explicitly via the `config_file` keyword:

```python
from cheating.main_test_cheating import n_node_logical_pair_cheating

result = n_node_logical_pair_cheating(
    verbose=True,
    config_file="config/standard_configs/line_6_2G.json",
    params={"num_logical_pairs": 5000, "two_qubit_gate_fidelity": 0.997},
    seed=42,
)
print(result["metrics"]["avg_end_to_end_logical_corrected"])
```

Or override on the command line via the env var `CHEATING_CONFIG_FILE`:

```bash
CHEATING_CONFIG_FILE=config/standard_configs/line_6_2G.json python cheating/main_test_cheating.py
```

### Selecting a config in `main_test_parallel_cheating.py`

The parallel runner doesn't read the JSON path directly — it reads
`CHEATING_NUM_NODES` (the chain length). To run on `line_6_2G.json` (which
has 6 `QuantumRouter` nodes), set:

```bash
CHEATING_NUM_NODES=6 python cheating/main_test_parallel_cheating.py
```

The full set of env-var overrides for the parallel runner:

| Variable | Default | Meaning |
|---|---|---|
| `CHEATING_NUM_NODES` | `2` | Chain length. `2`→`line_2_2G`, `3`→`line_3_2G`, `6`→`line_6_2G`, etc. |
| `CHEATING_NUM_FIDS` | `18` | Fidelity grid points (sweeps `two_qubit_gate_fidelity` from `1−10⁻⁴` to `1−10⁻¹·⁵`). |
| `CHEATING_NUM_PAIRS` | `100000` | Logical pairs simulated per fidelity point. |
| `CHEATING_MAX_WORKERS` | `8` | Cap on `ProcessPoolExecutor` workers. |
| `CHEATING_TARGET_CHUNK` | `50` | Pairs per chunk; controls inner `tqdm` granularity. |
| `CHEATING_BASE_SEED` | (time-based) | RNG seed root, persisted in output JSON for replay. |

Combine them as needed:

```bash
CHEATING_NUM_NODES=6 CHEATING_NUM_PAIRS=20000 CHEATING_MAX_WORKERS=12 \
    python cheating/main_test_parallel_cheating.py
```

### Selecting a config in `validate_against_main_test.py`

Use `-c / --config-file`:

```bash
python cheating/validate_against_main_test.py -c config/standard_configs/line_6_2G.json -n 50
```

---

## What the cheating module models faithfully

The cheating's `simulate_logical_pairs_nnode` is a step-for-step transcription
of the SeQUeNCe per-pair quantum sequence:

1. **Encoding.** Steane `[[7,1,3]]` Paetznick-Reichardt encoder (8 CNOTs +
   3 H), with optional minimal/standard FT-prep retry. Right-facing blocks
   end in `|+⟩_L`, left-facing in `|0⟩_L`.
2. **Bell-pair generation.** At `measurement_fidelity = 1.0`, prepared
   directly via `H + CX` on the comm registers. At `measurement_fidelity < 1.0`,
   each pair instead goes through the **explicit BarretKok-BSM simulation**
   (`_simulate_barret_kok_pair`, mirrors `sequence/components/bsm.py:512-580`
   and `sequence/entanglement_management/generation/barret_kok.py:69-92`)
   so the meas-fid coupling into Bell-pair fidelity matches the reference.
3. **Per-hop teleported CNOT.** Phase A (`CX(data, comm)` + measure comm),
   Phase B (`CX(comm, data)` + feed-forward `X` + `H(comm)` + measure comm),
   Phase C (feed-forward `Z` on data). Identical to `TeleportedCNOT.py`.
4. **Middle-node encoded swap.** Transversal `CX(L, R)` + transversal `H(L)` +
   measurement of all 14 qubits. Decoded via `_decode_middle_bsm` (mirrors
   `css_codes.py:491-528`); the per-node frame contribution
   `(b_x_corrected, b_z_corrected)` propagates to the initiator.
5. **Initiator frame correction.** `X^final_bx Z^final_bz` applied
   transversally on `router_0`'s data block (= logical `LX`/`LZ` for Steane).
6. **Noiseless endpoint stabilizer recovery.** Peek X- and Z-stabilizers,
   decode each syndrome, apply single-qubit corrections.
7. **End-to-end fidelity.** `F = (1 + ⟨X̄X̄⟩ − ⟨ȲȲ⟩ + ⟨Z̄Z̄⟩) / 4` between
   `router_0`'s and `router_{N−1}`'s data blocks.

### Noise model (matches `quantum_manager_tableau.py`)

- 1q gate noise: `DEPOLARIZE1` with `p = 1.5 × (1 − gate_fidelity)`.
- 2q gate noise: `DEPOLARIZE2` with `p = 1.25 × (1 − two_qubit_gate_fidelity)`.
- Measurement noise: per-bit flip with `p = 1 − measurement_fidelity`.
- State-preparation noise: `X_ERROR` after every reset with
  `p = 1 − state_preparation_fidelity`.
- Bell-pair Werner noise: `DEPOLARIZE2` with `p = 16/15 × (1 − physical_bell_pair_fidelity)`.
- Custom Pauli error channels via `pauli_1q_weights` / `pauli_2q_weights`
  with `gate_error_channel = "pauli"`.

### Known limitations

- **Idle decoherence is dropped.** Negligible at the default
  `idle_t1_sec = idle_t2_sec = 1e12 s` (per-qubit Pauli probabilities are
  ~`idle_sec / T1` ≈ `10⁻¹²` per ns of idle); not modeled if you set very
  small `T1`/`T2`. As a corollary, `idle_pauli_x/y/z` only matter when idle
  decoherence fires — at the default `T1`/`T2` they have no measurable
  effect on `F` in either pipeline (cheating drops them, reference's
  `apply_idling_decoherence` produces noise below `1e-9` per qubit). They
  are accepted by the cheating's params dict for schema parity but not used.
- **Latency / throughput / per-link logical fidelities are returned as `NaN`.**
  These are properties of the SeQUeNCe event simulator that the cheating
  module bypasses by design. The corresponding fields exist in the metrics
  dict for schema parity, but they don't carry information. Likewise
  `round_spacing_ms` is a SeQUeNCe scheduling parameter with no per-pair
  physics impact; both pipelines are insensitive to it.
- **`correction_mode = "qec"` and `"qec+cec"`** are not implemented for
  `N > 2`. They would require pre-swap QEC at every middle node. Use
  `"cec"` (default) or `"none"`. A `NotImplementedError` is raised for the
  unsupported combos.
- **Small residual at `measurement_fidelity < 1` combined with
  unusually large `link_distance_km`.** The explicit BarretKok-BSM
  simulation (`_simulate_barret_kok_pair` in `utilities.py`) captures the
  dominant `measurement_fidelity` corruption mechanism — line_3 with
  `meas_fid=0.99` agrees with the reference at ~0.45σ. The residual grows
  to ~2σ when `meas_fid < 1` and `link_distance_km >= 50` (high photon
  loss + flip noise), reflecting that the simulation models the BSM
  protocol but not every micro-detail of the SeQUeNCe state machine. At
  the default `measurement_fidelity = 1.0` the residual is exactly zero —
  no impact on standard sweeps.

---

## Plotting

After a sweep finishes, open `plot_cheating_sweep.ipynb` from the `cheating/`
folder (or use `cd cheating && jupyter notebook plot_cheating_sweep.ipynb`).
Run all cells; it loads the JSON next to the notebook and renders both a
linear `F_corrected` plot and a log-log `1−F` plot.

---

## Recommended workflows

### Calibrating against the reference once

If you've never run the cheating module before in a new clone:

```bash
python cheating/validate_against_main_test.py -n 50 -f 0.99 -c config/standard_configs/line_2_2G.json
```

Confirm `|ΔF| < 3σ`. Then trust the cheating runner for production sweeps.

### Standard fidelity sweep at the line_6 topology

```bash
CHEATING_NUM_NODES=6 CHEATING_MAX_WORKERS=12 python cheating/main_test_parallel_cheating.py
```

Outputs land at:
- `cheating/main_test_parallel_cheating_sweep_n6.json`
- `plot_pngs/main_test_parallel_cheating_sweep_n6.png`

### Replaying a previous run

The actual seed used is printed at the start of every parallel-runner run and
saved into the output JSON's `"base_seed"` field. To reproduce:

```bash
CHEATING_BASE_SEED=20260427 CHEATING_NUM_NODES=6 \
    python cheating/main_test_parallel_cheating.py
```

---

## Performance

Profiled per-shot:
- `line_2`: ~0.05 ms / shot
- `line_3`: ~0.14 ms / shot
- `line_6`: ~0.6 ms / shot (5 hops + 4 BSMs)

A full 18-fidelity × 100 000-pair sweep at `line_2` finishes in under 4
minutes on 12 workers; the same sweep at `line_6` takes ~20 minutes. The
SeQUeNCe reference runs the same workload roughly 1000× slower.
