"""Central settings for the QEC experiment pipeline.

Edit this file to control what the pipeline runs.

Responsibilities:
- define named presets for codes and sweep ranges
- choose the active preset used by the pipeline
- define default output paths for generated configs, manifests, pickles, CSVs, and figures

This file does not run experiments by itself.

Typical pipeline:
1) `python experiment_config_generator.py` (generate configs/manifests)
2) `python main.py` (run active preset) or `python main.py --all-presets`
3) `python process_results.py` (pickle -> CSV)
4) `python plot_results.py` (CSV -> figures)
"""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
CONFIG_GENERATED_DIR = BASE_DIR / "config" / "generated"
CSV_DIR = RESULTS_DIR / "csv"
PICKLE_DIR = RESULTS_DIR / "pickle"
FIGURES_DIR = RESULTS_DIR / "figures"
GENERATED_CONFIGS_DIR = RESULTS_DIR / "generated_configs"

# Edit only this value for normal use.
# - `python main.py` will use ACTIVE_PRESET.
# - `python main.py --preset <name>` ignores ACTIVE_PRESET.
# - `python main.py --all-presets` runs all presets with existing manifests.
# Available presets are listed in PRESETS below.
# Available presets:
#   "shor_smoke"              Fast one-code sanity check
#   "paper_code_compare"      Compare CSS codes at a fixed hardware point (similar to Fig. 3a)
#   "paper_initial_fidelity"  Sweep initial physical Bell-pair fidelity
#   "paper_twoq_gate"         Sweep two-qubit gate fidelity
#   "paper_oneq_gate"         Sweep single-qubit gate fidelity
#   "paper_distance_sweep"    Sweep elementary-link distance
#   "paper_link_count_sweep"  Sweep number of links in the chain
#   "paper_coherence_sweep"   Sweep memory coherence time
#   "all_presets"             Special mode: run every preset in PRESETS
ACTIVE_PRESET = "all_presets"

# Available CSS codes:
#   "[[7,1,3]]"   Steane
#   "[[9,1,3]]"   Shor
#   "[[15,1,3]]"  Reed-Muller
#   "[[23,1,7]]"  Golay
#   "[[31,1,7]]"  BCH
PRESETS = {
    # Fixed hardware point, compare all CSS codes directly.
    "paper_code_compare": {
        "experiment": "code_compare",
        "idle_decoherence_debug": False,
        "codes": [
            "[[7,1,3]]",
            "[[9,1,3]]",
            "[[15,1,3]]",
            "[[23,1,7]]",
            "[[31,1,7]]",
        ],
        "distance_values_km": [1.0],
        "linear_sizes": [5],
        "coherence_time_values": [10.0],
        "initial_fidelity_values": [1.0],
        "twoq_values": [0.999],
        "oneq_values": [0.9999],
        "output_csv": CSV_DIR / "code_compare.csv",
        "output_figures": FIGURES_DIR / "code_compare",
    },
    # Sweep raw physical Bell-pair quality for all CSS codes.
    "paper_initial_fidelity": {
        "experiment": "initial_fidelity_sweep",
        "idle_decoherence_debug": False,
        "codes": [
            "[[7,1,3]]",
            "[[9,1,3]]",
            "[[15,1,3]]",
            "[[23,1,7]]",
            "[[31,1,7]]",
        ],
        "distance_values_km": [1.0],
        "linear_sizes": [5],
        "coherence_time_values": [10.0],
        "initial_fidelity_values": [0.99, 0.999, 0.9999, 1.0],
        "twoq_values": [0.999],
        "oneq_values": [0.9999],
        "output_csv": CSV_DIR / "initial_fidelity_sweep.csv",
        "output_figures": FIGURES_DIR / "initial_fidelity_sweep",
    },
    # Sweep two-qubit gate fidelity for all CSS codes.
    "paper_twoq_gate": {
        "experiment": "two_qubit_gate_sweep",
        "idle_decoherence_debug": False,
        "codes": [
            "[[7,1,3]]",
            "[[9,1,3]]",
            "[[15,1,3]]",
            "[[23,1,7]]",
            "[[31,1,7]]",
        ],
        "distance_values_km": [1.0],
        "linear_sizes": [5],
        "coherence_time_values": [10.0],
        "initial_fidelity_values": [1.0],
        "twoq_values": [0.995, 0.999, 0.9999, 1.0],
        "oneq_values": [0.9999],
        "output_csv": CSV_DIR / "two_qubit_gate_sweep.csv",
        "output_figures": FIGURES_DIR / "two_qubit_gate_sweep",
    },
    # Sweep single-qubit gate fidelity for all CSS codes.
    "paper_oneq_gate": {
        "experiment": "single_qubit_gate_sweep",
        "idle_decoherence_debug": False,
        "codes": [
            "[[7,1,3]]",
            "[[9,1,3]]",
            "[[15,1,3]]",
            "[[23,1,7]]",
            "[[31,1,7]]",
        ],
        "distance_values_km": [1.0],
        "linear_sizes": [5],
        "coherence_time_values": [10.0],
        "initial_fidelity_values": [1.0],
        "twoq_values": [0.999],
        "oneq_values": [0.995, 0.997, 0.9985, 0.9995, 1.0],
        "output_csv": CSV_DIR / "single_qubit_gate_sweep.csv",
        "output_figures": FIGURES_DIR / "single_qubit_gate_sweep",
    },
    "paper_distance_sweep": {
        "experiment": "distance_sweep",
        "idle_decoherence_debug": False,
        "codes": [
            "[[7,1,3]]",
            "[[9,1,3]]",
        ],
        "distance_values_km": [0.5, 1.0, 2.0, 5.0, 10.0],
        "linear_sizes": [5],
        "coherence_time_values": [10.0],
        "initial_fidelity_values": [1.0],
        "twoq_values": [0.999],
        "oneq_values": [0.9999],
        "output_csv": CSV_DIR / "distance_sweep.csv",
        "output_figures": FIGURES_DIR / "distance_sweep",
    },
    "paper_link_count_sweep": {
        "experiment": "link_count_sweep",
        "idle_decoherence_debug": False,
        "codes": [
            "[[7,1,3]]",
            "[[9,1,3]]",
        ],
        "distance_values_km": [1.0],
        "linear_sizes": [2, 3, 5, 10, 20],
        "coherence_time_values": [10.0],
        "initial_fidelity_values": [1.0],
        "twoq_values": [0.999],
        "oneq_values": [0.9999],
        "output_csv": CSV_DIR / "link_count_sweep.csv",
        "output_figures": FIGURES_DIR / "link_count_sweep",
    },
    "paper_coherence_sweep": {
        "experiment": "coherence_sweep",
        "idle_decoherence_debug": True,
        "codes": [
            "[[7,1,3]]",
        ],
        "distance_values_km": [1.0],
        "linear_sizes": [5],
        "coherence_time_values": [10.0, 5.0, 1.0, 0.5, 0.1, 0.05, 0.01],
        "initial_fidelity_values": [1.0],
        "twoq_values": [0.999],
        "oneq_values": [0.9999],
        "output_csv": CSV_DIR / "coherence_sweep.csv",
        "output_figures": FIGURES_DIR / "coherence_sweep",
    },
    # Fast sanity check: one code, very short sweeps.
    "shor_smoke": {
        "experiment": "all",
        "idle_decoherence_debug": False,
        "codes": ["[[7,1,3]]"],
        "initial_fidelity_values": [0.999, 1.0],
        "twoq_values": [0.999, 1.0],
        "oneq_values": [0.999, 1.0],
        "distance_values_km": [1.0],
        "linear_sizes": [5],
        "coherence_time_values": [10.0, 1.0, 0.1, 0.01, 0.001],
        "output_csv": CSV_DIR / "shor_smoke.csv",
        "output_figures": FIGURES_DIR / "shor_smoke",
    },
}


def get_settings_for_preset(preset_name: str):
    if preset_name not in PRESETS:
        raise KeyError(f"Unknown preset '{preset_name}'. Available: {sorted(PRESETS.keys())}")

    settings = PRESETS[preset_name].copy()
    settings["config_output_dir"] = str(CONFIG_GENERATED_DIR / preset_name)
    settings["manifest_path"] = str(RESULTS_DIR / "manifests" / f"{preset_name}.json")
    settings["pickle_path"] = str(PICKLE_DIR / f"{preset_name}.pkl")
    settings["output_csv"] = str(settings["output_csv"])
    settings["output_figures"] = str(settings["output_figures"])
    settings["results_dir"] = str(RESULTS_DIR)
    settings["config_generated_dir"] = str(CONFIG_GENERATED_DIR)
    settings["csv_dir"] = str(CSV_DIR)
    settings["pickle_dir"] = str(PICKLE_DIR)
    settings["figures_dir"] = str(FIGURES_DIR)
    settings["generated_configs_dir"] = str(GENERATED_CONFIGS_DIR)
    settings["active_preset"] = preset_name
    return settings


def get_all_preset_names():
    return list(PRESETS.keys())


def is_all_presets_active():
    return ACTIVE_PRESET == "all_presets"


def get_effective_preset_names():
    if is_all_presets_active():
        return get_all_preset_names()
    return [ACTIVE_PRESET]


def get_active_settings():
    if is_all_presets_active():
        # Return minimal settings for callers that only need active_preset metadata.
        # Scripts that need concrete paths should iterate get_effective_preset_names()
        # and call get_settings_for_preset(name).
        return {
            "active_preset": ACTIVE_PRESET,
            "results_dir": str(RESULTS_DIR),
            "config_generated_dir": str(CONFIG_GENERATED_DIR),
            "csv_dir": str(CSV_DIR),
            "pickle_dir": str(PICKLE_DIR),
            "figures_dir": str(FIGURES_DIR),
            "generated_configs_dir": str(GENERATED_CONFIGS_DIR),
        }
    return get_settings_for_preset(ACTIVE_PRESET)
