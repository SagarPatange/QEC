"""Run generated experiment configs and save raw results to pickle.

This is step 2 of the QEC experiment pipeline.

Responsibilities:
- read the manifest produced by `experiment_config_generator.py`
- run the simulator for each generated config
- collect structured metrics for each run
- write all raw results into one pickle file

This file does not generate configs or plots.

Usage:
- Run active preset from `experiment_settings.py`:
  `python main.py`
- Run a specific preset:
  `python main.py --preset paper_initial_fidelity`
- Run all presets that already have generated manifests:
  `python main.py --all-presets`
"""

import argparse
import json
import pickle
from pathlib import Path

from experiment_settings import (
    get_active_settings,
    get_all_preset_names,
    get_settings_for_preset,
    is_all_presets_active,
)
from main_test import n_node_logical_pair_with_app


def load_manifest(path: Path):
    return json.loads(path.read_text())


def run_entry(entry: dict, preset_name: str) -> dict:
    print("\n" + "=" * 72)
    print(f"Preset: {preset_name}")
    print(f"Code: {entry['code_label']}")
    print(f"CSS code: {entry['css_code']}")
    print(f"Experiment: {entry['experiment']}")
    print(f"Sweep parameter: {entry['sweep_parameter']}")
    print(f"Sweep value: {entry['sweep_value']}")
    if entry.get("num_nodes") is not None:
        print(f"Number of nodes: {entry['num_nodes']}")
    if entry.get("num_links") is not None:
        print(f"Number of links: {entry['num_links']}")
    if entry.get("elementary_link_distance_km") is not None:
        print(f"Elementary-link distance (km): {entry['elementary_link_distance_km']}")
    print(f"Base config: {entry['base_config_file']}")
    print(f"Config: {entry['config_file']}")
    print("=" * 72)
    result = n_node_logical_pair_with_app(
        config_file=entry["config_file"],
        css_code=entry["css_code"],
    )
    return {
        "experiment": entry["experiment"],
        "code_label": entry["code_label"],
        "css_code": entry["css_code"],
        "base_config_file": entry["base_config_file"],
        "config_file": entry["config_file"],
        "sweep_parameter": entry["sweep_parameter"],
        "sweep_value": entry["sweep_value"],
        "num_nodes": entry.get("num_nodes"),
        "num_links": entry.get("num_links"),
        "elementary_link_distance_km": entry.get("elementary_link_distance_km"),
        "metrics": result["metrics"],
    }


def run_manifest_to_pickle(manifest_path: Path, output_path: Path, preset_name: str):
    manifest = load_manifest(manifest_path)
    results = []

    for entry in manifest["entries"]:
        results.append(run_entry(entry, preset_name))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fh:
        pickle.dump(
            {
                "preset": manifest.get("preset"),
                "experiment": manifest.get("experiment"),
                "manifest": str(manifest_path),
                "results": results,
            },
            fh,
        )

    print(f"Preset: {preset_name}")
    print(f"Ran {len(results)} experiments from {manifest_path}")
    print(f"Pickle: {output_path}")


def main():
    active_settings = get_active_settings()
    parser = argparse.ArgumentParser(description="Run experiment configs from a manifest and save pickle results.")
    parser.add_argument("--preset", type=str, default=None, help="Run a specific preset by name.")
    parser.add_argument("--all-presets", action="store_true", help="Run every preset that has a generated manifest.")
    parser.add_argument("--manifest", type=str, default=None, help="Override manifest path (single preset mode only).")
    parser.add_argument("--output", type=str, default=None, help="Override output pickle path (single preset mode only).")
    args = parser.parse_args()

    if is_all_presets_active() and not args.preset and not args.manifest and not args.output:
        args.all_presets = True

    if args.all_presets:
        if args.manifest or args.output:
            print("Ignoring --manifest/--output in --all-presets mode.")
        ran = 0
        for preset_name in get_all_preset_names():
            settings = get_settings_for_preset(preset_name)
            manifest_path = Path(settings["manifest_path"])
            output_path = Path(settings["pickle_path"])
            if not manifest_path.exists():
                print(f"Skipping preset '{preset_name}': manifest not found at {manifest_path}")
                continue
            run_manifest_to_pickle(manifest_path, output_path, preset_name)
            ran += 1
        print(f"Completed {ran} preset run(s).")
        return

    if args.preset:
        settings = get_settings_for_preset(args.preset)
    else:
        settings = active_settings

    manifest_path = Path(args.manifest) if args.manifest else Path(settings["manifest_path"])
    output_path = Path(args.output) if args.output else Path(settings["pickle_path"])
    run_manifest_to_pickle(manifest_path, output_path, settings["active_preset"])


if __name__ == "__main__":
    main()
