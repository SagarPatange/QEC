"""Convert raw pickle experiment results into CSV tables.

This is step 3 of the QEC experiment pipeline.

Responsibilities:
- read the pickle file produced by `main.py`
- flatten the raw run results into row-based data
- write a CSV file for analysis and plotting

This file does not run simulations or make plots.
"""

import argparse
import csv
import pickle
import sys
from pathlib import Path

from experiment_settings import (
    get_active_settings,
    get_effective_preset_names,
    get_settings_for_preset,
    is_all_presets_active,
)


CSV_COLUMNS = [
    "experiment",
    "code_label",
    "css_code",
    "config_file",
    "base_config_file",
    "sweep_parameter",
    "sweep_value",
    "num_nodes",
    "num_links",
    "elementary_link_distance_km",
    "physical_bell_fidelity",
    "single_qubit_gate_fidelity",
    "two_qubit_gate_fidelity",
    "avg_initial_phys",
    "avg_link_logical_fidelity",
    "end_to_end_logical_fidelity",
    "link",
    "link_index",
    "initial_phys",
    "logical_link_fidelity",
]

PAIR_CSV_COLUMNS = [
    "experiment",
    "code_label",
    "css_code",
    "config_file",
    "base_config_file",
    "sweep_parameter",
    "sweep_value",
    "num_nodes",
    "num_links",
    "elementary_link_distance_km",
    "link",
    "link_index",
    "pair_index",
    "alice_memory",
    "bob_memory",
    "alice_wait_time_sec",
    "bob_wait_time_sec",
    "alice_p_idle",
    "bob_p_idle",
    "initial_pair_fidelity",
    "post_idle_pair_fidelity",
]


def flatten_run(run: dict) -> list[dict]:
    metrics = run["metrics"]
    base_row = {
        "experiment": run["experiment"],
        "code_label": run["code_label"],
        "css_code": run["css_code"],
        "config_file": run["config_file"],
        "base_config_file": run["base_config_file"],
        "sweep_parameter": run["sweep_parameter"],
        "sweep_value": run["sweep_value"],
        "num_nodes": run.get("num_nodes"),
        "num_links": run.get("num_links"),
        "elementary_link_distance_km": run.get("elementary_link_distance_km"),
        "physical_bell_fidelity": metrics["avg_initial_phys"],
        "single_qubit_gate_fidelity": metrics["gate_fidelity"],
        "two_qubit_gate_fidelity": metrics["two_qubit_gate_fidelity"],
        "avg_initial_phys": metrics["avg_initial_phys"],
        "avg_link_logical_fidelity": metrics["avg_link_logical"],
        "end_to_end_logical_fidelity": metrics["end_to_end_logical"],
    }

    rows = [
        {
            **base_row,
            "link": "end_to_end",
            "link_index": -1,
            "initial_phys": "",
            "logical_link_fidelity": "",
        }
    ]

    for link_row in metrics["link_rows"]:
        rows.append(
            {
                **base_row,
                "link": link_row["link"],
                "link_index": link_row["link_index"],
                "initial_phys": link_row["initial_phys"],
                "logical_link_fidelity": link_row["logical"],
            }
        )
    return rows


def flatten_pair_rows(run: dict) -> list[dict]:
    metrics = run["metrics"]
    base_row = {
        "experiment": run["experiment"],
        "code_label": run["code_label"],
        "css_code": run["css_code"],
        "config_file": run["config_file"],
        "base_config_file": run["base_config_file"],
        "sweep_parameter": run["sweep_parameter"],
        "sweep_value": run["sweep_value"],
        "num_nodes": run.get("num_nodes"),
        "num_links": run.get("num_links"),
        "elementary_link_distance_km": run.get("elementary_link_distance_km"),
    }

    rows = []
    for row in metrics.get("pair_rows", []):
        rows.append({**base_row, **row})
    return rows


def process_one(settings: dict, input_path: Path, output_path: Path):
    with input_path.open("rb") as fh:
        payload = pickle.load(fh)

    rows = []
    pair_rows = []
    for run in payload["results"]:
        rows.extend(flatten_run(run))
        pair_rows.extend(flatten_pair_rows(run))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    pair_output_path = output_path.with_name(f"{output_path.stem}_pairs.csv")
    with pair_output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=PAIR_CSV_COLUMNS)
        writer.writeheader()
        for row in pair_rows:
            writer.writerow(row)

    print(f"Preset: {settings['active_preset']}")
    print(f"Wrote {len(rows)} CSV rows to {output_path}")
    print(f"Wrote {len(pair_rows)} pair CSV rows to {pair_output_path}")


def main():
    if is_all_presets_active() and len(sys.argv) == 1:
        done = 0
        for preset_name in get_effective_preset_names():
            settings = get_settings_for_preset(preset_name)
            input_path = Path(settings["pickle_path"])
            output_path = Path(settings["output_csv"])
            if not input_path.exists():
                print(f"Skipping preset '{preset_name}': pickle not found at {input_path}")
                continue
            process_one(settings, input_path, output_path)
            done += 1
        print(f"Completed CSV processing for {done} preset(s).")
        return

    settings = get_active_settings()
    parser = argparse.ArgumentParser(description="Process pickle experiment results into CSV.")
    parser.add_argument("--input", type=str, default=settings["pickle_path"])
    parser.add_argument("--output", type=str, default=settings["output_csv"])
    args = parser.parse_args()
    process_one(settings, Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
