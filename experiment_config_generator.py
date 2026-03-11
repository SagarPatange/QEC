"""Generate experiment-specific JSON configs and a manifest.

This is step 1 of the QEC experiment pipeline.

Responsibilities:
- read the active preset from `experiment_settings.py`
- generate concrete JSON config files for each sweep point
- write a manifest listing every generated config and its metadata

This file does not run simulations, process results, or make plots.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from experiment_settings import (
    get_active_settings,
    get_effective_preset_names,
    get_settings_for_preset,
    is_all_presets_active,
)
from main_test import resolve_config_path


CODE_CONFIGS = [
    ("Steane [[7,1,3]]", "[[7,1,3]]", "config/line_5_2G_near_term.json"),
    ("Shor [[9,1,3]]", "[[9,1,3]]", "config/line_5_2G_near_term_shor.json"),
    ("Reed-Muller [[15,1,3]]", "[[15,1,3]]", "config/line_5_2G_near_term_rm15.json"),
    ("Golay [[23,1,7]]", "[[23,1,7]]", "config/line_5_2G_near_term_golay.json"),
    ("BCH [[31,1,7]]", "[[31,1,7]]", "config/line_5_2G_near_term_bch31.json"),
]


def select_code_configs(selected_codes: list[str] | None):
    if not selected_codes:
        return CODE_CONFIGS
    selected = set(selected_codes)
    return [row for row in CODE_CONFIGS if row[1] in selected or row[0] in selected]


def load_config(config_file: str):
    path = Path(resolve_config_path(config_file))
    return json.loads(path.read_text()), path


def update_memory_fidelity(config: dict, fidelity: float):
    template_name = next(iter(config["templates"].keys()))
    template = config["templates"][template_name]
    if "memory" in template:
        template["memory"]["fidelity"] = fidelity
    if "MemoryArray" in template:
        template["MemoryArray"]["fidelity"] = fidelity
    if "DataMemoryArray" in template:
        template["DataMemoryArray"]["fidelity"] = fidelity
    if "AncillaMemoryArray" in template:
        template["AncillaMemoryArray"]["fidelity"] = fidelity


def update_memory_coherence_time(config: dict, coherence_time: float):
    template_name = next(iter(config["templates"].keys()))
    template = config["templates"][template_name]
    if "memory" in template:
        template["memory"]["coherence_time"] = coherence_time
    if "MemoryArray" in template:
        template["MemoryArray"]["coherence_time"] = coherence_time
    if "DataMemoryArray" in template:
        template["DataMemoryArray"]["coherence_time"] = coherence_time
    if "AncillaMemoryArray" in template:
        template["AncillaMemoryArray"]["coherence_time"] = coherence_time


def update_idle_decoherence_debug(config: dict, enabled: bool):
    for node in config["nodes"]:
        if node.get("type") != "QuantumRouter":
            continue
        node["idle_decoherence_debug"] = bool(enabled)


def update_gate_fidelities(config: dict, oneq: float | None = None, twoq: float | None = None):
    for node in config["nodes"]:
        if node.get("type") != "QuantumRouter":
            continue
        if oneq is not None:
            node["gate_fidelity"] = oneq
        if twoq is not None:
            node["two_qubit_gate_fidelity"] = twoq


def write_variant_config(out_dir: Path, base_config_file: str, tag: str, config: dict) -> str:
    base_name = Path(base_config_file).stem
    out_path = out_dir / f"{base_name}_{tag}.json"
    out_path.write_text(json.dumps(config, indent=4) + "\n")
    return str(out_path)


def manifest_entry(
    experiment: str,
    code_label: str,
    css_code: str,
    base_config_file: str,
    generated_config_file: str,
    sweep_parameter: str,
    sweep_value: float,
    num_nodes: int | None = None,
    num_links: int | None = None,
    elementary_link_distance_km: float | None = None,
):
    return {
        "experiment": experiment,
        "code_label": code_label,
        "css_code": css_code,
        "base_config_file": base_config_file,
        "config_file": generated_config_file,
        "sweep_parameter": sweep_parameter,
        "sweep_value": sweep_value,
        "num_nodes": num_nodes,
        "num_links": num_links,
        "elementary_link_distance_km": elementary_link_distance_km,
    }


def build_code_compare(code_configs, out_dir: Path, idle_decoherence_debug: bool = False) -> list[dict]:
    rows = []
    for code_label, css_code, config_file in code_configs:
        config, _ = load_config(config_file)
        apply_common_config_overrides(config, idle_decoherence_debug)
        generated = write_variant_config(out_dir, config_file, "base", config)
        rows.append(
            manifest_entry("code_compare", code_label, css_code, config_file, generated, "code_index", 0.0)
        )
    return rows


def build_initial_fidelity_sweep(code_configs, values, out_dir: Path, idle_decoherence_debug: bool = False) -> list[dict]:
    rows = []
    for code_label, css_code, config_file in code_configs:
        base_config, _ = load_config(config_file)
        for fidelity in values:
            config = json.loads(json.dumps(base_config))
            update_memory_fidelity(config, fidelity)
            apply_common_config_overrides(config, idle_decoherence_debug)
            generated = write_variant_config(out_dir, config_file, f"physfid_{str(fidelity).replace('.', '_')}", config)
            rows.append(
                manifest_entry(
                    "initial_fidelity_sweep",
                    code_label,
                    css_code,
                    config_file,
                    generated,
                    "physical_bell_fidelity",
                    fidelity,
                    num_nodes=len([n for n in config["nodes"] if n["type"] == "QuantumRouter"]),
                    num_links=len([n for n in config["nodes"] if n["type"] == "QuantumRouter"]) - 1,
                )
            )
    return rows


def build_two_qubit_sweep(code_configs, values, out_dir: Path, idle_decoherence_debug: bool = False) -> list[dict]:
    rows = []
    for code_label, css_code, config_file in code_configs:
        base_config, _ = load_config(config_file)
        for fidelity in values:
            config = json.loads(json.dumps(base_config))
            update_gate_fidelities(config, twoq=fidelity)
            apply_common_config_overrides(config, idle_decoherence_debug)
            generated = write_variant_config(out_dir, config_file, f"twoq_{str(fidelity).replace('.', '_')}", config)
            rows.append(
                manifest_entry(
                    "two_qubit_gate_sweep",
                    code_label,
                    css_code,
                    config_file,
                    generated,
                    "two_qubit_gate_fidelity",
                    fidelity,
                    num_nodes=len([n for n in config["nodes"] if n["type"] == "QuantumRouter"]),
                    num_links=len([n for n in config["nodes"] if n["type"] == "QuantumRouter"]) - 1,
                )
            )
    return rows


def build_single_qubit_sweep(code_configs, values, out_dir: Path, idle_decoherence_debug: bool = False) -> list[dict]:
    rows = []
    for code_label, css_code, config_file in code_configs:
        base_config, _ = load_config(config_file)
        for fidelity in values:
            config = json.loads(json.dumps(base_config))
            update_gate_fidelities(config, oneq=fidelity)
            apply_common_config_overrides(config, idle_decoherence_debug)
            generated = write_variant_config(out_dir, config_file, f"oneq_{str(fidelity).replace('.', '_')}", config)
            rows.append(
                manifest_entry(
                    "single_qubit_gate_sweep",
                    code_label,
                    css_code,
                    config_file,
                    generated,
                    "single_qubit_gate_fidelity",
                    fidelity,
                    num_nodes=len([n for n in config["nodes"] if n["type"] == "QuantumRouter"]),
                    num_links=len([n for n in config["nodes"] if n["type"] == "QuantumRouter"]) - 1,
                )
            )
    return rows


def build_coherence_sweep(code_configs, values, out_dir: Path, idle_decoherence_debug: bool = False) -> list[dict]:
    rows = []
    for code_label, css_code, config_file in code_configs:
        base_config, _ = load_config(config_file)
        for coherence_time in values:
            config = json.loads(json.dumps(base_config))
            update_memory_coherence_time(config, coherence_time)
            apply_common_config_overrides(config, idle_decoherence_debug)
            generated = write_variant_config(out_dir, config_file, f"coh_{str(coherence_time).replace('.', '_')}", config)
            rows.append(
                manifest_entry(
                    "coherence_sweep",
                    code_label,
                    css_code,
                    config_file,
                    generated,
                    "coherence_time_seconds",
                    coherence_time,
                    num_nodes=len([n for n in config["nodes"] if n["type"] == "QuantumRouter"]),
                    num_links=len([n for n in config["nodes"] if n["type"] == "QuantumRouter"]) - 1,
                )
            )
    return rows


def update_elementary_link_distance(config: dict, distance_km: float):
    half_distance_m = distance_km * 1000 / 2
    for chan in config["qchannels"]:
        chan["distance"] = half_distance_m
    for chan in config["cchannels"]:
        if "distance" in chan:
            chan["distance"] = half_distance_m


def infer_gate_fidelities_from_base(base_config_file: str):
    config, _ = load_config(base_config_file)
    routers = [n for n in config["nodes"] if n["type"] == "QuantumRouter"]
    return routers[0].get("gate_fidelity", 1.0), routers[0].get("two_qubit_gate_fidelity", 1.0)


def generate_linear_2g_config(
    out_path: Path,
    linear_size: int,
    css_code: str,
    gate_fid: float,
    twoq_fid: float,
    qc_length_km: float = 1.0,
    idle_decoherence_debug: bool = False,
):
    script = Path(__file__).resolve().parent / "config" / "config_generator_line_2G.py"
    memo_size = int(css_code.split(",")[0].strip("["))
    cmd = [
        sys.executable,
        str(script),
        str(linear_size),
        str(memo_size),
        str(qc_length_km),
        "0.0002",
        "-1",
        "-d",
        str(out_path.parent),
        "-o",
        out_path.name,
        "-s",
        "10",
        "--gen2",
        "--css_code",
        css_code,
        "--gate_fid",
        str(gate_fid),
        "--two_qubit_gate_fid",
        str(twoq_fid),
    ]
    subprocess.run(cmd, check=True)
    if idle_decoherence_debug:
        generated_config = json.loads(out_path.read_text())
        apply_common_config_overrides(generated_config, idle_decoherence_debug)
        out_path.write_text(json.dumps(generated_config, indent=4) + "\n")


def apply_common_config_overrides(config: dict, idle_decoherence_debug: bool):
    update_idle_decoherence_debug(config, idle_decoherence_debug)


def build_distance_sweep(code_configs, values, out_dir: Path, idle_decoherence_debug: bool = False) -> list[dict]:
    rows = []
    for code_label, css_code, config_file in code_configs:
        base_config, _ = load_config(config_file)
        num_nodes = len([n for n in base_config["nodes"] if n["type"] == "QuantumRouter"])
        num_links = num_nodes - 1
        for distance_km in values:
            config = json.loads(json.dumps(base_config))
            update_elementary_link_distance(config, distance_km)
            apply_common_config_overrides(config, idle_decoherence_debug)
            generated = write_variant_config(out_dir, config_file, f"dist_{str(distance_km).replace('.', '_')}", config)
            rows.append(
                manifest_entry(
                    "distance_sweep",
                    code_label,
                    css_code,
                    config_file,
                    generated,
                    "elementary_link_distance_km",
                    distance_km,
                    num_nodes=num_nodes,
                    num_links=num_links,
                    elementary_link_distance_km=distance_km,
                )
            )
    return rows


def base_config_for_code_and_size(css_code: str, linear_size: int) -> str | None:
    mapping = {
        ("[[7,1,3]]", 3): "config/line_3_2G.json",
        ("[[7,1,3]]", 5): "config/line_5_2G_near_term.json",
        ("[[9,1,3]]", 3): "config/line_3_2G_shor.json",
        ("[[9,1,3]]", 5): "config/line_5_2G_near_term_shor.json",
    }
    return mapping.get((css_code, linear_size))


def build_link_count_sweep(code_configs, linear_sizes, distance_values_km, out_dir: Path, idle_decoherence_debug: bool = False) -> list[dict]:
    rows = []
    fixed_distance_km = list(distance_values_km)[0]
    for code_label, css_code, config_file in code_configs:
        gate_fid, twoq_fid = infer_gate_fidelities_from_base(config_file)
        for linear_size in linear_sizes:
            base_config_file = base_config_for_code_and_size(css_code, linear_size)
            if base_config_file is not None:
                base_config, _ = load_config(base_config_file)
                update_gate_fidelities(base_config, oneq=gate_fid, twoq=twoq_fid)
                update_elementary_link_distance(base_config, fixed_distance_km)
                apply_common_config_overrides(base_config, idle_decoherence_debug)
                generated = write_variant_config(out_dir, base_config_file, f"links_{linear_size-1}", base_config)
            else:
                generated_path = out_dir / f"line_{linear_size}_{css_code.replace('[', '').replace(']', '').replace(',', '_')}.json"
                generate_linear_2g_config(
                    generated_path,
                    linear_size,
                    css_code,
                    gate_fid,
                    twoq_fid,
                    qc_length_km=fixed_distance_km,
                    idle_decoherence_debug=idle_decoherence_debug,
                )
                generated = str(generated_path)
                base_config_file = generated

            rows.append(
                manifest_entry(
                    "link_count_sweep",
                    code_label,
                    css_code,
                    base_config_file,
                    generated,
                    "num_links",
                    linear_size - 1,
                    num_nodes=linear_size,
                    num_links=linear_size - 1,
                    elementary_link_distance_km=fixed_distance_km,
                )
            )
    return rows


def run_generation(settings: dict, args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    code_configs = select_code_configs(args.codes)
    manifest_entries = []

    if args.experiment in ("code_compare", "all"):
        manifest_entries.extend(build_code_compare(code_configs, output_dir, args.idle_decoherence_debug))
    if args.experiment in ("initial_fidelity_sweep", "all"):
        manifest_entries.extend(
            build_initial_fidelity_sweep(code_configs, args.initial_fidelity_values, output_dir, args.idle_decoherence_debug)
        )
    if args.experiment in ("two_qubit_gate_sweep", "all"):
        manifest_entries.extend(build_two_qubit_sweep(code_configs, args.twoq_values, output_dir, args.idle_decoherence_debug))
    if args.experiment in ("single_qubit_gate_sweep", "all"):
        manifest_entries.extend(
            build_single_qubit_sweep(code_configs, args.oneq_values, output_dir, args.idle_decoherence_debug)
        )
    if args.experiment in ("coherence_sweep", "all"):
        manifest_entries.extend(
            build_coherence_sweep(code_configs, args.coherence_time_values, output_dir, args.idle_decoherence_debug)
        )
    if args.experiment in ("distance_sweep", "all"):
        manifest_entries.extend(build_distance_sweep(code_configs, args.distance_values_km, output_dir, args.idle_decoherence_debug))
    if args.experiment in ("link_count_sweep", "all"):
        manifest_entries.extend(
            build_link_count_sweep(
                code_configs,
                args.linear_sizes,
                args.distance_values_km,
                output_dir,
                args.idle_decoherence_debug,
            )
        )

    manifest = {
        "preset": settings["active_preset"],
        "experiment": args.experiment,
        "config_output_dir": str(output_dir),
        "entries": manifest_entries,
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=4) + "\n")

    print(f"Preset: {settings['active_preset']}")
    print(f"Generated {len(manifest_entries)} config entries")
    print(f"Config directory: {output_dir}")
    print(f"Manifest: {manifest_path}")


def build_default_args_from_settings(settings: dict):
    return SimpleNamespace(
        experiment=settings["experiment"],
        initial_fidelity_values=settings["initial_fidelity_values"],
        twoq_values=settings["twoq_values"],
        oneq_values=settings["oneq_values"],
        distance_values_km=settings["distance_values_km"],
        linear_sizes=settings["linear_sizes"],
        coherence_time_values=settings["coherence_time_values"],
        idle_decoherence_debug=settings.get("idle_decoherence_debug", False),
        codes=settings["codes"],
        output_dir=settings["config_output_dir"],
        manifest=settings["manifest_path"],
    )


def main():
    if is_all_presets_active() and len(sys.argv) == 1:
        total = 0
        for preset_name in get_effective_preset_names():
            settings = get_settings_for_preset(preset_name)
            args = build_default_args_from_settings(settings)
            run_generation(settings, args)
            total += 1
        print(f"Completed config generation for {total} preset(s).")
        return

    settings = get_active_settings()
    parser = argparse.ArgumentParser(description="Generate experiment config JSONs and a manifest.")
    parser.add_argument(
        "--experiment",
        choices=["code_compare", "initial_fidelity_sweep", "two_qubit_gate_sweep", "single_qubit_gate_sweep", "distance_sweep", "link_count_sweep", "coherence_sweep", "all"],
        default=settings["experiment"],
    )
    parser.add_argument("--initial-fidelity-values", nargs="*", type=float, default=settings["initial_fidelity_values"])
    parser.add_argument("--twoq-values", nargs="*", type=float, default=settings["twoq_values"])
    parser.add_argument("--oneq-values", nargs="*", type=float, default=settings["oneq_values"])
    parser.add_argument("--distance-values-km", nargs="*", type=float, default=settings["distance_values_km"])
    parser.add_argument("--linear-sizes", nargs="*", type=int, default=settings["linear_sizes"])
    parser.add_argument("--coherence-time-values", nargs="*", type=float, default=settings["coherence_time_values"])
    parser.add_argument(
        "--idle-decoherence-debug",
        action="store_true",
        default=settings.get("idle_decoherence_debug", False),
        help="Enable verbose idle-decoherence logging in generated router configs.",
    )
    parser.add_argument("--codes", nargs="*", default=settings["codes"])
    parser.add_argument("--output-dir", type=str, default=settings["config_output_dir"])
    parser.add_argument("--manifest", type=str, default=settings["manifest_path"])
    args = parser.parse_args()
    run_generation(settings, args)


if __name__ == "__main__":
    main()
