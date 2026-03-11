"""Generate matplotlib figures from processed CSV experiment results.

This is step 4 of the QEC experiment pipeline.

Responsibilities:
- read the CSV file produced by `process_results.py`
- generate the requested plots
- write those plots into the results figures directory

This file does not generate configs or run simulations.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment_settings import (
    get_active_settings,
    get_effective_preset_names,
    get_settings_for_preset,
    is_all_presets_active,
)


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {
        "experiment",
        "code_label",
        "sweep_value",
        "avg_link_logical_fidelity",
        "end_to_end_logical_fidelity",
        "link",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"CSV is missing required columns: {sorted(missing)}")
    return df


def load_pair_data(pair_csv_path: Path) -> pd.DataFrame | None:
    if not pair_csv_path.exists():
        return None
    df = pd.read_csv(pair_csv_path)
    required = {
        "experiment",
        "code_label",
        "pair_index",
        "link",
        "alice_wait_time_sec",
        "bob_wait_time_sec",
        "initial_pair_fidelity",
        "post_idle_pair_fidelity",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Pair CSV is missing required columns: {sorted(missing)}")
    return df


def end_to_end_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["link"] == "end_to_end"].copy()


def set_log_x_if_positive(ax, values, label: str):
    values = pd.Series(values).dropna()
    if values.empty:
        return
    if (values > 0).all() and values.nunique() > 1:
        ax.set_xscale("log")
    else:
        print(f"Skipping log x-scale for {label}: values must be positive and non-constant.")


def set_padded_fidelity_ylim(ax, values, padding_frac: float = 0.12, min_pad: float = 0.01):
    values = pd.Series(values).dropna()
    if values.empty:
        ax.set_ylim(0.0, 1.0)
        return

    vmin = float(values.min())
    vmax = float(values.max())
    span = vmax - vmin
    pad = max(span * padding_frac, min_pad)

    lower = max(0.0, vmin - pad)
    upper = min(1.0, vmax + pad)

    if upper - lower < 2 * min_pad:
        center = 0.5 * (vmin + vmax)
        lower = max(0.0, center - min_pad)
        upper = min(1.0, center + min_pad)

    ax.set_ylim(lower, upper)


def plot_code_compare(df: pd.DataFrame, output_dir: Path):
    sub = end_to_end_rows(df)
    sub = sub[sub["experiment"] == "code_compare"].copy()
    if sub.empty:
        return

    sub = sub.sort_values("code_label")
    x = range(len(sub))
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(x, sub["end_to_end_logical_fidelity"], marker="o", linewidth=2, label="End-to-end")
    ax.plot(x, sub["avg_link_logical_fidelity"], marker="s", linewidth=2, label="Average link")
    ax.set_xticks(list(x))
    ax.set_xticklabels(sub["code_label"], rotation=20, ha="right")
    set_padded_fidelity_ylim(
        ax,
        pd.concat([sub["end_to_end_logical_fidelity"], sub["avg_link_logical_fidelity"]]),
    )
    ax.set_ylabel("Fidelity")
    ax.set_title(_title_with_params("CSS Code Comparison", sub))
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "code_compare_lines.png", dpi=300)
    plt.close()


def _format_constant_params(sub: pd.DataFrame) -> str:
    parts = []
    if "physical_bell_fidelity" in sub.columns:
        vals = sorted(v for v in sub["physical_bell_fidelity"].dropna().unique())
        if len(vals) == 1:
            parts.append(f"Fphys={vals[0]:.4f}")
    if "single_qubit_gate_fidelity" in sub.columns:
        vals = sorted(v for v in sub["single_qubit_gate_fidelity"].dropna().unique())
        if len(vals) == 1:
            parts.append(f"1Q={vals[0]:.4f}")
    if "two_qubit_gate_fidelity" in sub.columns:
        vals = sorted(v for v in sub["two_qubit_gate_fidelity"].dropna().unique())
        if len(vals) == 1:
            parts.append(f"2Q={vals[0]:.4f}")
    if "coherence_time_seconds" in sub.columns:
        vals = sorted(v for v in sub["coherence_time_seconds"].dropna().unique())
        if len(vals) == 1:
            parts.append(f"T2={vals[0]:g}s")
    if "num_links" in sub.columns:
        vals = sorted(v for v in sub["num_links"].dropna().unique())
        if len(vals) == 1:
            parts.append(f"links={int(vals[0])}")
    if "elementary_link_distance_km" in sub.columns:
        vals = sorted(v for v in sub["elementary_link_distance_km"].dropna().unique())
        if len(vals) == 1:
            parts.append(f"dist={vals[0]:g} km")
    return ", ".join(parts)


def _title_with_params(base_title: str, sub: pd.DataFrame) -> str:
    params = _format_constant_params(sub)
    if params:
        return f"{base_title}\n{params}"
    return base_title


def plot_initial_fidelity_sweep(df: pd.DataFrame, output_dir: Path):
    sub = end_to_end_rows(df)
    sub = sub[sub["experiment"] == "initial_fidelity_sweep"].copy()
    if sub.empty:
        return

    # Log-scaling fidelity directly is not informative near 1. Plot physical infidelity instead.
    x_inf = (1.0 - sub["sweep_value"].astype(float)).clip(lower=1e-6)
    sub = sub.assign(x_infidelity=x_inf)

    fig, ax = plt.subplots(figsize=(11, 6))
    for code_label, code_df in sub.groupby("code_label"):
        code_df = code_df.sort_values("x_infidelity")
        ax.plot(code_df["x_infidelity"], code_df["end_to_end_logical_fidelity"], marker="o", linewidth=2, label=code_label)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Initial Physical Bell Infidelity (1 - F_phys)")
    ax.set_ylabel("End-to-End Logical Fidelity")
    set_padded_fidelity_ylim(ax, sub["end_to_end_logical_fidelity"])
    ax.set_title(_title_with_params("Initial Physical Fidelity Sweep", sub))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "initial_fidelity_sweep.png", dpi=300)
    plt.close()


def plot_gate_sweep(df: pd.DataFrame, output_dir: Path, experiment_name: str, x_label: str, prefix: str):
    sub_end = end_to_end_rows(df)
    sub_end = sub_end[sub_end["experiment"] == experiment_name].copy()
    if sub_end.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    for code_label, code_df in sub_end.groupby("code_label"):
        code_df = code_df.sort_values("sweep_value")
        ax.plot(code_df["sweep_value"], code_df["end_to_end_logical_fidelity"], marker="o", linewidth=2, label=code_label)
    ax.set_xlabel(x_label)
    ax.set_ylabel("End-to-End Logical Fidelity")
    set_padded_fidelity_ylim(ax, sub_end["end_to_end_logical_fidelity"])
    ax.set_title(_title_with_params(f"{x_label} Sweep: End-to-End", sub_end))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_end_to_end.png", dpi=300)
    plt.close()

    sub_link = df[(df["experiment"] == experiment_name) & (df["link"] != "end_to_end")].copy()
    if sub_link.empty:
        return

    grouped = sub_link.groupby(["code_label", "sweep_value"])["logical_link_fidelity"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(11, 6))
    for code_label, code_df in grouped.groupby("code_label"):
        code_df = code_df.sort_values("sweep_value")
        ax.plot(code_df["sweep_value"], code_df["logical_link_fidelity"], marker="s", linewidth=2, label=code_label)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Average Single-Link Logical Fidelity")
    set_padded_fidelity_ylim(ax, grouped["logical_link_fidelity"])
    ax.set_title(_title_with_params(f"{x_label} Sweep: Average Link", sub_link))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_average_link.png", dpi=300)
    plt.close()


def plot_distance_sweep_end_to_end(df: pd.DataFrame, output_dir: Path):
    sub = end_to_end_rows(df)
    sub = sub[sub["experiment"] == "distance_sweep"].copy()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    for code_label, code_df in sub.groupby("code_label"):
        code_df = code_df.sort_values("sweep_value")
        ax.plot(code_df["sweep_value"], code_df["end_to_end_logical_fidelity"], marker="o", linewidth=2, label=code_label)
    ax.set_xlabel("Elementary-Link Distance (km)")
    ax.set_ylabel("End-to-End Logical Fidelity")
    set_padded_fidelity_ylim(ax, sub["end_to_end_logical_fidelity"])
    ax.set_title(_title_with_params("End-to-End Fidelity vs Elementary-Link Distance", sub))
    set_log_x_if_positive(ax, sub["sweep_value"], "distance_sweep_end_to_end")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "distance_sweep_end_to_end.png", dpi=300)
    plt.close()


def plot_distance_sweep_average_link(df: pd.DataFrame, output_dir: Path):
    sub = df[(df["experiment"] == "distance_sweep") & (df["link"] != "end_to_end")].copy()
    if sub.empty:
        return

    grouped = sub.groupby(["code_label", "sweep_value"])["logical_link_fidelity"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(11, 6))
    for code_label, code_df in grouped.groupby("code_label"):
        code_df = code_df.sort_values("sweep_value")
        ax.plot(code_df["sweep_value"], code_df["logical_link_fidelity"], marker="s", linewidth=2, label=code_label)
    ax.set_xlabel("Elementary-Link Distance (km)")
    ax.set_ylabel("Average Link Logical Fidelity")
    set_padded_fidelity_ylim(ax, grouped["logical_link_fidelity"])
    ax.set_title(_title_with_params("Average Link Fidelity vs Elementary-Link Distance", sub))
    set_log_x_if_positive(ax, grouped["sweep_value"], "distance_sweep_average_link")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "distance_sweep_average_link.png", dpi=300)
    plt.close()


def plot_link_count_sweep_end_to_end(df: pd.DataFrame, output_dir: Path):
    sub = end_to_end_rows(df)
    sub = sub[sub["experiment"] == "link_count_sweep"].copy()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    for code_label, code_df in sub.groupby("code_label"):
        code_df = code_df.sort_values("sweep_value")
        ax.plot(code_df["sweep_value"], code_df["end_to_end_logical_fidelity"], marker="o", linewidth=2, label=code_label)
    ax.set_xlabel("Number of Links")
    ax.set_ylabel("End-to-End Logical Fidelity")
    set_padded_fidelity_ylim(ax, sub["end_to_end_logical_fidelity"])
    ax.set_title(_title_with_params("End-to-End Fidelity vs Number of Links", sub))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "link_count_sweep_end_to_end.png", dpi=300)
    plt.close()


def plot_coherence_sweep_end_to_end(df: pd.DataFrame, output_dir: Path):
    sub = end_to_end_rows(df)
    sub = sub[sub["experiment"] == "coherence_sweep"].copy()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    for code_label, code_df in sub.groupby("code_label"):
        code_df = code_df.sort_values("sweep_value")
        ax.plot(code_df["sweep_value"], code_df["end_to_end_logical_fidelity"], marker="o", linewidth=2, label=code_label)
    ax.set_xlabel("Memory Coherence Time (s)")
    ax.set_ylabel("End-to-End Logical Fidelity")
    set_padded_fidelity_ylim(ax, sub["end_to_end_logical_fidelity"])
    ax.set_title(_title_with_params("End-to-End Fidelity vs Memory Coherence Time", sub))
    set_log_x_if_positive(ax, sub["sweep_value"], "coherence_sweep_end_to_end")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "coherence_sweep_end_to_end.png", dpi=300)
    plt.close()


def plot_coherence_sweep_average_link(df: pd.DataFrame, output_dir: Path):
    sub = df[(df["experiment"] == "coherence_sweep") & (df["link"] != "end_to_end")].copy()
    if sub.empty:
        return

    grouped = sub.groupby(["code_label", "sweep_value"])["logical_link_fidelity"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(11, 6))
    for code_label, code_df in grouped.groupby("code_label"):
        code_df = code_df.sort_values("sweep_value")
        ax.plot(code_df["sweep_value"], code_df["logical_link_fidelity"], marker="s", linewidth=2, label=code_label)
    ax.set_xlabel("Memory Coherence Time (s)")
    ax.set_ylabel("Average Link Logical Fidelity")
    set_padded_fidelity_ylim(ax, grouped["logical_link_fidelity"])
    ax.set_title(_title_with_params("Average Link Fidelity vs Memory Coherence Time", sub))
    set_log_x_if_positive(ax, grouped["sweep_value"], "coherence_sweep_average_link")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "coherence_sweep_average_link.png", dpi=300)
    plt.close()


def plot_pair_post_idle_fidelity_by_index(
    pair_df: pd.DataFrame,
    output_dir: Path,
    context_df: pd.DataFrame | None = None,
):
    if pair_df is None or pair_df.empty:
        return

    sub = pair_df[pair_df["experiment"] == "coherence_sweep"].copy()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    for (code_label, sweep_value), code_df in sub.groupby(["code_label", "sweep_value"]):
        grouped = code_df.groupby("pair_index")["post_idle_pair_fidelity"].mean().reset_index()
        grouped = grouped.sort_values("pair_index")
        ax.plot(
            grouped["pair_index"],
            grouped["post_idle_pair_fidelity"],
            marker="o",
            linewidth=2,
            label=f"{code_label} (T2={sweep_value})",
        )
    ax.set_xlabel("Pair Index")
    ax.set_ylabel("Post-Idle Physical Bell Fidelity")
    set_padded_fidelity_ylim(ax, sub["post_idle_pair_fidelity"])
    title_df = sub
    if context_df is not None and not context_df.empty:
        ctx = context_df[context_df["experiment"] == "coherence_sweep"].copy()
        if not ctx.empty:
            title_df = ctx
    ax.set_title(_title_with_params("Post-Idle Pair Fidelity vs Pair Index", title_df))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "pair_post_idle_fidelity_by_index.png", dpi=300)
    plt.close()


def plot_pair_post_idle_fidelity_vs_wait(
    pair_df: pd.DataFrame,
    output_dir: Path,
    context_df: pd.DataFrame | None = None,
):
    if pair_df is None or pair_df.empty:
        return

    sub = pair_df[pair_df["experiment"] == "coherence_sweep"].copy()
    if sub.empty:
        return

    sub["total_wait_time_sec"] = sub["alice_wait_time_sec"].fillna(0) + sub["bob_wait_time_sec"].fillna(0)

    fig, ax = plt.subplots(figsize=(11, 6))
    for (code_label, sweep_value), code_df in sub.groupby(["code_label", "sweep_value"]):
        code_df = code_df.sort_values("total_wait_time_sec")
        x = code_df["total_wait_time_sec"].to_numpy()
        y = code_df["post_idle_pair_fidelity"].to_numpy()

        # Show raw samples faintly, then overlay a binned mean trend.
        ax.scatter(
            x,
            y,
            s=25,
            alpha=0.25,
        )

        if len(code_df) >= 4 and np.nanmax(x) > np.nanmin(x):
            num_bins = min(10, max(4, len(code_df) // 3))
            bins = np.linspace(np.nanmin(x), np.nanmax(x), num_bins + 1)
            code_df = code_df.copy()
            code_df["wait_bin"] = pd.cut(
                code_df["total_wait_time_sec"],
                bins=bins,
                include_lowest=True,
                duplicates="drop",
            )
            smoothed = (
                code_df.groupby("wait_bin", observed=False)
                .agg(
                    total_wait_time_sec=("total_wait_time_sec", "mean"),
                    post_idle_pair_fidelity=("post_idle_pair_fidelity", "mean"),
                )
                .dropna()
                .reset_index(drop=True)
            )
            x_plot = smoothed["total_wait_time_sec"]
            y_plot = smoothed["post_idle_pair_fidelity"]
        else:
            grouped = (
                code_df.groupby("total_wait_time_sec", as_index=False)["post_idle_pair_fidelity"]
                .mean()
                .sort_values("total_wait_time_sec")
            )
            x_plot = grouped["total_wait_time_sec"]
            y_plot = grouped["post_idle_pair_fidelity"]

        ax.plot(
            x_plot,
            y_plot,
            marker="o",
            linewidth=2.5,
            label=f"{code_label} (T2={sweep_value})",
        )
    ax.set_xlabel("Total Pair Wait Time (s)")
    ax.set_ylabel("Post-Idle Physical Bell Fidelity")
    set_padded_fidelity_ylim(ax, sub["post_idle_pair_fidelity"])
    title_df = sub
    if context_df is not None and not context_df.empty:
        ctx = context_df[context_df["experiment"] == "coherence_sweep"].copy()
        if not ctx.empty:
            title_df = ctx
    ax.set_title(_title_with_params("Post-Idle Pair Fidelity vs Total Wait Time", title_df))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "pair_post_idle_fidelity_vs_wait.png", dpi=300)
    plt.close()


def plot_one(settings: dict, input_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(input_path)
    pair_df = load_pair_data(input_path.with_name(f"{input_path.stem}_pairs.csv"))
    plot_code_compare(df, output_dir)
    plot_initial_fidelity_sweep(df, output_dir)
    plot_gate_sweep(df, output_dir, "two_qubit_gate_sweep", "Two-Qubit Gate Fidelity", "two_qubit_gate")
    plot_gate_sweep(df, output_dir, "single_qubit_gate_sweep", "Single-Qubit Gate Fidelity", "single_qubit_gate")
    plot_distance_sweep_end_to_end(df, output_dir)
    plot_distance_sweep_average_link(df, output_dir)
    plot_link_count_sweep_end_to_end(df, output_dir)
    plot_coherence_sweep_end_to_end(df, output_dir)
    plot_coherence_sweep_average_link(df, output_dir)
    plot_pair_post_idle_fidelity_by_index(pair_df, output_dir, context_df=df)
    plot_pair_post_idle_fidelity_vs_wait(pair_df, output_dir, context_df=df)
    print(f"Preset: {settings['active_preset']}")
    print(f"Wrote figures to {output_dir}")


def main():
    if is_all_presets_active() and len(sys.argv) == 1:
        done = 0
        for preset_name in get_effective_preset_names():
            settings = get_settings_for_preset(preset_name)
            input_path = Path(settings["output_csv"])
            output_dir = Path(settings["output_figures"])
            if not input_path.exists():
                print(f"Skipping preset '{preset_name}': CSV not found at {input_path}")
                continue
            plot_one(settings, input_path, output_dir)
            done += 1
        print(f"Completed plotting for {done} preset(s).")
        return

    settings = get_active_settings()
    parser = argparse.ArgumentParser(description="Plot experiment CSV results.")
    parser.add_argument("--input", type=str, default=settings["output_csv"])
    parser.add_argument("--output-dir", type=str, default=settings["output_figures"])
    args = parser.parse_args()
    plot_one(settings, Path(args.input), Path(args.output_dir))


if __name__ == "__main__":
    main()
