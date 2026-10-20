"""Measured solid-line small multiples; compatible with v8.8 and v8.9 exports.

Historical fronts and frozen final fronts are separate. No smoothing,
extrapolation, endpoint scatter, selective seed removal or metric edits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

DATASETS = {
    "bbq": "BBQ",
    "civil_comments": "Civil Comments",
    "bias_in_bios": "Bias-in-Bios",
    "winobias": "WinoBias",
}
STYLE = {
    "Tri-Fair-v8.9": ("Tri-Fair v8.9", "#000000"),
    "Tri-Fair-v8.8": ("Tri-Fair v8.8", "#000000"),
    "NSGAII-PO-Fair": ("NSGA-II", "#E69F00"),
    "NSGAIII-PO-Fair": ("NSGA-III", "#785EF0"),
    "SPEA2-PO-Fair": ("SPEA2", "#009E73"),
    "RandomSearch-PO-Fair": ("Random Search", "#CC79A7"),
}
CSV_NAMES = (
    "measured_final_indicators.csv",
    "measured_historical_trajectories.csv",
    "fewshot_measured_points.csv",
)
TOKEN_LABEL = "Token Budget [×10⁶]"
COST_LABEL = "Avg. Cost [$] per 1M Calls"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observed_summary(group, metric):
    """Previous observation on common observed seed support, never extrapolated."""
    curves = []
    for _, g in group.groupby("seed"):
        g = g.sort_values("actual_budget_tokens").drop_duplicates(
            "actual_budget_tokens", keep="last"
        )
        curves.append(
            (g.actual_budget_tokens.to_numpy(float) / 1e6, g[metric].to_numpy(float))
        )
    if not curves:
        return np.array([]), np.array([]), None
    grid = np.unique(np.concatenate([x for x, _ in curves]))
    stack = []
    for x, y in curves:
        indices = np.searchsorted(x, grid, side="right") - 1
        values = np.full(len(grid), np.nan)
        valid = (indices >= 0) & (grid <= x[-1])
        values[valid] = y[indices[valid]]
        stack.append(values)
    stack = np.asarray(stack)
    valid = np.isfinite(stack).all(axis=0)
    return (
        grid[valid],
        stack[:, valid].mean(axis=0),
        (stack[:, valid].std(axis=0, ddof=1) if len(stack) > 1 else None),
    )


def export_measured(batch, dest, model, budget, split):
    from tri_fair.reporting.front_report import analyse
    from tri_fair.reporting.research_figures import final_rows, trajectory_rows

    paths = sorted(
        (Path(batch) / f"heldout_{split}" / model).glob(
            "*/seed*/eval_checkpoints_union.parquet"
        )
    )
    if not paths:
        raise ValueError("No held-out Parquet files in the selected batch")
    frame = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    if set(frame.evaluation_split) != {split} or set(frame.model) != {model}:
        raise ValueError("Mixed evaluation split or model")
    indicators = analyse(frame, split=split, budget=budget)
    history = trajectory_rows(frame, split=split, budget=budget)
    final = final_rows(frame[frame.actual_budget_tokens <= budget]).copy()
    inp, out = final[f"{split}_input_tokens"], final[f"{split}_output_tokens"]
    if (inp + out <= 0).any():
        raise ValueError("Missing measured input/output token counts")
    final["output_token_share"] = out / (inp + out)
    final["few_shot_count"] = final.few_shots_json.map(lambda s: len(json.loads(s)))
    for f, name in zip((indicators, history, final), CSV_NAMES):
        f.to_csv(dest / name, index=False)
    return indicators, history, final


def render(
    *,
    output,
    source_csv=None,
    batch_root=None,
    model="qwen-3-30b",
    split="validation",
    budget=2000000,
    formats=("png", "pdf", "svg"),
):
    output = Path(output).resolve()
    if source_csv and (
        Path(source_csv).resolve() == output
        or Path(source_csv).resolve() in output.parents
    ):
        raise ValueError("Use a separate sibling output directory")
    output.mkdir(parents=True, exist_ok=True)
    if source_csv:
        source = Path(source_csv).resolve()
        hashes = {n: sha(source / n) for n in CSV_NAMES}
        indicators, history, final = [pd.read_csv(source / n) for n in CSV_NAMES]
    else:
        indicators, history, final = export_measured(
            batch_root, output, model, budget, split
        )
        source, hashes = output, {n: sha(output / n) for n in CSV_NAMES}
    keys = ["dataset", "optimizer", "seed"]
    if indicators[keys].duplicated().any() or set(indicators.model) != {model}:
        raise ValueError("Duplicated final run identities or mixed models")
    if set(indicators.evaluation_split) != {split} or set(final.evaluation_split) != {
        split
    }:
        raise ValueError("Incorrect held-out split")
    if (
        not indicators.actual_budget_tokens.le(budget).all()
        or not final.actual_budget_tokens.le(budget).all()
    ):
        raise ValueError("Over-budget figure rows")
    if not final[f"{split}_fairness_ready"].eq(True).all():
        raise ValueError("Incomplete final fairness evidence")
    if set(final.optimizer) - set(STYLE):
        raise ValueError("Unknown optimizer requires an explicit style")
    datasets = [d for d in DATASETS if d in set(indicators.dataset)]
    methods = [m for m in STYLE if m in set(indicators.optimizer)]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    made = []

    def save(fig, name):
        for ext in formats:
            fig.savefig(output / f"{name}.{ext}", dpi=220, bbox_inches="tight")
        plt.close(fig)
        made.append(name)

    for metric, label, name in (
        ("noisy_r2_3d", "nR2 ↓", "trajectory_nR2_small_multiples"),
        ("hv_test_pessimistic_3d", "HVpes ↑", "trajectory_HVpes_small_multiples"),
    ):
        fig, axes = plt.subplots(
            len(methods),
            len(datasets),
            figsize=(3.15 * len(datasets), 1.9 * len(methods) + 1.0),
            squeeze=False,
            sharex=True,
            sharey="col",
        )
        for row, m in enumerate(methods):
            for col, ds in enumerate(datasets):
                ax = axes[row, col]
                group = history[(history.dataset == ds) & (history.optimizer == m)]
                expected = indicators[
                    (indicators.dataset == ds) & (indicators.optimizer == m)
                ].seed.nunique()
                if group.seed.nunique() != expected:
                    raise ValueError(f"Missing historical seed support for {ds}/{m}")
                x, mean, sd = observed_summary(group, metric)
                if len(x):
                    ax.step(
                        x, mean, where="post", color=STYLE[m][1], lw=1.9, linestyle="-"
                    )
                    if sd is not None:
                        ax.fill_between(
                            x,
                            mean - sd,
                            mean + sd,
                            step="post",
                            color=STYLE[m][1],
                            alpha=0.13,
                            lw=0,
                        )
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "No common observed support",
                        ha="center",
                        transform=ax.transAxes,
                        fontsize=8,
                    )
                ax.set_xlim(0, budget / 1e6)
                ax.yaxis.set_major_locator(MaxNLocator(3))
                ax.grid(axis="y", color="#e5e7eb", lw=0.6, linestyle="-")
                if row == 0:
                    ax.set_title(DATASETS[ds], fontsize=12, fontweight="bold")
                if col == 0:
                    ax.set_ylabel(
                        f"{STYLE[m][0]}\n{label}", color=STYLE[m][1], fontsize=10
                    )
                if row == len(methods) - 1:
                    ax.set_xlabel(TOKEN_LABEL)
        fig.subplots_adjust(
            left=0.10, right=0.99, top=0.92, bottom=0.11, hspace=0.22, wspace=0.22
        )
        fig.suptitle(f"{model} · Held-out {split} · {label}", fontsize=15, y=0.985)
        fig.text(
            0.5,
            0.025,
            "Solid lines: historical mean. Bands: sample SD when ≥2 seeds; not confidence intervals.\n"
            "Shared scales within each dataset. No smoothing or extrapolation. Frozen final fronts are reported separately.",
            ha="center",
            fontsize=9,
            color="#4b5563",
        )
        save(fig, name)
    # Named rows separate methods; no overlapping clouds or per-seed endpoint markers.
    summaries = [
        (
            [
                ("hv_test_optimistic_3d", "HVopt ↑"),
                ("hv_test_pessimistic_3d", "HVpes ↑"),
                ("noisy_r2_3d", "nR2 ↓"),
            ],
            "final_front_indicators_named_rows",
            "Frozen final fronts",
        ),
        (
            [
                ("balanced_test_quality", "Validation quality (%) ↑"),
                ("balanced_test_cost", COST_LABEL + " ↓"),
                ("balanced_test_fairness", "Unfairness ↓"),
            ],
            "balanced_objectives_named_rows",
            "Balanced search-selected prompts",
        ),
    ]
    for metrics, name, title in summaries:
        fig, axes = plt.subplots(
            len(datasets), 3, figsize=(12, 2.35 * len(datasets) + 0.5), squeeze=False
        )
        for row, ds in enumerate(datasets):
            for col, (metric, label) in enumerate(metrics):
                ax = axes[row, col]
                for i, m in enumerate(methods):
                    values = indicators[
                        (indicators.dataset == ds) & (indicators.optimizer == m)
                    ][metric].to_numpy(float)
                    if metric == "balanced_test_quality":
                        values = 100 * values
                    if not len(values):
                        raise ValueError(f"Missing final values for {ds}/{m}")
                    sd = values.std(ddof=1) if len(values) > 1 else None
                    ax.errorbar(
                        values.mean(),
                        i,
                        xerr=sd,
                        fmt="o",
                        color=STYLE[m][1],
                        ms=5.5,
                        capsize=3,
                        lw=1.3,
                    )
                ax.set_yticks(
                    range(len(methods)),
                    [STYLE[m][0] for m in methods] if col == 0 else [],
                )
                ax.invert_yaxis()
                ax.set_xlabel(label)
                ax.xaxis.set_major_locator(MaxNLocator(4))
                ax.grid(axis="x", color="#e5e7eb", lw=0.6)
                ax.set_title(
                    DATASETS[ds] if col == 0 else "", loc="left", fontweight="bold"
                )
        fig.suptitle(f"{title} · {model} · {split}", y=0.99, fontsize=15)
        fig.subplots_adjust(
            left=0.12, right=0.99, top=0.94, bottom=0.09, hspace=0.65, wspace=0.24
        )
        fig.text(
            0.5,
            0.015,
            "Dots: mean across seeds. Whiskers: sample SD when ≥2 seeds. One seed has no uncertainty bar.",
            ha="center",
            fontsize=9,
        )
        save(fig, name)
    # Few-shot scatter is faceted by method: at most the small frozen shortlist per seed in each panel.
    fig, axes = plt.subplots(
        len(methods),
        len(datasets),
        figsize=(3.3 * len(datasets), 2.25 * len(methods) + 1),
        squeeze=False,
        sharex="col",
        sharey="col",
    )
    share_max = max(0.01, float(np.ceil(final.output_token_share.max() * 100) / 100))
    color_norm = Normalize(0, share_max)
    from matplotlib.lines import Line2D

    shot_values = sorted(map(int, final.few_shot_count.unique()))
    shapes = ["o", "s", "^", "D", "P", "X"]
    shot_marker = {n: shapes[i % len(shapes)] for i, n in enumerate(shot_values)}
    for row, m in enumerate(methods):
        for col, ds in enumerate(datasets):
            ax = axes[row, col]
            g = final[(final.dataset == ds) & (final.optimizer == m)]
            x, y = (
                g[f"{split}_cost"].to_numpy(float),
                100 * g[f"{split}_quality"].to_numpy(float),
            )
            inp, out = (
                g[f"{split}_input_tokens"].to_numpy(float),
                g[f"{split}_output_tokens"].to_numpy(float),
            )
            if ((inp + out) <= 0).any():
                raise ValueError("Invalid measured output-token share")
            shares = out / (inp + out)
            for shots, marker in shot_marker.items():
                mask = g.few_shot_count.to_numpy() == shots
                ax.scatter(
                    x[mask],
                    y[mask],
                    c=shares[mask],
                    cmap="viridis",
                    norm=color_norm,
                    marker=marker,
                    s=46,
                    edgecolors="white",
                    lw=0.5,
                )
            ax.margins(x=0.18, y=0.25)
            ax.grid(axis="y", color="#eceff3", lw=0.6)
            if row == 0:
                ax.set_title(DATASETS[ds], fontsize=12, fontweight="bold")
            if col == 0:
                ax.set_ylabel(
                    STYLE[m][0] + "\nValidation quality (%)",
                    color=STYLE[m][1],
                    fontsize=10,
                )
            if row == len(methods) - 1:
                ax.set_xlabel(COST_LABEL, fontsize=9)
            ax.xaxis.set_major_locator(MaxNLocator(4))
    fig.subplots_adjust(
        left=0.10, right=0.90, top=0.90, bottom=0.11, hspace=0.25, wspace=0.22
    )
    cax = fig.add_axes([0.925, 0.27, 0.012, 0.46])
    fig.colorbar(
        plt.cm.ScalarMappable(norm=color_norm, cmap="viridis"),
        cax=cax,
        label="Output Token Share",
    )
    fig.suptitle(
        "Frozen prompts · Few-shot count and measured token share", fontsize=15, y=0.985
    )
    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                color="#475569",
                marker=shot_marker[n],
                ls="none",
                label=f"{n} shots",
            )
            for n in shot_values
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=len(shot_values),
        frameon=False,
    )
    fig.text(
        0.5,
        0.025,
        "Shapes: few-shot count. Colour: measured output / (input + output) on a shared observed scale.\n"
        "Quality is Macro-F1 for Bios and accuracy otherwise. Cost uses configured token prices, not HPC charges.",
        ha="center",
        fontsize=9,
    )
    save(fig, "fewshot_cost_quality_small_multiples")
    if hashes != {n: sha(source / n) for n in CSV_NAMES}:
        raise RuntimeError("Source figure data changed")
    if source != output:
        for n in CSV_NAMES:
            (output / n).write_bytes((source / n).read_bytes())
    (output / "figure_provenance.json").write_text(
        json.dumps(
            dict(
                source_csv_directory=str(source),
                source_sha256=hashes,
                figures=made,
                model=model,
                split=split,
                budget=budget,
                metric_values_modified=False,
                history_interpolation="previous observation only within common observed support",
                bands="sample SD; not CI",
                all_trajectory_lines_solid=True,
                fewshot_colour_limits=[0, share_max],
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return made


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-csv")
    source.add_argument("--batch-root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="qwen-3-30b")
    parser.add_argument("--split", default="validation", choices=("validation", "test"))
    parser.add_argument("--budget", type=int, default=2000000)
    args = parser.parse_args()
    print(
        json.dumps(
            render(
                output=args.output_dir,
                source_csv=args.source_csv,
                batch_root=args.batch_root,
                model=args.model,
                split=args.split,
                budget=args.budget,
            )
        )
    )


if __name__ == "__main__":
    main()
