"""CPU-only paper figures from measured Tri-Fair held-out results.

Visual conventions inspected in mo374z/mo-capo analysis/analysis_pipeline.py,
commit 265848bb6e0482ab7ef11028d0b8483f6560e724: plot_trajectory_single,
plot_eaf_with_bands and calculate_nR2. Original plotting implementation here.
One seed has no shaded uncertainty band. No inference or optimizer edits.
"""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tri_fair.reporting.front_report import analyse
from tri_fair.reporting.research_figures import final_rows, trajectory_rows
from figure_style_v88 import extra_figures

DATASETS = {
    "bbq": "BBQ",
    "civil_comments": "Civil Comments",
    "bias_in_bios": "Bias-in-Bios",
    "winobias": "WinoBias",
}
STYLE = {
    "Tri-Fair-v8.8": ("Tri-Fair", "#000000", "o"),
    "Tri-Fair-v8.7": ("Tri-Fair", "#006D77", "o"),
    "Tri-Fair-v8.6": ("Tri-Fair", "#006D77", "o"),
    "NSGAIII-PO-Fair": ("NSGA-III-PO-Fair", "#785EF0", "^"),
    "SPEA2-PO-Fair": ("SPEA2-PO-Fair", "#009E73", "D"),
    "RandomSearch-PO-Fair": ("Random Search", "#CC79A7", "v"),
    "NSGAII-PO-Fair": ("NSGA-II-PO-Fair", "#E69F00", "s"),
}


def stair_front(group, x, y, minimize=False):
    """Exact affordable attainment: undefined before first measured solution."""
    if group.empty:
        return np.array([]), np.array([])
    ordered = (
        group.groupby(x, sort=True)[y].min()
        if minimize
        else group.groupby(x, sort=True)[y].max()
    )
    values = ordered.cummin() if minimize else ordered.cummax()
    keep = values.ne(values.shift())
    return values.index.to_numpy()[keep], values.to_numpy()[keep]


def right_steps(x, y, grid):
    """Previous observation only; no extrapolation past observed token support."""
    x, y, grid = map(np.asarray, (x, y, grid))
    result = np.full(len(grid), np.nan)
    indices = np.searchsorted(x, grid, side="right") - 1
    valid = (indices >= 0) & (grid <= x[-1])
    result[valid] = y[indices[valid]]
    return result


def make_figures(
    batch_root,
    output_dir,
    *,
    model="qwen-3-30b",
    split="validation",
    budget=2000000,
    formats=("png", "pdf", "svg"),
):
    batch_root, output_dir = Path(batch_root).resolve(), Path(output_dir).resolve()
    paths = sorted(
        (batch_root / f"heldout_{split}" / model).glob(
            "*/seed*/eval_checkpoints_union.parquet"
        )
    )
    if not paths:
        raise ValueError(
            "Pass the batch_... directory containing heldout_validation/model/dataset/seed..."
        )
    if output_dir in [batch_root] or output_dir in {p.parent for p in paths}:
        raise ValueError("Use a separate figure output directory")
    frame = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    if set(frame.evaluation_split) != {split} or set(frame.model) != {model}:
        raise ValueError("Mixed split/model in result input")
    if (
        frame[
            [
                "dataset",
                "optimizer",
                "seed",
                "selection_policy",
                "chosen_step",
                "prompt_id",
            ]
        ]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "Duplicate candidate memberships; pass one batch, not merged reruns"
        )
    indicators = analyse(frame, split=split, budget=budget)
    final = final_rows(frame[frame.actual_budget_tokens <= budget])
    trajectory = trajectory_rows(frame, split=split, budget=budget)
    datasets = [d for d in DATASETS if d in set(final.dataset)]
    methods = sorted(
        set(final.optimizer),
        key=lambda m: (list(STYLE).index(m) if m in STYLE else 99, m),
    )
    if set(methods) - set(STYLE):
        raise ValueError("Add an explicit colour/marker for unknown methods")
    seeds = sorted(map(int, final.seed.unique()))
    output_dir.mkdir(parents=True, exist_ok=True)
    indicators.to_csv(output_dir / "measured_final_indicators.csv", index=False)
    trajectory.to_csv(output_dir / "measured_historical_trajectories.csv", index=False)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.labelweight": "bold",
            "axes.linewidth": 1.1,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    handles = [
        Line2D([], [], color=STYLE[m][1], marker=STYLE[m][2], lw=2, label=STYLE[m][0])
        for m in methods
    ]
    made, support = [], []

    def panels():
        fig, axes = plt.subplots(
            1, len(datasets), figsize=(4.05 * len(datasets), 5.25), squeeze=False
        )
        for i, (ax, ds) in enumerate(zip(axes[0], datasets)):
            ax.set_box_aspect(1)
            ax.grid(True, ls="--", lw=0.6, alpha=0.3)
            ax.tick_params(direction="out", length=4, width=1)
            ax.text(
                0.5,
                -0.28,
                f"({chr(97 + i)}) {DATASETS[ds]}",
                ha="center",
                va="top",
                transform=ax.transAxes,
                fontsize=13,
            )
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=min(4, len(methods)),
            frameon=False,
            fontsize=12,
        )
        fig.subplots_adjust(left=0.055, right=0.99, bottom=0.26, top=0.83, wspace=0.35)
        return fig, axes[0]

    def save(fig, name, footer):
        fig.text(0.5, 0.045, footer, ha="center", fontsize=9, color="#46505A")
        for ext in formats:
            fig.savefig(output_dir / f"{name}.{ext}", dpi=250, bbox_inches="tight")
        plt.close(fig)
        made.append(name)

    def frontier(name, x, y, *, ymin=False, threshold=None):
        fig, axes = panels()
        for ax, ds in zip(axes, datasets):
            g = final[final.dataset == ds]
            xmax = float(g[x].max())
            present = False
            for m in methods:
                group = g[g.optimizer == m]
                if threshold is not None:
                    group = group[group[f"{split}_fairness"] <= threshold]
                eligible_seeds = group.seed.nunique()
                support.append(
                    dict(
                        figure=name,
                        dataset=ds,
                        method=m,
                        total_seeds=len(seeds),
                        eligible_seeds=int(eligible_seeds),
                    )
                )
                label, color, marker = STYLE[m]
                if len(seeds) == 1:
                    xs, ys = stair_front(group, x, y, ymin)
                    if not len(xs):
                        continue
                    ax.step(
                        np.r_[xs, xmax],
                        np.r_[ys, ys[-1]],
                        where="post",
                        color=color,
                        lw=2,
                        zorder=3,
                    )
                    ax.plot(
                        xs, ys, ls="none", marker=marker, color=color, ms=5, zorder=4
                    )
                    present = True
                else:
                    # Multi-run cost envelopes require every seed to contribute;
                    # finite-support coverage is saved explicitly, never filled.
                    if eligible_seeds != len(seeds):
                        continue
                    grid = np.sort(g[x].unique())
                    curves = []
                    for seed in seeds:
                        sx, sy = stair_front(group[group.seed == seed], x, y, ymin)
                        idx = np.searchsorted(sx, grid, side="right") - 1
                        curve = np.full(len(grid), np.nan)
                        ok = idx >= 0
                        curve[ok] = sy[idx[ok]]
                        curves.append(curve)
                    values = np.asarray(curves)
                    ok = np.isfinite(values).all(axis=0)
                    if not ok.any():
                        continue
                    values = values[:, ok]
                    q = np.median(values, axis=0)
                    ax.step(
                        grid[ok],
                        q,
                        where="post",
                        color=color,
                        marker=marker,
                        markevery=max(1, len(q) // 6),
                        ms=4,
                        lw=2,
                    )
                    ax.fill_between(
                        grid[ok],
                        values.min(axis=0),
                        values.max(axis=0),
                        step="post",
                        alpha=0.16,
                        color=color,
                        lw=0,
                    )
                    present = True
            if not present:
                ax.text(
                    0.5,
                    0.5,
                    "No common feasible support",
                    ha="center",
                    transform=ax.transAxes,
                    fontsize=10,
                )
            qlabel = "Macro-F1 (%)" if ds == "bias_in_bios" else "Accuracy (%)"
            ax.set_xlabel(
                "Avg. Cost [$] per 1M Calls"
                if x.endswith("cost")
                else "Unfairness ceiling"
            )
            ax.set_ylabel(qlabel if y.endswith("quality_percent") else "Unfairness")
        footer = f"{model} | Held-out {split} | " + (
            "One seed: measured attainment, no uncertainty band."
            if len(seeds) == 1
            else "Median attainment; shading is across-seed best–worst envelope."
        )
        footer += (
            " Fairness projected out."
            if threshold is None and y.endswith("quality_percent")
            else (f" Unfairness ≤ {threshold:.2f}." if threshold is not None else "")
        )
        save(fig, name, footer)

    final[f"{split}_quality_percent"] = 100 * final[f"{split}_quality"]
    frontier("attainment_quality_cost", f"{split}_cost", f"{split}_quality_percent")
    frontier(
        "attainment_unfairness_cost", f"{split}_cost", f"{split}_fairness", ymin=True
    )
    for threshold in (0.10, 0.20, 0.30):
        frontier(
            f"attainment_fairness_{int(threshold * 100):03d}",
            f"{split}_cost",
            f"{split}_quality_percent",
            threshold=threshold,
        )
    for metric, title, name in [
        ("noisy_r2_3d", "nR2 indicator (↓)", "trajectory_nR2"),
        ("hv_test_pessimistic_3d", "Pessimistic hypervolume (↑)", "trajectory_HVpes"),
    ]:
        if trajectory.empty:
            continue
        fig, axes = panels()
        for ax, ds in zip(axes, datasets):
            for m in methods:
                _, color, marker = STYLE[m]
                group = trajectory[
                    (trajectory.dataset == ds) & (trajectory.optimizer == m)
                ]
                curves = []
                for seed, g in group.groupby("seed"):
                    g = g.sort_values("actual_budget_tokens").drop_duplicates(
                        "actual_budget_tokens", keep="last"
                    )
                    curves.append(
                        (g.actual_budget_tokens.to_numpy(), g[metric].to_numpy())
                    )
                if not curves:
                    continue
                if len(curves) == 1:
                    xx, yy = curves[0]
                    ax.step(xx / 1e6, yy, where="post", color=color, lw=2)
                    ax.plot(
                        xx[[0, -1]] / 1e6,
                        yy[[0, -1]],
                        ls="none",
                        marker=marker,
                        color=color,
                        ms=5,
                    )
                else:
                    grid = np.unique(np.concatenate([c[0] for c in curves]))
                    stack = np.asarray([right_steps(x, y, grid) for x, y in curves])
                    valid = np.isfinite(stack).all(axis=0)
                    if valid.any():
                        stack = stack[:, valid]
                        mean = stack.mean(axis=0)
                        sd = stack.std(axis=0, ddof=1)
                        ax.step(
                            grid[valid] / 1e6, mean, where="post", color=color, lw=2
                        )
                        ax.fill_between(
                            grid[valid] / 1e6,
                            mean - sd,
                            mean + sd,
                            step="post",
                            alpha=0.16,
                            color=color,
                            lw=0,
                        )
                audit = indicators[
                    (indicators.dataset == ds) & (indicators.optimizer == m)
                ]
                # Open endpoints are a different final selection policy, not an
                # invented continuation of the historical incumbent trajectory.
                ax.scatter(
                    audit.actual_budget_tokens / 1e6,
                    audit[metric],
                    marker=marker,
                    facecolors="white",
                    edgecolors=color,
                    s=52,
                    lw=1.6,
                    zorder=5,
                )
            ax.set(xlim=(0, budget / 1e6), xlabel="Token Budget [×10⁶]", ylabel=title)
        save(
            fig,
            name,
            f"{model} | Held-out {split} | Lines: logged incumbent fronts; open markers: frozen final shortlist. "
            + (
                "One seed; no bands."
                if len(seeds) == 1
                else "Historical mean ± sample SD on common support."
            ),
        )
    made.extend(
        extra_figures(
            final,
            trajectory,
            indicators,
            output_dir,
            datasets=datasets,
            methods=methods,
            style=STYLE,
            names=DATASETS,
            split=split,
            model=model,
            budget=budget,
            formats=formats,
        )
    )
    pd.DataFrame(support).to_csv(output_dir / "attainment_support.csv", index=False)
    provenance = dict(
        model=model,
        split=split,
        budget=budget,
        seeds=seeds,
        figures=made,
        source_repository="https://github.com/mo374z/mo-capo",
        source_commit="265848bb6e0482ab7ef11028d0b8483f6560e724",
        inspected_functions=[
            "plot_trajectory_single",
            "plot_eaf_with_bands",
            "calculate_nR2",
        ],
        notes=[
            "Original rendering code; MO-CAPO visual conventions adapted to measured three-objective results",
            "One seed has no cross-seed bands; no synthetic initial baseline",
            "Two-objective quality-cost projection ignores unfairness; see fixed fairness slices",
            "Fixed Tri-Fair bounds and 4096 seeded three-objective preferences retained; not MO-CAPO two-objective numbers",
            "No post-run selection of a winning method, threshold or seed; no model calls",
            "MO-CAPO exact method palette and viridis output-token share; symbols remapped to Tri-Fair study methods",
            "Cost axis is the configured model price estimate per million calls, not observed HPC charges",
        ],
        sources=[
            dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            for p in paths
        ],
    )
    (output_dir / "plot_protocol.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    return provenance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", default="qwen-3-30b")
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument("--budget", type=int, default=2000000)
    p.add_argument("--formats", default="png,pdf,svg")
    args = p.parse_args()
    formats = tuple(args.formats.split(","))
    if not set(formats) <= {"png", "pdf", "svg"}:
        p.error("Formats must be png,pdf,svg")
    result = make_figures(
        args.batch_root,
        args.output_dir,
        model=args.model,
        split=args.split,
        budget=args.budget,
        formats=formats,
    )
    print(
        "Saved "
        + str(len(result["figures"]))
        + " measured figure sets. No GPU inference performed."
    )


if __name__ == "__main__":
    main()
