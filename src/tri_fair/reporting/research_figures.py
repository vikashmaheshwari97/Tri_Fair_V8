"""Measured MO-CAPO-style figures for three objectives; no fabricated curves.

Final figures use frozen audit memberships when present. Trajectories require
real historical memberships and are omitted for final-only archives. One seed
has no error band. Exact input checksums and plotted indicator tables are saved.
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
from tri_fair.config.datasets import ALL_DATASETS
from tri_fair.reporting.front_report import analyse
from tri_fair.reporting.metrics import analyse_evaluated_run
from tri_fair.reporting.objectives import Bounds

COLORS = {
    "Tri-Fair-v8.6": "#007F73",
    "NSGAII-PO-Fair": "#D55E00",
    "Tri-Fair-v8.3": "#0072B2",
    "Tri-Fair-v8.5": "#8B5FBF",
    "SPEA2-PO-Fair": "#CC79A7",
    "NSGAIII-PO-Fair": "#56B4E9",
    "RandomSearch-PO-Fair": "#7B8794",
    "NSGAII-PO-Fair-Enhanced": "#B38F00",
}
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h"]
DATASETS = {
    "bbq": "BBQ",
    "civil_comments": "Civil Comments",
    "bias_in_bios": "Bias-in-Bios",
    "winobias": "WinoBias",
}


def final_rows(frame):
    parts = []
    for _, group in frame.groupby(["dataset", "model", "optimizer", "seed"], sort=True):
        if (
            "selection_policy" in group
            and group.selection_policy.eq("audited_final_shortlist").any()
        ):
            group = group[group.selection_policy.eq("audited_final_shortlist")]
        last = group.sort_values(["actual_budget_tokens", "chosen_step"]).iloc[-1]
        group = group[
            (group.actual_budget_tokens == last.actual_budget_tokens)
            & (group.chosen_step == last.chosen_step)
        ]
        if group.prompt_id.duplicated().any():
            raise ValueError("Duplicate final candidate identity")
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def attainment_curve(frame, grid, *, split, threshold):
    """Best measured quality affordable under BOTH a cost and fairness ceiling."""
    eligible = frame[frame[f"{split}_fairness"] <= threshold]
    return np.asarray(
        [
            eligible.loc[eligible[f"{split}_cost"] <= cost, f"{split}_quality"].max()
            for cost in grid
        ],
        dtype=float,
    )


def trajectory_rows(frame, *, split, budget):
    if (
        "selection_policy" not in frame
        or not frame.selection_policy.eq("current_incumbents").any()
    ):
        return pd.DataFrame()
    rows = []
    history = frame[frame.selection_policy.eq("current_incumbents")]
    for identity, g in history.groupby(
        [
            "dataset",
            "model",
            "optimizer",
            "seed",
            "chosen_step",
            "actual_budget_tokens",
        ],
        sort=True,
    ):
        dataset, model, method, seed, step, tokens = identity
        if tokens > budget:
            continue
        g = g.sort_values("prompt_id", kind="stable")
        if g.prompt_id.duplicated().any():
            raise ValueError("Duplicate prompt in one real historical front")
        if (
            not g[f"{split}_fairness_ready"].eq(True).all()
            or not g.dev_fairness_ready.eq(True).all()
        ):
            continue
        if split == "validation":
            g = g.rename(
                columns={
                    f"validation_{s}": f"test_{s}"
                    for s in ["quality", "cost", "fairness", "fairness_ready"]
                }
            )
        bound = float(
            ALL_DATASETS[dataset].fairness.fairness_kwargs[
                "normalization_cost_upper_bound"
            ]
        )
        metrics = analyse_evaluated_run(
            g,
            Bounds(np.zeros(3), np.array([1, bound, 1]), source="fixed study bounds"),
            n_preferences=4096,
            preference_seed=2026,
        )
        rows.append(
            dict(
                dataset=dataset,
                model=model,
                optimizer=method,
                seed=seed,
                chosen_step=step,
                actual_budget_tokens=tokens,
                **{k: metrics[k] for k in ["noisy_r2_3d", "hv_test_pessimistic_3d"]},
            )
        )
    return pd.DataFrame(rows)


def plot_research(frame, output, *, split, budget, title=None, formats=("png", "svg")):
    if set(frame.evaluation_split) != {split} or frame.model.nunique() != 1:
        raise ValueError("Figures require one explicit held-out split and one model")
    frame = frame[pd.to_numeric(frame.actual_budget_tokens) <= budget].copy()
    if frame.empty:
        raise ValueError("No measured memberships within budget")
    if not frame[f"{split}_fairness_ready"].eq(True).all():
        raise ValueError("Incomplete held-out fairness support")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    final = final_rows(frame)
    metrics = analyse(frame, split=split, budget=budget)
    trajectories = trajectory_rows(frame, split=split, budget=budget)
    metrics.to_csv(output / "figure_indicators.csv", index=False)
    final.drop(columns=[c for c in final if "diagnostics_json" in c]).to_csv(
        output / "figure_final_candidates.csv", index=False
    )
    if not trajectories.empty:
        trajectories.to_csv(output / "figure_trajectories.csv", index=False)
    datasets = [d for d in DATASETS if d in set(frame.dataset)]
    methods = sorted(frame.optimizer.unique(), key=lambda m: (m != "Tri-Fair-v8.6", m))
    color = {
        m: COLORS.get(m, plt.get_cmap("tab10")(i % 10)) for i, m in enumerate(methods)
    }
    marker = {m: MARKERS[i % len(MARKERS)] for i, m in enumerate(methods)}

    def label(m):
        return "Tri-Fair" if m == "Tri-Fair-v8.6" else m

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "savefig.dpi": 220,
            "axes.grid": True,
            "grid.alpha": 0.18,
        }
    )
    made = []
    scope = title or f"{frame.model.iloc[0]} | held-out {split} | measured outcomes"

    def finish(fig, name, handles, labels):
        fig.suptitle(scope, fontsize=11, y=0.995)
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.950),
            ncol=min(4, len(methods)),
            frameon=False,
            fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.025, 1, 0.91 if len(datasets) == 1 else 0.94))
        fig.text(
            0.5,
            0.007,
            "One seed: point estimates only. Multiple seeds: descriptive variability; no significance claim.",
            ha="center",
            fontsize=7,
            color="#51606E",
        )
        for ext in formats:
            fig.savefig(output / f"{name}.{ext}", bbox_inches="tight")
        plt.close(fig)
        made.append(name)

    fig, axes = plt.subplots(
        len(datasets), 3, figsize=(11.5, 2.7 * len(datasets)), squeeze=False
    )
    handles = []
    for row, dataset in enumerate(datasets):
        quality = "Macro-F1" if dataset == "bias_in_bios" else "Accuracy"
        for m in methods:
            g = final[(final.dataset == dataset) & (final.optimizer == m)]
            if g.empty:
                continue
            for col, (x, y) in enumerate(
                [
                    (f"{split}_cost", f"{split}_quality"),
                    (f"{split}_fairness", f"{split}_quality"),
                    (f"{split}_cost", f"{split}_fairness"),
                ]
            ):
                artist = axes[row, col].scatter(
                    g[x],
                    g[y],
                    c=[color[m]],
                    marker=marker[m],
                    s=36,
                    alpha=0.85,
                    edgecolors="white",
                    linewidth=0.5,
                    label=label(m),
                )
                if row == 0 and col == 0:
                    handles.append(artist)
        for col in range(3):
            axes[row, col].set_title(DATASETS[dataset])
        axes[row, 0].set(
            xlabel="Inference cost score (lower)", ylabel=f"{quality} (higher)"
        )
        axes[row, 1].set(xlabel="Unfairness (lower)", ylabel=f"{quality} (higher)")
        axes[row, 2].set(
            xlabel="Inference cost score (lower)", ylabel="Unfairness (lower)"
        )
    finish(fig, "objective_projections", handles, [label(m) for m in methods])

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), squeeze=False)
    for col, (metric, name) in enumerate(
        [
            ("hv_test_optimistic_3d", "HVopt (higher)"),
            ("hv_test_pessimistic_3d", "HVpes (higher)"),
            ("noisy_r2_3d", "Noisy R2 (lower)"),
        ]
    ):
        ax = axes[0, col]
        for i, m in enumerate(methods):
            for j, d in enumerate(datasets):
                values = metrics.loc[
                    (metrics.dataset == d) & (metrics.optimizer == m), metric
                ].to_numpy(float)
                if not len(values):
                    continue
                err = float(np.std(values, ddof=1)) if len(values) > 1 else None
                ax.errorbar(
                    j + (i - (len(methods) - 1) / 2) * 0.16,
                    values.mean(),
                    yerr=err,
                    color=color[m],
                    marker=marker[m],
                    capsize=3,
                    linestyle="none",
                    label=label(m) if j == 0 else None,
                )
        ax.set_xticks(
            range(len(datasets)),
            [DATASETS[d] for d in datasets],
            rotation=20,
            ha="right",
        )
        ax.set_ylabel(name)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    finish(fig, "front_indicators", handles, labels)

    # A transparent extension of the paper's 2-D attainment curves to our
    # third objective: fixed fairness slices, not an unlabeled 3-D projection.
    thresholds = (0.10, 0.20, 0.30)
    fig, axes = plt.subplots(
        len(datasets), 3, figsize=(11.5, 2.7 * len(datasets)), squeeze=False
    )
    for row, dataset in enumerate(datasets):
        group = final[final.dataset == dataset]
        grid = np.sort(group[f"{split}_cost"].unique())
        for col, threshold in enumerate(thresholds):
            ax = axes[row, col]
            any_curve = False
            any_feasible = False
            for m in methods:
                method = group[group.optimizer == m]
                curves = [
                    attainment_curve(g, grid, split=split, threshold=threshold)
                    for _, g in method.groupby("seed")
                ]
                if not curves:
                    continue
                curves = np.asarray(curves)
                any_feasible = any_feasible or bool(np.isfinite(curves).any())
                supported = np.isfinite(curves).all(axis=0)
                if not supported.any():
                    continue
                values = curves[:, supported]
                ax.step(
                    grid[supported],
                    np.median(values, axis=0),
                    where="post",
                    color=color[m],
                    marker=marker[m],
                    markersize=3,
                    label=label(m),
                )
                if len(values) > 1:
                    ax.fill_between(
                        grid[supported],
                        values.min(axis=0),
                        values.max(axis=0),
                        step="post",
                        alpha=0.12,
                        color=color[m],
                    )
                any_curve = True
            if not any_curve:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.text(
                    0.5,
                    0.5,
                    "No common feasible support across seeds"
                    if any_feasible
                    else "No eligible measured prompts",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=8,
                    color="#657480",
                )
            ax.set(
                title=f"{DATASETS[dataset]} | unfairness <= {threshold:.2f}",
                xlabel="Inference cost ceiling",
                ylabel="Macro-F1" if dataset == "bias_in_bios" else "Accuracy",
            )
    handles, labels = [], []
    for ax in axes.ravel():
        hh, ll = ax.get_legend_handles_labels()
        for h, lab in zip(hh, ll):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    finish(fig, "fairness_conditioned_attainment", handles, labels)

    primary = "Tri-Fair-v8.6" if "Tri-Fair-v8.6" in methods else None
    if primary is not None:
        fig, axes = plt.subplots(
            1, len(datasets), figsize=(3.2 * len(datasets), 3.8), squeeze=False
        )
        artist = None
        for i, d in enumerate(datasets):
            g = final[(final.dataset == d) & (final.optimizer == primary)]
            artist = axes[0, i].scatter(
                g[f"{split}_cost"],
                g[f"{split}_quality"],
                c=g[f"{split}_fairness"],
                vmin=0,
                vmax=1,
                cmap="viridis",
                s=55,
                edgecolors="white",
            )
            for _, r in g.iterrows():
                demos = json.loads(r.get("few_shots_json", "[]") or "[]")
                axes[0, i].annotate(
                    str(len(demos)),
                    (r[f"{split}_cost"], r[f"{split}_quality"]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                )
            axes[0, i].set(
                title=DATASETS[d],
                xlabel="Inference cost score",
                ylabel="Macro-F1" if d == "bias_in_bios" else "Accuracy",
            )
        fig.suptitle(
            scope + " | Tri-Fair composition; numbers = demonstrations", fontsize=10
        )
        fig.tight_layout(rect=(0, 0, 0.91, 0.90))
        fig.colorbar(
            artist,
            ax=axes.ravel().tolist(),
            fraction=0.025,
            pad=0.03,
            label="Unfairness (lower)",
        )
        for ext in formats:
            fig.savefig(output / f"prompt_composition.{ext}", bbox_inches="tight")
        plt.close(fig)
        made.append("prompt_composition")

    if not trajectories.empty:
        fig, axes = plt.subplots(
            len(datasets), 2, figsize=(9.5, 2.7 * len(datasets)), squeeze=False
        )
        for row, d in enumerate(datasets):
            for col, metric in enumerate(["noisy_r2_3d", "hv_test_pessimistic_3d"]):
                ax = axes[row, col]
                for m in methods:
                    g = trajectories[
                        (trajectories.dataset == d) & (trajectories.optimizer == m)
                    ]
                    seeds = []
                    for _, s in g.groupby("seed"):
                        s = s.sort_values(
                            ["actual_budget_tokens", "chosen_step"]
                        ).drop_duplicates("actual_budget_tokens", keep="last")
                        seeds.append(s)
                        ax.step(
                            s.actual_budget_tokens / 1e6,
                            s[metric],
                            where="post",
                            color=color[m],
                            alpha=1 if g.seed.nunique() == 1 else 0.25,
                            label=label(m) if len(seeds) == 1 else None,
                        )
                        if len(s) == 1:
                            ax.scatter(
                                s.actual_budget_tokens / 1e6,
                                s[metric],
                                color=color[m],
                                marker=marker[m],
                                s=24,
                            )
                    if len(seeds) > 1:
                        lo = max(s.actual_budget_tokens.min() for s in seeds)
                        hi = min(s.actual_budget_tokens.max() for s in seeds)
                        if hi >= lo:
                            grid = np.unique(
                                np.concatenate(
                                    [s.actual_budget_tokens.to_numpy() for s in seeds]
                                )
                            )
                            grid = grid[(grid >= lo) & (grid <= hi)]
                            values = np.asarray(
                                [
                                    s[metric].to_numpy()[
                                        np.searchsorted(
                                            s.actual_budget_tokens.to_numpy(),
                                            grid,
                                            side="right",
                                        )
                                        - 1
                                    ]
                                    for s in seeds
                                ]
                            )
                            mean = values.mean(axis=0)
                            std = values.std(axis=0, ddof=1)
                            ax.step(grid / 1e6, mean, where="post", color=color[m])
                            ax.fill_between(
                                grid / 1e6,
                                mean - std,
                                mean + std,
                                step="post",
                                alpha=0.13,
                                color=color[m],
                            )
                ax.set(
                    title=DATASETS[d],
                    xlabel="Actual downstream optimization tokens (millions)",
                    ylabel="Noisy R2 (lower)" if col == 0 else "HVpes (higher)",
                )
        handles, labels = axes[0, 0].get_legend_handles_labels()
        finish(fig, "optimization_trajectories", handles, labels)
    protocol = dict(
        split=split,
        model=str(frame.model.iloc[0]),
        methods=methods,
        figures=made,
        seeds=sorted(map(int, frame.seed.unique())),
        budget_ceiling=budget,
        uncertainty="sample SD across seeds only; absent for one seed",
        trajectories="real logged fronts only; no extrapolation outside common seed support",
        projections="all audited points; not three-dimensional attainment surfaces",
        nR2="4096 shared Dirichlet preferences, seed 2026; search selection and held-out scoring",
        cost="configured weighted token score, not GPU charges",
        attainment="fixed unfairness ceilings 0.10/0.20/0.30; median and min-max over seeds with common support; not significance intervals",
        v86_measured="Tri-Fair-v8.6" in methods,
    )
    (output / "figure_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    return protocol


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluations", nargs="+", required=True)
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument("--budget", type=int, default=2000000)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--title")
    args = p.parse_args()
    frame = pd.concat([pd.read_parquet(x) for x in args.evaluations], ignore_index=True)
    protocol = plot_research(
        frame, args.output_dir, split=args.split, budget=args.budget, title=args.title
    )
    protocol["sources"] = [
        dict(path=str(Path(x)), sha256=hashlib.sha256(Path(x).read_bytes()).hexdigest())
        for x in args.evaluations
    ]
    (Path(args.output_dir) / "figure_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {k: protocol[k] for k in ["figures", "methods", "seeds", "v86_measured"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
