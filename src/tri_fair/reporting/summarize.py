"""Summarize v8 validation results and freeze representatives before final test.

Final-test reporting requires the selection JSON produced from validation.
No metric in the test file chooses the representative prompt. All normalization
bounds are fixed before either split is inspected; out-of-bound values are kept.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tri_fair.config.datasets import ALL_DATASETS
from tri_fair.config.study import apply_study_profile
from tri_fair.search.selection import hypervolume

KEYS = ["dataset", "model", "optimizer", "seed"]


def summarize(
    frame,
    *,
    split="validation",
    budget=5000000,
    selections=None,
    cost_upper_bounds=None,
):
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    if "evaluation_split" not in frame or set(frame.evaluation_split) != {split}:
        raise ValueError("input split does not match requested analysis split")
    if split == "test" and selections is None:
        raise ValueError("test reporting requires frozen validation selections")
    frame = frame[pd.to_numeric(frame.actual_budget_tokens) <= budget].copy()
    rows, locked = ([], [])
    for keys, group in frame.groupby(KEYS, sort=True):
        identity = dict(zip(KEYS, keys))
        step = group.sort_values(
            ["actual_budget_tokens", "chosen_step"], kind="stable"
        ).iloc[-1]
        group = group[
            (group.chosen_step == step.chosen_step)
            & (group.actual_budget_tokens == step.actual_budget_tokens)
        ]
        group = (
            group.drop_duplicates("prompt_id")
            .sort_values("prompt_id", kind="stable")
            .reset_index(drop=True)
        )
        columns = [f"{split}_{s}" for s in ["quality", "cost", "fairness"]]
        values = group[columns].to_numpy(dtype=float)
        ready = group[f"{split}_fairness_ready"].eq(True).to_numpy()
        finite = np.isfinite(values).all(axis=1)
        profile = apply_study_profile(ALL_DATASETS[identity["dataset"]])
        bound = (cost_upper_bounds or {}).get(
            identity["dataset"],
            profile.fairness.fairness_kwargs["normalization_cost_upper_bound"],
        )
        if bound <= 0:
            raise ValueError("cost bounds must be positive")
        losses = np.column_stack([1 - values[:, 0], values[:, 1] / bound, values[:, 2]])
        record = dict(
            identity,
            chosen_step=int(step.chosen_step),
            actual_budget_tokens=int(step.actual_budget_tokens),
            requested_budget=budget,
            budget_utilization=float(step.actual_budget_tokens) / budget,
            n_archive=len(group),
            n_fairness_ready=int((ready & finite).sum()),
            n_outside_fixed_bounds=int(
                np.sum(finite & ((losses < 0) | (losses > 1)).any(axis=1))
            ),
            archive_hypervolume=hypervolume(losses[finite & ready]),
            normalization_cost_upper_bound=float(bound),
        )
        if split == "validation":
            candidates = np.flatnonzero(finite & ready)
            if len(candidates):
                selected = int(
                    candidates[np.argmin(np.linalg.norm(losses[candidates], axis=1))]
                )
                chosen = group.iloc[selected]
                lock = dict(
                    identity,
                    chosen_step=int(step.chosen_step),
                    actual_budget_tokens=int(step.actual_budget_tokens),
                    prompt_id=str(chosen.prompt_id),
                    policy="minimum_fixed_normalized_distance_on_validation",
                )
                locked.append(lock)
            else:
                chosen = None
        else:
            matches = [
                s for s in selections if all((s[k] == identity[k] for k in KEYS))
            ]
            if len(matches) != 1:
                raise ValueError(f"need one validation selection for {identity}")
            lock = matches[0]
            if int(lock["chosen_step"]) != int(step.chosen_step) or int(
                lock["actual_budget_tokens"]
            ) != int(step.actual_budget_tokens):
                raise ValueError(
                    "test checkpoint differs from frozen validation checkpoint"
                )
            matches = group[group.prompt_id.astype(str) == str(lock["prompt_id"])]
            if len(matches) != 1:
                raise ValueError("frozen validation prompt missing from test archive")
            chosen = matches.iloc[0]
        record["representative_status"] = (
            "no_ready_candidate" if chosen is None else "selected_on_validation"
        )
        if chosen is not None:
            record.update(
                prompt_id=str(chosen.prompt_id),
                selected_quality=float(chosen[f"{split}_quality"]),
                selected_cost=float(chosen[f"{split}_cost"]),
                selected_fairness=float(chosen[f"{split}_fairness"]),
                selected_fairness_ready=bool(chosen[f"{split}_fairness_ready"]),
            )
        rows.append(record)
    if not rows:
        raise ValueError("no checkpoints at or below requested budget")
    per_seed = pd.DataFrame(rows)
    metrics = [
        "actual_budget_tokens",
        "budget_utilization",
        "archive_hypervolume",
        "n_fairness_ready",
        "n_outside_fixed_bounds",
        "selected_quality",
        "selected_cost",
        "selected_fairness",
    ]
    metrics = [m for m in metrics if m in per_seed]
    aggregated = per_seed.groupby(KEYS[:3], sort=True)[metrics].agg(
        ["mean", "std", "count"]
    )
    aggregated.columns = ["_".join(c) for c in aggregated.columns]
    return (per_seed, aggregated.reset_index(), locked)


def plot_summary(summary, path):
    """Export vector PDF/SVG or raster PNG from measured results only."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("archive_hypervolume", "Archive HV ↑"),
        ("selected_quality", "Selected quality ↑"),
        ("selected_fairness", "Selected unfairness ↓"),
    ]
    for (dataset, model), group in summary.groupby(["dataset", "model"], sort=True):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
        positions = np.arange(len(group))
        for ax, (metric, title) in zip(axes, panels):
            if f"{metric}_mean" not in group:
                ax.set_visible(False)
                continue
            ax.bar(
                positions,
                group[f"{metric}_mean"],
                yerr=group[f"{metric}_std"].fillna(0),
                color=[
                    "#0072B2" if m == "Tri-Fair-v8" else "#56B4E9"
                    for m in group.optimizer
                ],
                capsize=3,
            )
            ax.set_xticks(
                positions, group.optimizer, rotation=45, ha="right", fontsize=7
            )
            ax.set_title(title, fontsize=10)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
        fig.suptitle(
            f"{dataset} · {model} (mean ± sample SD across seeds)", fontsize=11
        )
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output.with_name(f"{output.stem}_{dataset}_{model}{output.suffix}"), dpi=450
        )
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs", nargs="+", required=True, help="Explicit evaluation parquet paths."
    )
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--budget", type=int, default=5000000)
    parser.add_argument(
        "--selection-file",
        help="Required for test; JSON emitted by validation analysis.",
    )
    parser.add_argument(
        "--cost-upper-bounds", help="Optional preregistered dataset-to-bound JSON."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--plot", choices=["pdf", "svg", "png"])
    args = parser.parse_args()
    if args.budget <= 0:
        parser.error("budget must be positive")
    frozen = (
        json.loads(Path(args.selection_file).read_text(encoding="utf-8"))
        if args.selection_file
        else None
    )
    if frozen and frozen["budget"] != args.budget:
        parser.error("budget differs from frozen validation analysis")
    selections = frozen["selections"] if frozen else None
    bounds = (
        json.loads(Path(args.cost_upper_bounds).read_text(encoding="utf-8"))
        if args.cost_upper_bounds
        else None
    )
    if frozen:
        if bounds is not None and bounds != frozen.get("cost_upper_bounds"):
            parser.error("normalization differs from frozen validation analysis")
        bounds = frozen.get("cost_upper_bounds")
    data = pd.concat([pd.read_parquet(p) for p in args.inputs], ignore_index=True)
    per_seed, summary, locked = summarize(
        data,
        split=args.split,
        budget=args.budget,
        selections=selections,
        cost_upper_bounds=bounds,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(output / f"{args.split}_per_seed.csv", index=False)
    summary.to_csv(output / f"{args.split}_summary.csv", index=False)
    if args.split == "validation":
        (output / "validation_selections.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "budget": args.budget,
                    "cost_upper_bounds": bounds,
                    "policy": "minimum fixed-normalized Euclidean distance to ideal",
                    "selections": locked,
                },
                indent=2,
                default=lambda x: int(x) if isinstance(x, np.integer) else str(x),
            )
            + "\n",
            encoding="utf-8",
        )
    if args.plot:
        plot_summary(summary, output / f"{args.split}_comparison.{args.plot}")
    print(output)


if __name__ == "__main__":
    main()
