"""MO-CAPO-style three-objective indicators for shared held-out evaluations.

Use validation while tuning; freeze the experimental protocol before test.
This summarizes existing evaluations and never calls a model or changes data.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tri_fair.config.datasets import ALL_DATASETS
from tri_fair.reporting.metrics import analyse_evaluated_run, aggregate_run_metrics
from tri_fair.reporting.objectives import Bounds


def analyse(frame, *, split, budget):
    if (
        split not in {"validation", "test"}
        or "evaluation_split" not in frame
        or set(frame.evaluation_split) != {split}
    ):
        raise ValueError("One explicitly matching held-out split is required")
    needed = {
        "dataset",
        "model",
        "optimizer",
        "seed",
        "prompt_id",
        "chosen_step",
        "actual_budget_tokens",
        "dev_quality",
        "dev_cost",
        "dev_fairness",
        "dev_fairness_ready",
        f"{split}_quality",
        f"{split}_cost",
        f"{split}_fairness",
        f"{split}_fairness_ready",
    }
    if needed - set(frame):
        raise ValueError(
            f"Missing aligned held-out columns: {sorted(needed - set(frame))}"
        )
    rows = []
    for identity, group in frame.groupby(
        ["dataset", "model", "optimizer", "seed"], sort=True
    ):
        if (
            "selection_policy" in group
            and group.selection_policy.eq("audited_final_shortlist").any()
        ):
            group = group[group.selection_policy.eq("audited_final_shortlist")]
        group = group[pd.to_numeric(group.actual_budget_tokens) <= budget]
        if group.empty:
            continue
        last = group.sort_values(["actual_budget_tokens", "chosen_step"]).iloc[-1]
        group = group[
            (group.actual_budget_tokens == last.actual_budget_tokens)
            & (group.chosen_step == last.chosen_step)
        ]
        if group.prompt_id.duplicated().any():
            # An evaluation can repeat membership at multiple budget checkpoints.
            if "budget_checkpoint" in group:
                group = group[group.budget_checkpoint == group.budget_checkpoint.max()]
            if group.prompt_id.duplicated().any():
                raise ValueError("Duplicate prompt IDs in one evaluated archive")
        if not group[f"{split}_fairness_ready"].eq(True).all():
            raise ValueError(
                "Incomplete held-out fairness support: cannot compute a comparable archive indicator"
            )
        group = group.sort_values("prompt_id", kind="stable").copy()
        if split == "validation":
            group = group.rename(
                columns={
                    f"validation_{s}": f"test_{s}"
                    for s in ["quality", "cost", "fairness", "fairness_ready"]
                }
            )
        dataset, model, optimizer, seed = identity
        bound = float(
            ALL_DATASETS[dataset].fairness.fairness_kwargs[
                "normalization_cost_upper_bound"
            ]
        )
        metrics = analyse_evaluated_run(
            group,
            Bounds(
                np.zeros(3), np.array([1, bound, 1]), source="fixed study configuration"
            ),
            n_preferences=4096,
            preference_seed=2026,
        )
        rows.append(
            dict(
                dataset=dataset,
                model=model,
                optimizer=optimizer,
                seed=int(seed),
                evaluation_split=split,
                budget_checkpoint=budget,
                chosen_step=int(last.chosen_step),
                actual_budget_tokens=int(last.actual_budget_tokens),
                run_key=f"{dataset}/{model}/{optimizer}/{seed}",
                **metrics,
            )
        )
    if not rows:
        raise ValueError("No eligible held-out experiment rows")
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluations", nargs="+", required=True)
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument("--budget", type=int, default=1000000)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    frame = pd.concat([pd.read_parquet(x) for x in args.evaluations], ignore_index=True)
    runs = analyse(frame, split=args.split, budget=args.budget)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output / "front_indicators_per_seed.csv", index=False)
    summary = aggregate_run_metrics(runs)
    for column in summary.columns:
        if column.endswith("_std"):
            summary.loc[summary.n_runs < 2, column] = np.nan
    summary.to_csv(output / "front_indicators_summary.csv", index=False)
    (output / "indicator_protocol.json").write_text(
        json.dumps(
            dict(
                split=args.split,
                selection_split="search",
                reference_point=[1.1, 1.1, 1.1],
                coordinates=["1-quality", "cost/fixed_dataset_bound", "unfairness"],
                nR2="search-selected weighted Chebyshev loss, evaluated on aligned held-out outcomes",
                preferences="4096 Dirichlet(1,1,1), seed 2026",
                std="sample standard deviation across seeds; one seed has no estimated variability",
                significance_established=False,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        summary[
            [
                "dataset",
                "model",
                "optimizer",
                "n_runs",
                "hv_test_optimistic_3d_mean",
                "hv_test_pessimistic_3d_mean",
                "noisy_r2_3d_mean",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
