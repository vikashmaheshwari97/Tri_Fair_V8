"""Audit a result ZIP without extracting or unpickling its checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = {
    "quality": ("balanced_test_quality", True),
    "cost": ("balanced_test_cost", False),
    "unfairness": ("balanced_test_fairness", False),
    "HVopt": ("hv_test_optimistic_3d", True),
    "HVpes": ("hv_test_pessimistic_3d", True),
    "nR2": ("noisy_r2_3d", False),
}


def audit(archive, out):
    archive, out = Path(archive), Path(out)
    with zipfile.ZipFile(archive) as z:
        names = [
            n
            for n in z.namelist()
            if "/aggregate_validation/" in n
            and n.endswith("/front_indicators_per_seed.csv")
        ]
        if len(names) != 1:
            raise ValueError("Expected exactly one aggregate per-seed table")
        source = names[0]
        frame = pd.read_csv(io.BytesIO(z.read(source)))
        expected = {"bbq", "civil_comments", "bias_in_bios", "winobias"}
        if set(frame.dataset) != expected or len(set(frame.model)) != 1:
            raise ValueError("Require four datasets and exactly one model")
        if frame.duplicated(["dataset", "optimizer", "seed"]).any():
            raise ValueError("Duplicate method/seed result")
        if len(frame) != 4 * frame.optimizer.nunique() * frame.seed.nunique():
            raise ValueError("Incomplete comparison matrix")
        if frame[[k for k, _ in METRICS.values()]].isna().any().any():
            raise ValueError("Unavailable objective/indicator values")
        counts = {
            m: dict(method=m, outright_wins=0, tied_leads=0)
            for m in sorted(set(frame.optimizer))
        }
        contests = []
        for (ds, seed), group in frame.groupby(["dataset", "seed"]):
            for metric, (column, maximize) in METRICS.items():
                optimum = group[column].max() if maximize else group[column].min()
                leaders = group.loc[
                    np.isclose(group[column], optimum, atol=1e-10, rtol=0), "optimizer"
                ].tolist()
                for leader in leaders:
                    counts[leader][
                        "outright_wins" if len(leaders) == 1 else "tied_leads"
                    ] += 1
                contests.append(
                    dict(
                        dataset=ds,
                        seed=int(seed),
                        metric=metric,
                        best_value=float(optimum),
                        leaders=";".join(leaders),
                        tie=len(leaders) > 1,
                    )
                )
        budgets = [
            n
            for n in z.namelist()
            if "/aggregate_validation/" in n
            and n.endswith("/budget_utilization_all_runs.csv")
        ]
        budget = (
            pd.read_csv(io.BytesIO(z.read(budgets[0]))) if len(budgets) == 1 else None
        )
        report = dict(
            source_archive=archive.name,
            source_csv=source,
            archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            runs=len(frame),
            seed_count=int(frame.seed.nunique()),
            contests=len(contests),
            independent_contests=False,
            interpretation="descriptive within-dataset/seed comparisons; correlated metrics, no significance inference",
            counts=list(counts.values()),
            actual_expenditure_matched=False,
        )
        if budget is not None:
            report["mean_budget_utilization"] = float(budget.utilization.mean())
            report["all_search_phase_at_least_90_percent"] = bool(
                budget.search_phase_utilization.ge(0.9).all()
            )
    out.mkdir(parents=True, exist_ok=True)
    cols = ["dataset", "model", "optimizer", "seed", *[v[0] for v in METRICS.values()]]
    frame[cols].to_csv(out / "results_all_methods.csv", index=False)
    pd.DataFrame(contests).to_csv(out / "metric_leaders.csv", index=False)
    pd.DataFrame(counts.values()).to_csv(out / "win_counts.csv", index=False)
    if budget is not None:
        budget.to_csv(out / "budget_utilization.csv", index=False)
    (out / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.archive, args.output_dir), indent=2))
