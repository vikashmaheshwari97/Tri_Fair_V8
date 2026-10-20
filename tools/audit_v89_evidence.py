"""Read-only audit of extracted v8.9/v8.8 pilot exports; never rewrites results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def audit(root):
    root = Path(root)
    paths = [
        p
        for p in root.rglob("front_indicators_per_seed.csv")
        if "aggregate_validation" in p.parts
    ]
    if len(paths) != 1:
        raise ValueError(
            "Select one extracted batch with exactly one aggregate_validation table"
        )
    table = pd.read_csv(paths[0])
    metrics = {
        "balanced_test_quality": 1,
        "balanced_test_cost": -1,
        "balanced_test_fairness": -1,
        "hv_test_optimistic_3d": 1,
        "hv_test_pessimistic_3d": 1,
        "noisy_r2_3d": -1,
    }
    wins = {m: dict(outright=0, including_ties=0) for m in table.optimizer.unique()}
    for _, group in table.groupby(["dataset", "seed"]):
        for metric, sign in metrics.items():
            values = group[metric].to_numpy(float) * sign
            leaders = group[np.isclose(values, values.max(), atol=1e-10, rtol=0)]
            for method in leaders.optimizer:
                wins[method]["including_ties"] += 1
                wins[method]["outright"] += int(len(leaders) == 1)
    summaries, sources, edits = [], [], []
    for p in sorted(root.rglob("run_summary.json")):
        row = json.loads(p.read_text())
        sources.append(
            dict(
                path=p.relative_to(root).as_posix(),
                sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            )
        )
        summaries.append(
            dict(
                dataset=row["dataset"],
                method=row["optimizer"],
                seed=row["seed"],
                status=row["status"],
                utilization=row["budget_utilization"],
                budget=row["requested_budget"],
                actual_tokens=row["final_downstream_tokens"]["total_tokens"],
            )
        )
    for p in sorted(root.rglob("operator_outcomes.jsonl")):
        edits.extend(
            json.loads(line) for line in p.read_text().splitlines() if line.strip()
        )
    strategies = {}
    for mode in sorted({e["mode"] for e in edits}):
        rows = [e for e in edits if e["mode"] == mode]
        strategies[mode] = dict(
            n=len(rows),
            positive_rewards=sum(e["strategy_reward"] > 0 for e in rows),
            mean_quality_change=float(np.mean([e["quality_change"] for e in rows])),
            mean_unfairness_change=float(
                np.mean([e["unfairness_change"] for e in rows])
            ),
        )
    fields = [
        "dataset",
        "optimizer",
        "seed",
        "quality_first_dev_quality",
        "quality_first_test_quality",
        "balanced_dev_quality",
        "balanced_test_quality",
        "balanced_test_cost",
        "balanced_test_fairness",
        "hv_test_optimistic_3d",
        "hv_test_pessimistic_3d",
        "noisy_r2_3d",
    ]
    return dict(
        source_root=str(root),
        source_table_sha256=hashlib.sha256(paths[0].read_bytes()).hexdigest(),
        runs=summaries,
        wins=wins,
        operator_outcomes=strategies,
        selected_metrics=table[fields].to_dict("records"),
        sources=sources,
        interpretation="test-named metric columns contain the evaluation_split recorded by the table; these pilots use validation",
        performance_guarantee=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v89", required=True)
    parser.add_argument("--v88")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = {"v89": audit(args.v89)}
    if args.v88:
        result["v88"] = audit(args.v88)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                name: {
                    "runs": len(value["runs"]),
                    "wins": value["wins"],
                    "operator_outcomes": value["operator_outcomes"],
                }
                for name, value in result.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
