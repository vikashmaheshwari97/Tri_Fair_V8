"""Require a complete seed/dataset/method matrix before aggregating held-out data."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import pandas as pd

from tri_fair.reporting.front_report import analyse
from tri_fair.reporting.metrics import aggregate_run_metrics
from tri_fair.reporting.research_figures import plot_research

DATASETS = ("bbq", "civil_comments", "bias_in_bios", "winobias")


def validate_matrix(frame, *, seeds, methods, model, split, budget):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Expected unique nonempty seeds")
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("Expected unique nonempty methods")
    if set(frame.model) != {model}:
        raise ValueError("A single explicitly matching model is required")
    if "selection_policy" not in frame:
        raise ValueError("Missing audited-final membership")
    audited = frame[frame.selection_policy.eq("audited_final_shortlist")]
    expected = set(product(DATASETS, methods, seeds))
    actual = set(
        audited[["dataset", "optimizer", "seed"]].itertuples(index=False, name=None)
    )
    if actual != expected:
        raise ValueError(
            f"Incomplete or unexpected audited matrix: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    runs = analyse(frame, split=split, budget=budget)
    counted = set(
        runs[["dataset", "optimizer", "seed"]].itertuples(index=False, name=None)
    )
    if counted != expected or len(runs) != len(expected):
        raise ValueError(
            "Not every expected run has an eligible held-out archive within budget"
        )
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", nargs="+", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--methods", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--budget", type=int, default=2000000)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    paths = [Path(x).resolve() for x in args.evaluations]
    if len(set(paths)) != len(paths):
        raise ValueError("Repeated evaluation input paths")
    frame = pd.concat([pd.read_parquet(x) for x in paths], ignore_index=True)
    seeds = [int(x) for x in args.seeds.split(",")]
    methods = args.methods.split(",")
    runs = validate_matrix(
        frame,
        seeds=seeds,
        methods=methods,
        model=args.model,
        split=args.split,
        budget=args.budget,
    )
    summary = aggregate_run_metrics(runs)
    for col in summary:
        if col.endswith("_std"):
            summary.loc[summary.n_runs < 2, col] = float("nan")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output / "front_indicators_per_seed.csv", index=False)
    summary.to_csv(output / "front_indicators_summary.csv", index=False)
    plot_research(
        frame,
        output / "figures",
        split=args.split,
        budget=args.budget,
        title=f"Qwen | seeds {', '.join(map(str, seeds))} | held-out {args.split}"
        if args.model == "qwen-3-30b"
        else None,
    )
    protocol = dict(
        model=args.model,
        split=args.split,
        seeds=seeds,
        datasets=DATASETS,
        methods=methods,
        n_optimization_runs=len(runs),
        budget_ceiling=args.budget,
        uncertainty="Mean and sample SD across seed-specific runs; descriptive, not a significance test",
        selection="Frozen search-selected audited shortlist; validation never feeds back into optimization",
        sources=[
            dict(path=str(x), sha256=hashlib.sha256(x.read_bytes()).hexdigest())
            for x in paths
        ],
    )
    (output / "aggregate_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
