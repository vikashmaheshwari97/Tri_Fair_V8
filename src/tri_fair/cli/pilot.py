"""Isolated, explicitly exploratory token-budget screening; dry-run by default."""

from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
from uuid import uuid4

from tri_fair.cli.common import atomic_write_json
from tri_fair.config.registry import METHODS

VARIANT = "Tri-Fair-v8.6"
DEFAULT_METHODS = (
    VARIANT,
    "NSGAII-PO-Fair",
    "SPEA2-PO-Fair",
    "NSGAIII-PO-Fair",
    "RandomSearch-PO-Fair",
    "NSGAII-PO-Fair-Enhanced",
)


def measured_accuracy(row, quality_metric):
    if quality_metric == "accuracy":
        return float(row["quality"])
    # For single-label Bios, support-weighted per-profession recall is accuracy.
    diagnostics = json.loads(row.get("fairness_diagnostics_json", "{}"))
    support = json.loads(row.get("group_support_json", "{}"))
    recall = diagnostics.get("profession_recall", {})
    counts = {
        label: sum(
            int(support.get(f"{label}/{gender}", 0)) for gender in ("male", "female")
        )
        for label in recall
    }
    total = sum(counts.values())
    return (
        sum(float(recall[label]) * count for label, count in counts.items()) / total
        if total
        else None
    )


def train_command(args, method, root):
    """All methods receive the same pilot settings and unmodified metric gates."""
    command = [
        sys.executable,
        "-m",
        "tri_fair.cli.train",
        "--experiment-name",
        str(getattr(args, "experiment_name", "tri_fair_v86_portfolio_pilot")),
        "--model",
        args.model,
        "--dataset",
        args.dataset,
        "--optimizer",
        method,
        "--random-seed",
        str(args.seed),
        "--n-init-prompts",
        str(getattr(args, "n_init_prompts", 2)),
        "--max-steps",
        "200",
        "--max-idle-steps",
        "3",
        "--meta-max-output-tokens",
        "512",
        "--crossovers-per-iteration",
        str(getattr(args, "crossovers_per_iteration", 1)),
        "--max-few-shot-examples",
        str(getattr(args, "max_few_shot_examples", 1)),
        "--final-search-candidates",
        str(getattr(args, "final_search_candidates", 2)),
        "--final-search-reserve",
        str(getattr(args, "final_search_reserve", 0.35)),
        "--budget-per-run",
        str(args.budget),
        "--budget-checkpoints",
        str(args.budget),
        "--manifest-dir",
        str(Path(args.manifest_dir).expanduser().resolve()),
        "--output-dir",
        str(root / method),
        "--run-id",
        "run",
        "--allow-under-budget-exit",
    ]
    if not getattr(args, "unconstrained_labels", False):
        command.append("--structured-labels")
    return command


def read_result(run_dir, method):
    """Do not compare partial-fidelity scores or replace a failed run with a zero."""
    import pandas as pd

    result = {"method": method, "run_dir": str(run_dir), "comparable": False}
    path = run_dir / "run_summary.json"
    if not path.is_file():
        return dict(result, status="failed", reason="missing_run_summary")
    summary = json.loads(path.read_text(encoding="utf-8"))
    result["training_status"] = summary.get("status")
    events = run_dir / "budget_events.jsonl"
    if events.is_file():
        observed = 0
        for line in events.read_text(encoding="utf-8").splitlines():
            try:
                observed = max(observed, int(json.loads(line).get("after_tokens", 0)))
            except (ValueError, TypeError):
                continue  # A killed process may leave a partial final line.
        result["observed_downstream_tokens"] = observed
    if summary.get("status") != "complete":
        insufficient = summary.get("error_type") == "BudgetExhausted"
        readiness = str(summary.get("error", "")).startswith(
            "No initial prompt reached statistical fairness readiness"
        )
        return dict(
            result,
            status="insufficient_budget"
            if insufficient
            else "insufficient_evidence"
            if readiness
            else "failed",
            reason=summary.get("error", "training did not complete"),
        )
    for name in (
        "requested_budget",
        "final_downstream_tokens",
        "final_meta_tokens",
        "budget_utilization",
        "budget_overshoot_tokens",
        "code_revision",
        "source_sha256",
        "fairness_metric",
        "method_version",
        "search_diagnostics",
        "runtime_stop_reason",
    ):
        result[name] = summary.get(name)
    actual = summary["final_downstream_tokens"]["total_tokens"]
    if actual > summary["requested_budget"]:
        return dict(result, status="failed", reason="budget_exceeded")
    resolved = summary["resolved_configuration"]
    partition = resolved["search_validation_partition"]
    n_blocks = len(partition["search_original_blocks"])
    if n_blocks < 1:
        return dict(result, status="failed", reason="empty_search_partition")
    result["search_blocks_required"] = n_blocks
    result["search_original_blocks"] = partition["search_original_blocks"]
    result["dataset_configuration"] = resolved["dataset"]
    result["downstream_decoding"] = resolved.get("downstream_decoding", {})
    audit = summary.get("final_search_audit")
    if audit:
        result["final_search_audit"] = audit
        if audit.get("status") != "complete":
            return dict(
                result, status="inconclusive", reason="incomplete_final_search_audit"
            )
        path = run_dir / "final_search_results.parquet"
        result["metric_source"] = "frozen_shortlist_full_search_audit"
    else:
        path = run_dir / "step_results.parquet"
        result["metric_source"] = "final_logged_search_incumbents"
    if not path.is_file():
        return dict(
            result,
            status="inconclusive",
            reason="missing_final_search_results" if audit else "missing_step_results",
        )
    frame = pd.read_parquet(path)
    if frame.empty:
        return dict(result, status="inconclusive", reason="empty_step_results")
    result["final_step"] = int(frame["step"].max())
    if result["final_step"] < 1:
        return dict(
            result,
            status="inconclusive",
            reason="initialization_only_no_optimizer_step",
        )
    frame = frame[frame["step"] == frame["step"].max()].copy()
    full = frame["evaluated_on_blocks"].map(
        lambda value: set(json.loads(value)) == set(range(n_blocks))
    )
    eligible = frame[
        full & frame["is_incumbent"].eq(True) & frame["fairness_ready"].eq(True)
    ].copy()
    eligible = eligible[
        eligible[["quality", "cost", "fairness"]].apply(
            lambda row: (
                all(math.isfinite(float(v)) for v in row)
                and 0 <= float(row["quality"]) <= 1
                and 0 <= float(row["fairness"]) <= 1
                and float(row["cost"]) >= 0
            ),
            axis=1,
        )
    ]
    if eligible.empty:
        return dict(
            result,
            status="inconclusive",
            reason="no_fully_evaluated_fairness_ready_incumbent",
        )
    fairness_config = resolved["dataset"]["fairness"]
    bound = float(fairness_config["fairness_kwargs"]["normalization_cost_upper_bound"])
    distance = (
        (1 - eligible["quality"]) ** 2
        + (eligible["cost"] / bound) ** 2
        + eligible["fairness"] ** 2
    )
    best = eligible.loc[distance.idxmin()]
    result.update(
        status="measured",
        comparable=True,
        quality_metric=fairness_config["quality_metric"],
        accuracy=measured_accuracy(best, fairness_config["quality_metric"]),
        quality=float(best["quality"]),
        cost=float(best["cost"]),
        unfairness=float(best["fairness"]),
        instruction=str(best["instruction"]),
        few_shots_json=str(best["few_shots_json"]),
        fully_evaluated_incumbents=len(eligible),
        final_incumbent_candidates=eligible[
            ["quality", "cost", "fairness", "instruction", "few_shots_json"]
        ].to_dict("records"),
    )
    from tri_fair.reporting.pilot_indicators import shortlist_indicators

    result["front_indicators"] = shortlist_indicators(result)
    return result


def compare_results(records, variant_name=VARIANT):
    """Point estimates on search data, never a claim of held-out superiority."""
    variant = next((row for row in records if row["method"] == variant_name), None)
    comparisons = []
    for control in records:
        if control["method"] == variant_name:
            continue
        comparable = bool(variant and variant["comparable"] and control["comparable"])
        reason = None
        if comparable:
            for field in (
                "search_original_blocks",
                "dataset_configuration",
                "requested_budget",
                "source_sha256",
                "metric_source",
                "downstream_decoding",
            ):
                if variant.get(field) != control.get(field):
                    comparable, reason = False, f"mismatched_{field}"
                    break
        comparison = {"control": control["method"], "comparable": comparable}
        if not comparable:
            comparison["reason"] = reason or "missing_comparable_evidence"
            comparison["all_three_strictly_better"] = None
        else:
            comparison["all_three_strictly_better"] = bool(
                variant["quality"] > control["quality"]
                and variant["cost"] < control["cost"]
                and variant["unfairness"] < control["unfairness"]
            )
            comparison["differences_variant_minus_control"] = {
                key: variant[key] - control[key]
                for key in ("quality", "cost", "unfairness")
            }
        comparisons.append(comparison)
    return comparisons


def main(
    argv=None,
    *,
    variant_name=VARIANT,
    namespace_prefix="tri_fair_v86_pilot",
    default_methods=DEFAULT_METHODS,
    defaults=None,
):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["bbq", "civil_comments", "bias_in_bios", "winobias"],
    )
    parser.add_argument(
        "--model",
        default="qwen-3-30b",
        choices=["qwen-3-30b", "mistral-3-24b", "gpt-oss-120b"],
    )
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument("--budget", type=int, default=2000000)
    parser.add_argument("--n-init-prompts", type=int, default=4)
    parser.add_argument("--max-few-shot-examples", type=int, default=2)
    parser.add_argument("--crossovers-per-iteration", type=int, default=1)
    parser.add_argument("--final-search-candidates", type=int, default=4)
    parser.add_argument("--final-search-reserve", type=float, default=0.30)
    parser.add_argument(
        "--unconstrained-labels",
        action="store_true",
        help="Disable the shared choice grammar for every method, never selectively.",
    )
    parser.add_argument("--methods", default=",".join(default_methods))
    parser.add_argument("--manifest-dir", default="data/splits_v8_clean")
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--execute", action="store_true", help="Without this flag, print the plan only."
    )
    parser.set_defaults(**(defaults or {}))
    args = parser.parse_args(argv)
    methods = list(dict.fromkeys(args.methods.split(",")))
    if (
        args.budget <= 0
        or args.n_init_prompts < 2
        or args.max_few_shot_examples < 0
        or args.crossovers_per_iteration < 1
        or args.final_search_candidates < 1
        or not 0 < args.final_search_reserve < 1
        or variant_name not in methods
        or len(methods) < 2
        or set(methods) - set(METHODS)
    ):
        parser.error(
            "Use positive budget/challenger/audit counts, at least two initial prompts, a reserve in (0,1), and at least two registered methods including the experimental variant."
        )
    stamp = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    )
    root = (
        Path(args.output_root or f"results/{namespace_prefix}_{args.budget}")
        .expanduser()
        .resolve()
        / args.model
        / args.dataset
        / f"seed{args.seed}"
        / stamp
    )
    commands = [(method, train_command(args, method, root)) for method in methods]
    if not args.execute:
        for _, command in commands:
            print(shlex.join(command))
        print("DRY RUN: no model loaded and no results modified.")
        return 0
    root.mkdir(parents=True, exist_ok=False)
    policy = {
        "benchmark_result": False,
        "split": "search",
        "seed": args.seed,
        "dataset": args.dataset,
        "model": args.model,
        "budget_per_method": args.budget,
        "methods": methods,
        "pilot_configuration": {
            "structured_labels": not args.unconstrained_labels,
            "max_idle_steps": 3,
            "max_search_steps": 200,
            "meta_max_output_tokens": 512,
            "n_init_prompts": args.n_init_prompts,
            "max_few_shot_examples": args.max_few_shot_examples,
            "crossovers_per_iteration": args.crossovers_per_iteration,
            "final_search_candidates": args.final_search_candidates,
            "final_search_reserve": args.final_search_reserve,
            "search_budget_tokens": int(args.budget * (1 - args.final_search_reserve)),
            "audit_is_included_in_total_cap": True,
        },
        "units": {
            "quality": "fraction",
            "accuracy": "fraction",
            "unfairness": "loss, smaller is better",
            "cost": "configured input/output token cost score, not HPC charges",
        },
        "selection": "freeze the same-sized shortlist by search distance; complete full search coverage; select by squared normalized distance to (quality=1,cost=0,unfairness=0)",
        "limitations": [
            "Single-seed screening, not a significance test.",
            "The same explicit pilot settings and reserve policy apply to every method; this differs from the final 5M study.",
            "Original fairness thresholds; insufficient evidence remains inconclusive.",
            "Under-budget exits permitted for this pilot only; utilization is reported.",
            "Downstream tokens capped per method; meta-generation tokens reported separately.",
            "Freeze configuration and validate on held-out splits before final conclusions.",
        ],
    }
    atomic_write_json(root / "pilot_policy.json", policy)
    records = []
    for method, command in commands:
        with (root / f"{method}.log").open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                command, stdout=stream, stderr=subprocess.STDOUT, check=False
            )
        try:
            record = read_result(root / method / "run", method)
        except Exception as error:
            record = dict(
                method=method,
                comparable=False,
                status="failed",
                reason=f"report_error: {error}",
            )
        record["process_exit_code"] = completed.returncode
        if completed.returncode and record["comparable"]:
            record.update(
                comparable=False, status="failed", reason="nonzero_process_exit"
            )
        records.append(record)
        comparisons = compare_results(records, variant_name=variant_name)
        failed = any(row["status"] == "failed" for row in records)
        complete = len(records) == len(methods)
        comparable = (
            complete
            and all(row["comparable"] for row in records)
            and all(row["comparable"] for row in comparisons)
        )
        report = dict(
            policy,
            status="running"
            if not complete
            else "failed"
            if failed
            else "exploratory_complete"
            if comparable
            else "inconclusive",
            runs=records,
            comparisons=comparisons,
            all_controls_comparable=comparable,
            all_three_better_than_every_control=(
                all(row["all_three_strictly_better"] for row in comparisons)
                if comparable
                else None
            ),
        )
        atomic_write_json(root / "pilot_report.json", report)
        print(f"{method}: {record['status']}", flush=True)
        print(
            f"  downstream={record.get('observed_downstream_tokens', 'unknown')}; reason={record.get('reason', 'fully evaluated')}",
            flush=True,
        )
    print(f"Report: {root / 'pilot_report.json'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
