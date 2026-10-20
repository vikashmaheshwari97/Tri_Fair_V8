"""Frozen, fail-closed orchestration; no optimizer, metric or inference bypasses."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("bbq", "civil_comments", "bias_in_bios", "winobias")
METHODS = (
    "Tri-Fair-v8.9",
    "NSGAII-PO-Fair",
    "NSGAIII-PO-Fair",
    "SPEA2-PO-Fair",
    "RandomSearch-PO-Fair",
)


def selected_seeds():
    values = tuple(int(v) for v in os.environ.get("PILOT_SEEDS", "72").split(","))
    if not values or len(values) != len(set(values)) or set(values) - {72, 73, 74}:
        raise ValueError(
            "PILOT_SEEDS must be a unique subset of 72,73,74; start with 72"
        )
    return values


SEEDS = selected_seeds()
BUDGET = 2_000_000


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def plan():
    if os.environ.get("MODEL", "qwen-3-30b") != "qwen-3-30b":
        raise ValueError(
            "This frozen matrix is Qwen-only; validate other models separately"
        )
    if selected_seeds() != SEEDS:
        raise ValueError("Seed configuration changed after module import")
    if int(os.environ.get("PILOT_BUDGET", BUDGET)) != BUDGET:
        raise ValueError(
            "This launch requires a 2000000 downstream-token cap per method"
        )
    concurrency = int(os.environ.get("MAX_CONCURRENT", 5))
    if not 1 <= concurrency <= 12:
        raise ValueError("MAX_CONCURRENT must be 1-12; recommended 5")
    return dict(
        model="qwen-3-30b",
        seeds=SEEDS,
        datasets=DATASETS,
        methods=METHODS,
        per_method_cap=BUDGET,
        task_count=4 * len(SEEDS),
        optimizer_runs=20 * len(SEEDS),
        downstream_ceiling=40_000_000 * len(SEEDS),
        max_concurrent_gpus=min(concurrency, 4 * len(SEEDS)),
        shared_budget_fill=True,
        batch_invariant=True,
        structured_labels=True,
        split="validation",
        final_test_used=False,
        pilot_settings=dict(
            n_init_prompts=4,
            max_few_shot_examples=2,
            crossovers_per_iteration=1,
            final_search_candidates=4,
            final_search_reserve=0.30,
            reclaim_audit_reserve=True,
        ),
    )


def code_hashes():
    paths = set((ROOT / "src").rglob("*.py"))
    paths.update((ROOT / "configs").rglob("*.json"))
    for name in (
        "research/run_pilot_v89.py",
        "research/plot_v89.py",
        "jobs/submit_pilot_v89.sh",
        "jobs/pilot_v89.sbatch",
        "jobs/pilot_v89_report.sbatch",
        "jobs/lib/activate_existing_venv.sh",
        "jobs/lib/common.sh",
        "jobs/lib/model_cache.sh",
    ):
        paths.add(ROOT / name)
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)}


def package_versions():
    names = (
        "torch",
        "vllm",
        "promptolution",
        "datasets",
        "pymoo",
        "numpy",
        "pandas",
        "pyarrow",
    )
    values = {}
    for name in names:
        try:
            values[name] = version(name)
        except PackageNotFoundError:
            raise ValueError(
                f"Missing required package {name}; activate the existing Rocket venv"
            ) from None
    return values


def load_config():
    base = Path(os.environ["PILOT_RESULTS_ROOT"]).expanduser().resolve()
    config = json.loads((base / "matrix_config.json").read_text(encoding="utf-8"))
    if (
        tuple(config["seeds"]) != SEEDS
        or config["per_method_cap"] != BUDGET
        or tuple(config["methods"]) != METHODS
    ):
        raise ValueError(
            "Restore the saved PILOT_SEEDS, budget and methods before using this batch"
        )
    if config["batch_root"] != str(base):
        raise ValueError("Saved batch root differs from PILOT_RESULTS_ROOT")
    if config["code_sha256"] != code_hashes():
        raise ValueError(
            "Source/job/plot files changed since submission. Restore the frozen code before retrying"
        )
    if config["package_versions"] != package_versions():
        raise ValueError(
            "Package versions changed after submission; restore the frozen venv"
        )
    for name, expected in config["existing_manifest_sha256"].items():
        if digest(Path(config["manifest_dir"]) / name) != expected:
            raise ValueError("A pre-existing split manifest changed")
    return base, config


def init():
    config = plan()
    base = Path(os.environ["PILOT_RESULTS_ROOT"]).expanduser().resolve()
    versions = package_versions()
    # Do not accidentally merge timestamps or resubmit into an existing batch.
    base.mkdir(parents=True, exist_ok=False)
    manifest = (
        Path(os.environ.get("MANIFEST_DIR", "data/splits_v8_clean"))
        .expanduser()
        .resolve()
    )
    config.update(
        batch_root=str(base),
        manifest_dir=str(manifest),
        code_sha256=code_hashes(),
        package_versions=versions,
        existing_manifest_sha256={
            p.relative_to(manifest).as_posix(): digest(p)
            for p in sorted(manifest.rglob("*.json"))
        },
        created_utc=datetime.now(timezone.utc).isoformat(),
    )
    write_json(base / "matrix_config.json", config)
    return base


def task(index):
    if index is None or not 0 <= index < 4 * len(SEEDS):
        raise ValueError(f"Array index must be 0-{4 * len(SEEDS) - 1}")
    return DATASETS[index % 4], SEEDS[index // 4]


def run(command):
    print("Running:", json.dumps([str(x) for x in command]), flush=True)
    subprocess.run([str(x) for x in command], cwd=ROOT, check=True)


def training_report(base, dataset, seed):
    matches = list(
        (base / "qwen-3-30b" / dataset / f"seed{seed}").glob("*/pilot_report.json")
    )
    if len(matches) != 1:
        raise ValueError(
            f"Expected one saved pilot report for {dataset}/seed{seed}; found {len(matches)}"
        )
    report = json.loads(matches[0].read_text(encoding="utf-8"))
    rows = report.get("runs", [])
    if (
        not report.get("all_controls_comparable")
        or len(rows) != 5
        or {r.get("method") for r in rows} != set(METHODS)
    ):
        raise ValueError("Training report does not contain five comparable methods")
    if any(
        r.get("status") != "measured" or r.get("process_exit_code") != 0 for r in rows
    ):
        raise ValueError(
            "Incomplete optimizer run: inspect its training log before validation"
        )
    return matches[0]


def train(base, config, index):
    dataset, seed = task(index)
    dest = base / "qwen-3-30b" / dataset / f"seed{seed}"
    if dest.exists():
        raise ValueError(
            f"Training output already exists: {dest}. Do not blindly rerun successful methods"
        )
    run(
        [
            sys.executable,
            "-m",
            "tri_fair.cli.pilot_v89",
            "--execute",
            "--model",
            "qwen-3-30b",
            "--dataset",
            dataset,
            "--seed",
            seed,
            "--budget",
            BUDGET,
            "--batch-invariant",
            "--n-init-prompts",
            4,
            "--max-few-shot-examples",
            2,
            "--crossovers-per-iteration",
            1,
            "--final-search-candidates",
            4,
            "--final-search-reserve",
            "0.30",
            "--reclaim-audit-reserve",
            "--methods",
            ",".join(METHODS),
            "--manifest-dir",
            config["manifest_dir"],
            "--output-root",
            base,
        ]
    )
    report = training_report(base, dataset, seed)
    write_json(
        base / f"train_ready_{index}.json",
        dict(report=str(report), report_sha256=digest(report)),
    )


def evaluate(base, config, index):
    dataset, seed = task(index)
    report = training_report(base, dataset, seed)
    out = base / "heldout_validation/qwen-3-30b" / dataset / f"seed{seed}"
    run(
        [
            sys.executable,
            "-m",
            "tri_fair.cli.evaluate",
            "--study-root",
            report.parent,
            "--study-profile",
            "v8",
            "--model",
            "qwen-3-30b",
            "--dataset",
            dataset,
            "--random-seed",
            seed,
            "--optimizers",
            ",".join(METHODS),
            "--manifest-dir",
            config["manifest_dir"],
            "--maximum-actual-tokens",
            BUDGET,
            "--max-output-tokens",
            16,
            "--structured-labels",
            "--batch-invariant",
            "--audited-final",
            "--include-history",
            "--evaluation-split",
            "validation",
            "--output-file",
            out / "eval_checkpoints_union.parquet",
        ]
    )
    run(
        [
            sys.executable,
            "-m",
            "tri_fair.reporting.front_report",
            "--evaluations",
            out / "eval_checkpoints_union.parquet",
            "--split",
            "validation",
            "--budget",
            BUDGET,
            "--output-dir",
            out / "indicators",
        ]
    )
    write_json(
        base / f"validation_ready_{index}.json",
        dict(
            dataset=dataset,
            seed=seed,
            evaluations=str(out / "eval_checkpoints_union.parquet"),
        ),
    )


def descriptive_summary(runs):
    """Keep means unchanged; sample SD is unavailable with fewer than two observations."""
    from tri_fair.reporting.metrics import aggregate_run_metrics

    summary = aggregate_run_metrics(runs)
    keys = ["dataset", "model", "optimizer", "budget_checkpoint"]
    for i, row in summary.iterrows():
        group = runs
        for key in keys:
            group = group[group[key] == row[key]]
        for column in summary:
            if column.endswith("_std") and group[column[:-4]].notna().sum() < 2:
                summary.loc[i, column] = float("nan")
    return summary


def report(base):
    paths = [
        base
        / "heldout_validation/qwen-3-30b"
        / ds
        / f"seed{s}/eval_checkpoints_union.parquet"
        for s in SEEDS
        for ds in DATASETS
    ]
    if any(not p.is_file() for p in paths):
        raise ValueError(
            "All requested seed/dataset held-out files are required; no partial aggregate"
        )
    out = base / "aggregate_validation/qwen-3-30b"
    # Reuse established completeness checks and indicator calculations unchanged.
    # Only the new MO-CAPO palette is rendered for this launch.
    import pandas as pd
    from tri_fair.reporting.multiseed_report import validate_matrix

    frame = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    runs = validate_matrix(
        frame,
        seeds=SEEDS,
        methods=METHODS,
        model="qwen-3-30b",
        split="validation",
        budget=BUDGET,
    )
    summary = descriptive_summary(runs)
    out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out / "front_indicators_per_seed.csv", index=False)
    summary.to_csv(out / "front_indicators_summary.csv", index=False)
    write_json(
        out / "aggregate_protocol.json",
        dict(
            model="qwen-3-30b",
            split="validation",
            seeds=SEEDS,
            datasets=DATASETS,
            methods=METHODS,
            n_optimization_runs=len(runs),
            budget_ceiling=BUDGET,
            uncertainty="Mean and sample SD across seeds; descriptive, not significance intervals",
            selection="Frozen search-selected audited shortlist; validation never feeds back into optimization",
            sources=[dict(path=str(p), sha256=digest(p)) for p in paths],
        ),
    )
    budget_audit(base, out)
    run(
        [
            sys.executable,
            "research/plot_v89.py",
            "--batch-root",
            base,
            "--output-dir",
            out / "paper_figures",
            "--model",
            "qwen-3-30b",
            "--split",
            "validation",
            "--budget",
            BUDGET,
        ]
    )


def budget_audit(base, out):
    import pandas as pd

    rows = []
    for seed in SEEDS:
        for dataset in DATASETS:
            path = training_report(base, dataset, seed)
            for run in json.loads(path.read_text())["runs"]:
                summary = json.loads(
                    (Path(run["run_dir"]) / "run_summary.json").read_text()
                )
                ctrl = summary["budget_controller"]
                diag = summary.get("search_diagnostics") or {}
                prices = summary["resolved_configuration"]["model"]
                inputs = (
                    summary["final_downstream_tokens"]["input_tokens"]
                    + summary["final_meta_tokens"]["input_tokens"]
                )
                outputs = (
                    summary["final_downstream_tokens"]["output_tokens"]
                    + summary["final_meta_tokens"]["output_tokens"]
                )
                rows.append(
                    dict(
                        dataset=dataset,
                        seed=seed,
                        method=run["method"],
                        downstream_tokens=summary["final_downstream_tokens"][
                            "total_tokens"
                        ],
                        meta_tokens=summary["final_meta_tokens"]["total_tokens"],
                        total_search_tokens_including_meta=inputs + outputs,
                        configured_search_cost_usd=(
                            inputs * prices["input_costs"]
                            + outputs * prices["output_costs"]
                        )
                        / 1000000,
                        utilization=summary["budget_utilization"],
                        remaining_tokens=ctrl["remaining_tokens"],
                        stopping_reason=ctrl.get("stopping_reason"),
                        runtime_stop_reason=summary.get("runtime_stop_reason"),
                        proposal_attempts=diag.get("proposal_attempts"),
                        retries_exhausted=summary.get("runtime_stop_reason")
                        == "proposal_retry_limit_no_novel_candidate",
                    )
                )
    table = pd.DataFrame(rows)
    table.to_csv(out / "budget_utilization_all_runs.csv", index=False)
    write_json(
        out / "budget_readiness.json",
        dict(
            all_at_least_90_percent=bool(table.utilization.ge(0.90).all()),
            all_within_cap=bool(table.downstream_tokens.le(BUDGET).all()),
            proposal_failures=int(table.retries_exhausted.sum()),
            minimum_utilization=float(table.utilization.min()),
            mean_utilization=float(table.utilization.mean()),
            exact_full_expenditure_guaranteed=False,
            automatic_5m_submission=False,
            next_step="Review held-out fronts, actual expenditure and operator_outcomes.jsonl; freeze before additional seeds.",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("plan", "init", "train", "evaluate", "report", "record", "check"),
    )
    parser.add_argument("--index", type=int)
    parser.add_argument("--job-ids")
    args = parser.parse_args()
    if args.stage == "plan":
        print(json.dumps(plan(), indent=2))
        return
    if args.stage == "init":
        print(init())
        return
    base, config = load_config()
    if args.stage == "check":
        for index in range(4 * len(SEEDS)):
            dataset, seed = task(index)
            training_report(base, dataset, seed)
        print(
            f"Frozen configuration and all {20 * len(SEEDS)} completed optimizer runs verified"
        )
        return
    if args.stage in {"train", "evaluate"}:
        if os.environ.get("VLLM_BATCH_INVARIANT") != "1":
            raise ValueError(
                "Matched train and validation require VLLM_BATCH_INVARIANT=1"
            )
        (train if args.stage == "train" else evaluate)(base, config, args.index)
    elif args.stage == "report":
        report(base)
    else:
        ids = (args.job_ids or "").split(",")
        if len(ids) != 3 or not all(i.isdigit() for i in ids):
            raise ValueError("Expected training, validation and report job IDs")
        write_json(
            base / "submission_jobs.json",
            dict(zip(("training", "validation", "report"), ids)),
        )


if __name__ == "__main__":
    main()
