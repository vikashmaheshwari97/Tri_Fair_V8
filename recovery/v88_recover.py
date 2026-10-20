"""Read-only diagnosis and failure-only recovery for the Qwen/seed72 v8.8 screen.

No optimizer, metric, backend setting or existing result is changed. Kept out of
src/ and scripts/ because those paths are part of the measured source digest.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

DATASETS = ("bbq", "civil_comments", "bias_in_bios", "winobias")
METHODS = (
    "Tri-Fair-v8.8",
    "NSGAII-PO-Fair",
    "NSGAIII-PO-Fair",
    "SPEA2-PO-Fair",
    "RandomSearch-PO-Fair",
)
TARGET = ("bias_in_bios", "Tri-Fair-v8.8")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    tmp.replace(path)


def source_digest(repo):
    """Match effective_run_metadata exactly; do not exempt changed source files."""
    repo = Path(repo)
    digest = hashlib.sha256()
    for path in sorted(
        [*(repo / "src").rglob("*.py"), *(repo / "scripts").rglob("*.py")]
    ):
        digest.update(str(path.relative_to(repo)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def probe_details(probe):
    invalid = probe.get("invalid_label_examples", [])
    mismatch = probe.get("mismatches", [])
    reason = (
        "invalid_labels"
        if invalid
        else "batch_order_disagreement"
        if mismatch
        else "unclassified_probe_failure"
    )
    return dict(
        status=probe.get("status"),
        reason=reason,
        invalid_label_examples=invalid,
        mismatches=mismatch,
        prediction_count=probe.get("prediction_count"),
        actual_downstream_tokens=probe.get("actual_downstream_tokens"),
        hostname=probe.get("hostname"),
        slurm_job_id=probe.get("slurm_job_id"),
    )


def inspect_batch(batch):
    batch = Path(batch).expanduser().resolve()
    reports, runs = {}, {}
    for dataset in DATASETS:
        paths = list(
            (batch / "qwen-3-30b" / dataset / "seed72").glob("*/pilot_report.json")
        )
        if len(paths) != 1:
            raise ValueError(
                f"Expected exactly one {dataset}/seed72 pilot report, found {len(paths)}"
            )
        path = paths[0]
        report = read(path)
        if (report.get("dataset"), report.get("model"), report.get("seed")) != (
            dataset,
            "qwen-3-30b",
            72,
        ):
            raise ValueError(f"Wrong dataset/model/seed in {path}")
        records = report.get("runs", [])
        if len(records) != 5 or {r.get("method") for r in records} != set(METHODS):
            raise ValueError(f"Expected the complete five-method record list in {path}")
        reports[dataset] = (path, report)
        for record in records:
            directory = (path.parent / record["method"] / "run").resolve()
            if not directory.is_relative_to(batch) or not directory.is_dir():
                raise ValueError(f"Missing/out-of-batch run directory: {directory}")
            runs[dataset, record["method"]] = (directory, record)
    target_dir, target = runs[TARGET]
    probe_path = target_dir / "repeatability_probe.json"
    if not probe_path.is_file():
        raise ValueError(f"Missing probe evidence: {probe_path}")
    probe = read(probe_path)
    failures = [
        list(key)
        for key, (_, r) in runs.items()
        if r.get("status") != "measured"
        or not r.get("comparable")
        or r.get("process_exit_code", 0) != 0
    ]
    result = dict(
        batch_root=str(batch),
        expected_optimizer_runs=20,
        measured_runs=20 - len(failures),
        incomplete_runs=failures,
        probe_path=str(probe_path),
        probe=probe_details(probe),
        predictions=probe.get("predictions", []),
        diagnosis="No optimizer or inference-backend change is authorized by this recovery tool.",
    )
    return result, reports, runs


def validate_plan(batch, repo):
    info, reports, runs = inspect_batch(batch)
    if info["incomplete_runs"] != [list(TARGET)]:
        raise ValueError(
            "Recovery requires exactly 19 measured runs and only Tri-Fair/Bios incomplete"
        )
    if (
        info["probe"]["status"] != "failed"
        or info["probe"]["reason"] != "batch_order_disagreement"
    ):
        raise ValueError(
            "Do not retry blindly: inspect invalid labels or missing probe evidence first"
        )
    if os.environ.get("VLLM_BATCH_INVARIANT", "0") not in {"", "0"}:
        raise ValueError(
            "Do not change batch-invariance for only one method; keep the original backend profile"
        )
    current = source_digest(repo)
    for (dataset, method), (directory, record) in runs.items():
        if (dataset, method) == TARGET:
            continue
        summary = read(directory / "run_summary.json")
        if (
            summary.get("status") != "complete"
            or summary.get("source_sha256") != current
        ):
            raise ValueError(
                f"Incomplete or changed measured source for {dataset}/{method}; restore the exact v8.8 source"
            )
        if (summary.get("model"), summary.get("dataset"), summary.get("optimizer")) != (
            "qwen-3-30b",
            dataset,
            method,
        ):
            raise ValueError("Run summary identity mismatch")
        if int(summary.get("requested_budget", 0)) != 2000000:
            raise ValueError("Recovery is restricted to the original 2M pilot")
        if (summary.get("final_downstream_tokens") or {}).get(
            "total_tokens", 2000001
        ) > 2000000:
            raise ValueError("Existing run exceeded its downstream cap")
    for _, report in reports.values():
        if report.get("budget_per_method") != 2000000:
            raise ValueError("Pilot policy budget mismatch")
        policy = report["pilot_configuration"]
        expected = dict(
            n_init_prompts=4,
            max_few_shot_examples=2,
            crossovers_per_iteration=1,
            final_search_candidates=4,
            final_search_reserve=0.30,
            reclaim_audit_reserve=True,
            structured_labels=True,
        )
        if any(policy.get(k) != v for k, v in expected.items()):
            raise ValueError(
                "Existing pilot settings differ from the matched recovery protocol"
            )
    old_args = read(runs[TARGET][0] / "args.json")
    if (
        old_args.get("optimizer_overrides")
        or old_args.get("batch_invariant")
        or old_args.get("enable_prefix_caching")
    ):
        raise ValueError(
            "Nondefault old runtime/optimizer settings require explicit reviewed recovery"
        )
    expected_args = dict(
        model="qwen-3-30b",
        dataset="bias_in_bios",
        random_seed=72,
        optimizer=TARGET[1],
        budget_per_run=2000000,
        structured_labels=True,
        max_output_tokens=16,
        meta_max_output_tokens=512,
        max_steps=200,
        max_idle_steps=3,
        n_init_prompts=4,
        max_few_shot_examples=2,
        crossovers_per_iteration=1,
    )
    if any(old_args.get(k) != v for k, v in expected_args.items()):
        raise ValueError(
            "Failed run arguments differ from the supported original v8.8 pilot"
        )
    info.update(
        retry_method=TARGET[1],
        retry_dataset=TARGET[0],
        retry_seed=72,
        retry_downstream_cap=2000000,
        source_sha256=current,
        baseline_runs_to_retrain=0,
        changes_optimizer=False,
        changes_backend_flags=False,
        repeatability_gate_preserved=True,
        gpu_retry="one H200 from any eligible Firefly node; not a guarantee of stability",
    )
    return info, reports, runs, old_args


def command_for_retry(policy, old_args, root, train_command):
    cfg = policy["pilot_configuration"]
    args = SimpleNamespace(
        experiment_name=old_args.get(
            "experiment_name", "tri_fair_v88_rendered_edits_pilot"
        ),
        model="qwen-3-30b",
        dataset="bias_in_bios",
        seed=72,
        budget=2000000,
        manifest_dir=old_args.get("manifest_dir", "data/splits_v8_clean"),
        n_init_prompts=cfg["n_init_prompts"],
        max_few_shot_examples=cfg["max_few_shot_examples"],
        crossovers_per_iteration=cfg["crossovers_per_iteration"],
        final_search_candidates=cfg["final_search_candidates"],
        final_search_reserve=cfg["final_search_reserve"],
        reclaim_audit_reserve=cfg["reclaim_audit_reserve"],
        unconstrained_labels=not cfg["structured_labels"],
    )
    return train_command(args, TARGET[1], Path(root) / "training")


def make_view(output, runs, recovered):
    """Use existing evaluator's standard pointers; never move/delete source runs."""
    output = Path(output)
    for (dataset, method), (directory, _) in runs.items():
        source = Path(recovered) if (dataset, method) == TARGET else directory
        if not source.is_dir():
            raise ValueError(f"Missing measured run: {source}")
        pointer = (
            output / "qwen-3-30b" / dataset / method / "seed72" / "logging_dir.txt"
        )
        pointer.parent.mkdir(parents=True, exist_ok=True)
        with pointer.open("x", encoding="utf-8") as stream:
            stream.write(str(source.resolve()) + "\n")


def validate_output(batch, output):
    output = Path(output).expanduser().resolve()
    batch = Path(batch).expanduser().resolve()
    if output == batch or not output.is_relative_to(batch) or output.exists():
        raise ValueError(
            "Use a NEW recovery output directory inside the original batch"
        )
    return output


def execute(batch, repo, output, *, bridge=None, run_process=subprocess.run):
    info, reports, runs, old_args = validate_plan(batch, repo)
    output = validate_output(batch, output)
    if bridge is None:
        sys.path.insert(0, str(Path(repo).resolve() / "src"))
        from tri_fair.cli import pilot as bridge
    output.mkdir(parents=True, exist_ok=False)
    command = command_for_retry(
        reports[TARGET[0]][1], old_args, output, bridge.train_command
    )
    receipt = dict(
        info,
        started_utc=datetime.now(timezone.utc).isoformat(),
        original_results_modified=False,
        original_failed_probe=info["probe"],
        original_failed_probe_tokens_additional=info["probe"][
            "actual_downstream_tokens"
        ],
        retry_command=command,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        status="retry_running",
        validation_ready=False,
    )
    write(output / "recovery_receipt.json", receipt)
    logfile = output / "Tri-Fair-v8.8_recovery.log"
    print("Retry command:", shlex.join(command), flush=True)
    with logfile.open("w", encoding="utf-8") as stream:
        process = run_process(
            command, stdout=stream, stderr=subprocess.STDOUT, check=False
        )
    recovered = output / "training" / TARGET[1] / "run"
    receipt["retry_process_exit_code"] = process.returncode
    new_probe = recovered / "repeatability_probe.json"
    receipt["retry_probe"] = read(new_probe) if new_probe.is_file() else None
    if process.returncode:
        receipt["status"] = "retry_failed"
        write(output / "recovery_receipt.json", receipt)
        print(
            f"Retry failed; inspect {logfile} and its repeatability_probe.json. Validation remains blocked."
        )
        return 1
    record = bridge.read_result(recovered, TARGET[1])
    record["process_exit_code"] = process.returncode
    if (
        not record.get("comparable")
        or record.get("status") != "measured"
        or not receipt["retry_probe"]
        or receipt["retry_probe"].get("status") != "passed"
        or receipt["retry_probe"].get("mismatches")
        or receipt["retry_probe"].get("invalid_label_examples")
    ):
        receipt["status"] = "retry_inconclusive"
        write(output / "recovery_receipt.json", receipt)
        raise RuntimeError(
            "Retry did not produce measured, probe-passing evidence; no evaluation view created"
        )
    controls = [r for _, r in (runs[TARGET[0], m] for m in METHODS if m != TARGET[1])]
    comparisons = bridge.compare_results([record, *controls], variant_name=TARGET[1])
    if not comparisons or not all(r.get("comparable") for r in comparisons):
        receipt["status"] = "runtime_or_protocol_mismatch"
        receipt["comparisons"] = comparisons
        write(output / "recovery_receipt.json", receipt)
        raise RuntimeError(
            "Recovered run differs from saved control protocol; no evaluation view created"
        )
    make_view(output, runs, recovered)
    receipt.update(
        status="recovered",
        validation_ready=True,
        recovered_run_directory=str(recovered),
        measured_recovered_run=record,
        preserved_method_runs=19,
        note="Original Slurm task remains FAILED; this separate retry supplies the missing measured run.",
    )
    write(output / "recovery_receipt.json", receipt)
    print(
        f"Recovered one method; preserved 19. Validation input view: {output}",
        flush=True,
    )
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["inspect", "plan", "execute"])
    parser.add_argument("--batch-root", required=True)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output-root")
    args = parser.parse_args()
    try:
        if args.action == "inspect":
            print(json.dumps(inspect_batch(args.batch_root)[0], indent=2))
        elif args.action == "plan":
            info = validate_plan(args.batch_root, args.repo_root)[0]
            if args.output_root:
                info["recovery_output"] = str(
                    validate_output(args.batch_root, args.output_root)
                )
            print(json.dumps(info, indent=2))
        else:
            if not args.output_root:
                parser.error("execute requires --output-root")
            return execute(args.batch_root, args.repo_root, args.output_root)
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"Recovery blocked: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
