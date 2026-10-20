"""Recovery must preserve nineteen measured runs and never bypass the probe."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "recovery_test_module", ROOT / "recovery/v88_recover.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def fixture(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / "src/unchanged.py").write_text("# unchanged measured code\n")
    batch = repo / "results/batch_original"
    policy = dict(
        n_init_prompts=4,
        max_few_shot_examples=2,
        crossovers_per_iteration=1,
        final_search_candidates=4,
        final_search_reserve=0.30,
        reclaim_audit_reserve=True,
        structured_labels=True,
    )
    for dataset in module.DATASETS:
        root = batch / "qwen-3-30b" / dataset / "seed72/stamp"
        records = []
        for method in module.METHODS:
            run = root / method / "run"
            run.mkdir(parents=True)
            failed = (dataset, method) == module.TARGET
            record = dict(
                method=method,
                status="failed" if failed else "measured",
                comparable=not failed,
                process_exit_code=int(failed),
                source_sha256=module.source_digest(repo),
            )
            records.append(record)
            module.write(
                run / "run_summary.json",
                dict(
                    status="failed" if failed else "complete",
                    source_sha256=module.source_digest(repo),
                    model="qwen-3-30b",
                    dataset=dataset,
                    optimizer=method,
                    requested_budget=2000000,
                    final_downstream_tokens=dict(total_tokens=1800000),
                ),
            )
            if failed:
                module.write(
                    run / "repeatability_probe.json",
                    dict(
                        status="failed",
                        invalid_label_examples=[],
                        mismatches=[
                            dict(
                                example_id="example",
                                batched="nurse",
                                single="physician",
                            )
                        ],
                        actual_downstream_tokens=2658,
                        prediction_count=24,
                    ),
                )
                module.write(
                    run / "args.json",
                    dict(
                        model="qwen-3-30b",
                        dataset="bias_in_bios",
                        random_seed=72,
                        optimizer=module.TARGET[1],
                        budget_per_run=2000000,
                        structured_labels=True,
                        max_output_tokens=16,
                        meta_max_output_tokens=512,
                        max_steps=200,
                        max_idle_steps=3,
                        n_init_prompts=4,
                        max_few_shot_examples=2,
                        crossovers_per_iteration=1,
                        manifest_dir="data/frozen",
                    ),
                )
        module.write(
            root / "pilot_report.json",
            dict(
                dataset=dataset,
                model="qwen-3-30b",
                seed=72,
                runs=records,
                budget_per_method=2000000,
                pilot_configuration=policy,
            ),
        )
    return repo, batch


def snapshot(path):
    return {
        str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*") if p.is_file()
    }


def test_diagnosis_identifies_only_one_missing_run_and_mismatch(fixture):
    repo, batch = fixture
    info, *_ = module.validate_plan(batch, repo)
    assert info["measured_runs"] == 19
    assert info["incomplete_runs"] == [list(module.TARGET)]
    assert info["probe"]["reason"] == "batch_order_disagreement"
    assert info["baseline_runs_to_retrain"] == 0


@pytest.mark.parametrize("reason", ["invalid", "unknown", "missing"])
def test_invalid_or_unknown_probe_never_allows_blind_retry(fixture, reason):
    repo, batch = fixture
    run = module.inspect_batch(batch)[2][module.TARGET][0]
    path = run / "repeatability_probe.json"
    if reason == "missing":
        path.unlink()
    else:
        probe = module.read(path)
        probe["mismatches"] = []
        if reason == "invalid":
            probe["invalid_label_examples"] = ["example"]
        module.write(path, probe)
    with pytest.raises(ValueError):
        module.validate_plan(batch, repo)


def test_source_change_blocks_reuse_but_recovery_files_do_not_change_digest(fixture):
    repo, batch = fixture
    expected = module.source_digest(repo)
    (repo / "recovery").mkdir()
    (repo / "recovery/helper.py").write_text("# operational helper\n")
    assert module.source_digest(repo) == expected
    (repo / "src/unchanged.py").write_text("# changed optimizer code\n")
    with pytest.raises(ValueError, match="changed measured source"):
        module.validate_plan(batch, repo)


def test_invariance_profile_cannot_change_for_only_one_method(fixture, monkeypatch):
    repo, batch = fixture
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    with pytest.raises(ValueError, match="batch-invariance"):
        module.validate_plan(batch, repo)


def bridge_for_test():
    def train_command(args, method, root):
        return [
            "test-runner",
            "--optimizer",
            method,
            "--budget-per-run",
            str(args.budget),
            "--output-dir",
            str(root / method),
        ]

    def read_result(run, method):
        return dict(method=method, status="measured", comparable=True)

    def compare_results(records, variant_name):
        return [dict(comparable=True) for r in records if r["method"] != variant_name]

    return SimpleNamespace(
        train_command=train_command,
        read_result=read_result,
        compare_results=compare_results,
    )


def fake_success(command, **kwargs):
    root = Path(command[command.index("--output-dir") + 1]) / "run"
    module.write(
        root / "repeatability_probe.json",
        dict(status="passed", mismatches=[], invalid_label_examples=[]),
    )
    return SimpleNamespace(returncode=0)


def test_successful_retry_creates_twenty_pointers_without_modifying_old_data(fixture):
    repo, batch = fixture
    original = snapshot(batch)
    output = batch / "recovery_new"
    assert (
        module.execute(
            batch, repo, output, bridge=bridge_for_test(), run_process=fake_success
        )
        == 0
    )
    assert len(list((output / "qwen-3-30b").rglob("logging_dir.txt"))) == 20
    for relative, data in original.items():
        assert (batch / relative).read_bytes() == data
    target = output / "qwen-3-30b/bias_in_bios/Tri-Fair-v8.8/seed72/logging_dir.txt"
    assert target.read_text().strip() == str(
        (output / "training/Tri-Fair-v8.8/run").resolve()
    )
    receipt = module.read(output / "recovery_receipt.json")
    assert receipt["validation_ready"] and receipt["preserved_method_runs"] == 19
    assert receipt["original_failed_probe_tokens_additional"] == 2658
    assert not receipt["original_results_modified"]


def test_failed_retry_never_creates_evaluation_pointers(fixture):
    repo, batch = fixture
    output = batch / "recovery_failed"
    assert (
        module.execute(
            batch,
            repo,
            output,
            bridge=bridge_for_test(),
            run_process=lambda *a, **k: SimpleNamespace(returncode=1),
        )
        == 1
    )
    assert not list(output.rglob("logging_dir.txt"))
    assert module.read(output / "recovery_receipt.json")["status"] == "retry_failed"


def test_return_zero_without_passing_probe_still_blocks_validation(fixture):
    repo, batch = fixture
    output = batch / "recovery_bad_probe"

    def runner(command, **kwargs):
        fake_success(command)
        run = Path(command[command.index("--output-dir") + 1]) / "run"
        module.write(
            run / "repeatability_probe.json", dict(status="failed", mismatches=["x"])
        )
        return SimpleNamespace(returncode=0)

    with pytest.raises(RuntimeError, match="probe-passing"):
        module.execute(
            batch, repo, output, bridge=bridge_for_test(), run_process=runner
        )
    assert not list(output.rglob("logging_dir.txt"))


def test_existing_or_external_output_is_rejected(fixture):
    repo, batch = fixture
    for output in [batch, repo / "outside"]:
        with pytest.raises(ValueError, match="NEW recovery"):
            module.execute(
                batch, repo, output, bridge=bridge_for_test(), run_process=fake_success
            )


def test_mismatched_control_protocol_blocks_view(fixture):
    repo, batch = fixture
    bridge = bridge_for_test()
    bridge.compare_results = lambda *a, **k: [dict(comparable=False)]
    output = batch / "recovery_mismatch"
    with pytest.raises(RuntimeError, match="control protocol"):
        module.execute(batch, repo, output, bridge=bridge, run_process=fake_success)
    assert not list(output.rglob("logging_dir.txt"))


@pytest.mark.skipif(os.name == "nt" or not shutil.which("bash"), reason="requires Bash")
def test_submission_uses_one_h200_retry_and_dependent_validation(fixture):
    repo, batch = fixture
    shutil.copytree(ROOT / "recovery", repo / "recovery")
    shutil.copytree(ROOT / "jobs", repo / "jobs")
    stub = repo / "bin"
    stub.mkdir()
    log = repo / "sbatch_calls.jsonl"
    sbatch = stub / "sbatch"
    sbatch.write_text(
        "#!"
        + sys.executable
        + "\nimport os,sys,json,pathlib\np=pathlib.Path(os.environ['RECOVERY_STUB_LOG'])\nwith p.open('a') as f:f.write(json.dumps(sys.argv[1:])+'\\n')\nprint(9900000+len(p.read_text().splitlines()))\n"
    )
    sbatch.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(stub) + os.pathsep + os.environ["PATH"],
        "RECOVERY_STUB_LOG": str(log),
        "RECOVERY_BATCH": str(batch),
        "DRY_RUN": "0",
    }
    env.pop("VLLM_BATCH_INVARIANT", None)
    result = subprocess.run(
        ["bash", "jobs/submit_recover_v88.sh"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    calls = [json.loads(s) for s in log.read_text().splitlines()]
    assert len(calls) == 3
    assert "--gpus-per-task=h200-141g:1" in calls[0]
    assert not any(s.startswith("--array=") for s in calls[0])
    assert "--array=0-3%2" in calls[1]
    assert "--dependency=afterok:9900001" in calls[1]
    assert "--dependency=afterok:9900002" in calls[2]
