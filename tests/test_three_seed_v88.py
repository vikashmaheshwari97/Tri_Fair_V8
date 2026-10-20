"""CPU verification of the frozen 60-run launch; no Slurm or model calls."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "research" / f"{name}.py"
    )
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


matrix = module("run_three_seed_v88")
style = module("figure_style_v88")


def clean_env(monkeypatch):
    for key in ("MODEL", "PILOT_SEEDS", "PILOT_BUDGET", "MAX_CONCURRENT"):
        monkeypatch.delenv(key, raising=False)


def test_plan_covers_exactly_sixty_runs(monkeypatch):
    clean_env(monkeypatch)
    p = matrix.plan()
    assert p["task_count"] == 12 and p["optimizer_runs"] == 60
    assert p["downstream_ceiling"] == 120_000_000
    assert p["max_concurrent_gpus"] == 5
    assert {matrix.task(i) for i in range(12)} == {
        (d, s) for d in matrix.DATASETS for s in (72, 73, 74)
    }
    assert matrix.task(2) == ("bias_in_bios", 72)
    assert matrix.task(11) == ("winobias", 74)


@pytest.mark.parametrize(
    "key,value",
    [
        ("PILOT_SEEDS", "72"),
        ("PILOT_BUDGET", "5000000"),
        ("MAX_CONCURRENT", "0"),
        ("MODEL", "mistral-3-24b"),
    ],
)
def test_stale_matrix_settings_fail_before_submission(monkeypatch, key, value):
    clean_env(monkeypatch)
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        matrix.plan()


def setup_saved(tmp_path, monkeypatch):
    clean_env(monkeypatch)
    monkeypatch.setenv("PILOT_RESULTS_ROOT", str(tmp_path / "batch"))
    monkeypatch.setenv("MANIFEST_DIR", str(tmp_path / "manifests"))
    monkeypatch.setattr(matrix, "code_hashes", lambda: {"file": "unchanged"})
    monkeypatch.setattr(matrix, "package_versions", lambda: {"package": "version"})
    base = matrix.init()
    return base, matrix.load_config()[1]


def test_saved_batch_refuses_source_change_and_duplicate_submission(
    tmp_path, monkeypatch
):
    setup_saved(tmp_path, monkeypatch)
    with pytest.raises(FileExistsError):
        matrix.init()
    monkeypatch.setattr(matrix, "code_hashes", lambda: {"file": "changed"})
    with pytest.raises(ValueError, match="changed since submission"):
        matrix.load_config()


def test_saved_batch_refuses_package_change(tmp_path, monkeypatch):
    setup_saved(tmp_path, monkeypatch)
    monkeypatch.setattr(matrix, "package_versions", lambda: {"package": "new-version"})
    with pytest.raises(ValueError, match="Package versions"):
        matrix.load_config()


def make_report(base, dataset, seed):
    path = base / "qwen-3-30b" / dataset / f"seed{seed}" / "stamp" / "pilot_report.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            dict(
                all_controls_comparable=True,
                runs=[
                    dict(method=m, status="measured", process_exit_code=0)
                    for m in matrix.METHODS
                ],
            )
        )
    )
    return path


def test_validation_passes_batch_invariance_and_reuses_training(tmp_path, monkeypatch):
    base, config = setup_saved(tmp_path, monkeypatch)
    path = make_report(base, "bias_in_bios", 72)
    before = path.read_bytes()
    calls = []
    monkeypatch.setattr(matrix, "run", lambda cmd: calls.append(list(map(str, cmd))))
    matrix.evaluate(base, config, 2)
    assert len(calls) == 2
    assert "tri_fair.cli.evaluate" in calls[0]
    for flag in (
        "--batch-invariant",
        "--structured-labels",
        "--audited-final",
        "--include-history",
    ):
        assert flag in calls[0]
    assert calls[0][calls[0].index("--study-root") + 1] == str(path.parent)
    assert not any("tri_fair.cli.train" in cmd for cmd in calls)
    assert before == path.read_bytes()


def test_validation_refuses_incomplete_method(tmp_path, monkeypatch):
    base, config = setup_saved(tmp_path, monkeypatch)
    path = make_report(base, "bbq", 72)
    saved = json.loads(path.read_text())
    saved["runs"][0]["status"] = "failed"
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="Incomplete"):
        matrix.evaluate(base, config, 0)


def test_training_does_not_overwrite_existing_run(tmp_path, monkeypatch):
    base, config = setup_saved(tmp_path, monkeypatch)
    make_report(base, "bbq", 72)
    with pytest.raises(ValueError, match="already exists"):
        matrix.train(base, config, 0)


def test_output_share_is_tokens_not_price_weighted_cost():
    frame = pd.DataFrame(
        dict(validation_input_tokens=[90, 60], validation_output_tokens=[10, 40])
    )
    np.testing.assert_allclose(style.output_share(frame, "validation"), [0.1, 0.4])
    frame.loc[0, "validation_input_tokens"] = np.nan
    with pytest.raises(ValueError):
        style.output_share(frame, "validation")


def test_report_rejects_incomplete_heldout_matrix(tmp_path):
    with pytest.raises(ValueError, match="twelve"):
        matrix.report(tmp_path)


@pytest.mark.skipif(
    os.name == "nt" or not shutil.which("bash"), reason="Rocket Bash test"
)
def test_dry_run_submits_nothing_and_uses_five_gpu_task_limit(tmp_path):
    bins = tmp_path / "bin"
    bins.mkdir()
    stub = bins / "sbatch"
    stub.write_text("#!/bin/sh\necho ERROR_ACTUAL_SUBMIT >&2\nexit 99\n")
    stub.chmod(0o755)
    env = dict(
        os.environ,
        MODEL="qwen-3-30b",
        PILOT_SEEDS="72,73,74",
        PILOT_BUDGET="2000000",
        MAX_CONCURRENT="5",
        DRY_RUN="1",
        PATH=str(bins) + os.pathsep + os.environ["PATH"],
    )
    result = subprocess.run(
        ["bash", "jobs/submit_three_seed_v88.sh"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "ERROR_ACTUAL_SUBMIT" not in result.stderr
    assert result.stdout.count("--array=0-11%5") == 2
    assert (
        "afterok:TRAIN_JOB" in result.stdout
        and "afterok:VALIDATION_JOB" in result.stdout
    )
    assert result.stdout.count("--kill-on-invalid-dep=yes") == 2


def test_plot_uses_exact_mocapo_palette_and_requested_axis_labels(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "research"))
    plots = module("plot_mocapo_style")
    assert plots.STYLE["Tri-Fair-v8.8"][1:] == ("#000000", "o")
    assert plots.STYLE["NSGAII-PO-Fair"][1:] == ("#E69F00", "s")
    assert style.COST_LABEL == "Avg. Cost [$] per 1M Calls"
    assert style.TOKEN_LABEL == "Token Budget [×10⁶]"


@pytest.mark.skipif(
    os.name == "nt" or not shutil.which("bash"), reason="Rocket Bash test"
)
@pytest.mark.parametrize("fail_validation", [False, True])
def test_submission_dependency_chain_with_stub_scheduler(tmp_path, fail_validation):
    repo = tmp_path / "repo"
    (repo / "jobs").mkdir(parents=True)
    shutil.copy(ROOT / "jobs/submit_three_seed_v88.sh", repo / "jobs")
    bins = repo / "bin"
    bins.mkdir()
    # Stub only the external Python entry points and scheduler; execute real Bash orchestration.
    python = bins / "python"
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    sbatch = bins / "sbatch"
    sbatch.write_text("""#!/usr/bin/env bash
set -eu
n=0
[[ ! -e "$TEST_COUNT" ]] || n=$(cat "$TEST_COUNT")
n=$((n+1)); echo "$n" > "$TEST_COUNT"
printf '%s|%s\\n' "$TF_MATRIX_STAGE" "$*" >> "$TEST_LOG"
if [[ "$n" == 2 && "$FAIL_VALIDATION" == 1 ]]; then exit 1; fi
echo "$((1000+n))"
""")
    sbatch.chmod(0o755)
    env = dict(
        os.environ,
        MODEL="qwen-3-30b",
        PILOT_SEEDS="72,73,74",
        PILOT_BUDGET="2000000",
        MAX_CONCURRENT="5",
        DRY_RUN="0",
        PATH=str(bins) + os.pathsep + os.environ["PATH"],
        TEST_COUNT=str(repo / "count"),
        TEST_LOG=str(repo / "calls"),
        FAIL_VALIDATION=str(int(fail_validation)),
    )
    result = subprocess.run(
        ["bash", "jobs/submit_three_seed_v88.sh"],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
    )
    calls = (repo / "calls").read_text().splitlines()
    assert calls[0].startswith("train|") and "--array=0-11%5" in calls[0]
    assert calls[1].startswith("evaluate|") and "--dependency=afterok:1001" in calls[1]
    if fail_validation:
        assert result.returncode != 0 and len(calls) == 2
        assert "Training 1001 already submitted" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert len(calls) == 3 and "--dependency=afterok:1002" in calls[2]
