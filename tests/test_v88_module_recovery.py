"""Startup and validation-only submission tests. No Slurm or GPU calls."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import venv

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(os.name == "nt" or shutil.which("bash") is None, reason="Linux Bash startup tests")


def test_existing_venv_works_when_python_venv_module_is_unavailable(tmp_path):
    env_path = tmp_path / "venv"
    venv.EnvBuilder(with_pip=False).create(env_path)
    log = tmp_path / "modules.log"
    env = dict(os.environ, TF_VENV=str(env_path), MODULE_LOG=str(log), TF_HELPER=str(ROOT / "jobs/lib/activate_existing_venv.sh"))
    script = '''
set -Eeuo pipefail
module() {
  printf '%s\n' "$*" >> "$MODULE_LOG"
  [[ "$*" != *python-venv* ]]
}
source "$TF_HELPER"
tf_activate_existing_venv
'''
    result = subprocess.run(["bash", "-c", script], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert '"stage": "existing_venv_activated"' in result.stdout
    assert str(env_path) in result.stdout
    assert "python-venv" not in log.read_text()
    assert "--ignore_cache load python/3.11.14" in log.read_text()


@pytest.mark.parametrize("failure", ["missing_base_module", "missing_venv", "fake_activation"])
def test_bootstrap_stops_before_using_an_invalid_environment(tmp_path, failure):
    env_path = tmp_path / "venv"
    if failure == "fake_activation":
        (env_path / "bin").mkdir(parents=True)
        (env_path / "bin/activate").write_text('export VIRTUAL_ENV="$TF_VENV"\n')
        (env_path / "bin/python").symlink_to(shutil.which("python"))
    env = dict(os.environ, TF_VENV=str(env_path), TF_HELPER=str(ROOT / "jobs/lib/activate_existing_venv.sh"))
    module = "module() { return 1; }" if failure == "missing_base_module" else "module() { return 0; }"
    result = subprocess.run(["bash", "-c", 'set -Eeuo pipefail\n' + module + '\nsource "$TF_HELPER"\ntf_activate_existing_venv\necho SHOULD_NOT_RUN'], env=env, text=True, capture_output=True)
    assert result.returncode != 0
    assert "SHOULD_NOT_RUN" not in result.stdout


def setup_batch(tmp_path):
    repo = tmp_path / "repo"
    (repo / "jobs").mkdir(parents=True)
    shutil.copy(ROOT / "jobs/submit_bios_validation_v88.sh", repo / "jobs")
    (repo / "recovery").mkdir()
    # Stub ONLY the previously tested proof verifier. Submission/report checks
    # below execute the real Bash helper and inline Python code.
    (repo / "recovery/v88_bios_stability.py").write_text('print("Proof verifier called")\n')
    batch = repo / "results/bios"
    report_path = batch / "qwen-3-30b/bias_in_bios/seed72/stamp/pilot_report.json"
    report_path.parent.mkdir(parents=True)
    methods = ["Tri-Fair-v8.8", "NSGAII-PO-Fair", "NSGAIII-PO-Fair", "SPEA2-PO-Fair", "RandomSearch-PO-Fair"]
    report = dict(all_controls_comparable=True, runs=[dict(method=m, status="measured", process_exit_code=0) for m in methods])
    report_path.write_text(json.dumps(report))
    proof = repo / "proof.json"
    proof.write_text('{"status":"passed"}')
    digest = hashlib.sha256(proof.read_bytes()).hexdigest()
    profile = dict(batch_invariant=True, dataset="bias_in_bios", seed=72, budget=2000000, proof_path=str(proof), proof_sha256=digest)
    (batch / "stable_profile.json").write_text(json.dumps(profile))
    (batch / "ready_for_validation.json").write_text(json.dumps(dict(report=str(report_path), proof_sha256=digest)))
    bins = repo / "bin"
    bins.mkdir()
    stub = bins / "sbatch"
    stub.write_text('#!/usr/bin/env bash\nprintf "%s|%s|%s\\n" "$STABILITY_STAGE" "$BIOS_OUTPUT" "$*" >> "$SUBMIT_LOG"\necho 12345\n')
    stub.chmod(0o755)
    env = dict(os.environ, BIOS_OUTPUT=str(batch), SUBMIT_LOG=str(repo / "submissions.log"), PATH=str(bins) + os.pathsep + os.environ["PATH"])
    return repo, batch, report_path, env


def test_retry_submits_exactly_one_evaluation_and_preserves_training(tmp_path):
    repo, batch, _, env = setup_batch(tmp_path)
    before = {str(p): p.read_bytes() for p in batch.rglob("*") if p.is_file()}
    result = subprocess.run(["bash", "jobs/submit_bios_validation_v88.sh"], cwd=repo, env=dict(env, DRY_RUN="0"), text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    lines = (repo / "submissions.log").read_text().splitlines()
    assert len(lines) == 1 and lines[0].startswith("evaluate|")
    assert "jobs/bios_stability_v88.sbatch" in lines[0]
    assert "Validation and figure job=12345" in result.stdout
    assert before == {str(p): p.read_bytes() for p in batch.rglob("*") if p.is_file()}


def test_dry_run_does_not_submit(tmp_path):
    repo, _, _, env = setup_batch(tmp_path)
    result = subprocess.run(["bash", "jobs/submit_bios_validation_v88.sh"], cwd=repo, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "DRY RUN:" in result.stdout
    assert not (repo / "submissions.log").exists()


@pytest.mark.parametrize("failure", ["incomplete_method", "wrong_proof", "external_report"])
def test_invalid_saved_batch_cannot_submit(tmp_path, failure):
    repo, batch, report_path, env = setup_batch(tmp_path)
    if failure == "incomplete_method":
        report = json.loads(report_path.read_text())
        report["runs"][0]["status"] = "failed"
        report_path.write_text(json.dumps(report))
    else:
        ready_path = batch / "ready_for_validation.json"
        ready = json.loads(ready_path.read_text())
        ready["proof_sha256" if failure == "wrong_proof" else "report"] = "invalid" if failure == "wrong_proof" else str(repo / "external.json")
        ready_path.write_text(json.dumps(ready))
    result = subprocess.run(["bash", "jobs/submit_bios_validation_v88.sh"], cwd=repo, env=dict(env, DRY_RUN="0"), text=True, capture_output=True)
    assert result.returncode != 0
    assert not (repo / "submissions.log").exists()
