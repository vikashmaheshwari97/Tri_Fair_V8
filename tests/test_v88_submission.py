"""The redesigned pilot must reject stale settings before allocating GPUs."""

import os
from pathlib import Path
import shutil
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    os.name == "nt" or not shutil.which("bash"),
    reason="Rocket Bash checks require Linux Bash",
)


def run(**env):
    clean = {
        k: v
        for k, v in os.environ.items()
        if k
        not in {
            "MODEL",
            "PILOT_SEEDS",
            "SEED",
            "PILOT_METHODS",
            "PILOT_GPU_TYPE",
            "MAX_CONCURRENT",
            "PILOT_RESULTS_ROOT",
            "DRY_RUN",
        }
    }
    return subprocess.run(
        ["bash", "jobs/submit_pilot_v88.sh"],
        cwd=ROOT,
        env={**clean, "DRY_RUN": "1", **env},
        capture_output=True,
        text=True,
    )


def test_default_screen_has_five_methods_four_tasks_and_dependency_chain():
    result = run(PILOT_SEEDS="72", MAX_CONCURRENT="2")
    assert result.returncode == 0, result.stderr
    assert (
        "Tri-Fair-v8.8,NSGAII-PO-Fair,NSGAIII-PO-Fair,SPEA2-PO-Fair,RandomSearch-PO-Fair"
        in result.stdout
    )
    assert result.stdout.count("--array=0-3%2") == 2
    assert "--dependency=afterok:TRAIN_JOB" in result.stdout
    assert "--dependency=afterok:VALIDATION_JOB" in result.stdout
    assert "jobs/pilot_v88.sbatch" in result.stdout
    assert "results/tri_fair_v88_pilot_2000000/batch_" in result.stdout


def test_stale_method_or_incompatible_gpu_is_rejected_and_gpu_family_is_shared():
    assert run(PILOT_METHODS="Tri-Fair-v8.6,NSGAII-PO-Fair").returncode != 0
    assert run(MODEL="gpt-oss-120b", PILOT_GPU_TYPE="a100-80g").returncode != 0
    assert run(PILOT_METHODS="Tri-Fair-v8.8,Tri-Fair-v8.8").returncode != 0
    result = run(PILOT_GPU_TYPE="h200-141g")
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("--gpus-per-task=h200-141g:1") == 2


def test_default_three_seed_matrix_is_twelve_tasks_not_six():
    result = run()
    assert result.returncode == 0, result.stderr
    assert "seeds=72,73,74" in result.stdout
    assert result.stdout.count("--array=0-11%5") == 2
    assert "Optimizer runs=60" in result.stdout
    assert "downstream ceiling=120000000 tokens" in result.stdout
