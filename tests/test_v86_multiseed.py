"""Exercise the Slurm task mapping, dependency wiring and complete reporting matrix."""

import os
import subprocess
import shutil
from pathlib import Path

import pandas as pd
import pytest

from tests.test_research_figures import rows
from tri_fair.reporting.multiseed_report import DATASETS, validate_matrix

ROOT = Path(__file__).resolve().parents[1]
shell_test = pytest.mark.skipif(
    os.name == "nt" or shutil.which("bash") is None,
    reason="Rocket Bash submission checks require Linux Bash",
)


def bash(script, **env):
    clean = {
        k: v
        for k, v in os.environ.items()
        if k not in {"PILOT_SEEDS", "SEED", "MAX_CONCURRENT", "PILOT_METHODS"}
    }
    return subprocess.run(
        ["bash", "-c", script],
        cwd=ROOT,
        env={**clean, **env},
        capture_output=True,
        text=True,
    )


@shell_test
def test_twelve_tasks_cover_exactly_three_seeds_and_four_datasets():
    result = bash(
        'source jobs/lib/pilot_v86_matrix.sh; tf_v86_matrix || exit; for ((i=0;i<tf_v86_tasks;i++)); do tf_v86_task "$i" || exit; echo "$seed/$dataset"; done',
        PILOT_SEEDS="72,73,74",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        f"{seed}/{dataset}" for seed in (72, 73, 74) for dataset in DATASETS
    ]
    # Existing single-seed invocations still map exactly four tasks.
    result = bash(
        'source jobs/lib/pilot_v86_matrix.sh; tf_v86_matrix; tf_v86_task 3; echo "$tf_v86_tasks/$seed/$dataset"',
        SEED="73",
    )
    assert result.stdout.strip() == "4/73/winobias"


@pytest.mark.parametrize("seeds", ["72,72", "72,", "72,,73", "072", "-1", "72,x"])
@shell_test
def test_invalid_seed_lists_fail(seeds):
    result = bash(
        "source jobs/lib/pilot_v86_matrix.sh; tf_v86_matrix", PILOT_SEEDS=seeds
    )
    assert result.returncode != 0


@shell_test
def test_array_bounds_and_dry_run_dependencies():
    result = bash(
        "source jobs/lib/pilot_v86_matrix.sh; tf_v86_matrix; tf_v86_task 12",
        PILOT_SEEDS="72,73,74",
    )
    assert result.returncode != 0
    result = bash(
        "bash jobs/submit_pilot_v86.sh",
        PILOT_SEEDS="72,73,74",
        MAX_CONCURRENT="5",
        DRY_RUN="1",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("--array=0-11%5") == 2
    assert "--dependency=afterok:TRAIN_JOB" in result.stdout
    assert "--dependency=afterok:VALIDATION_JOB" in result.stdout
    result = bash(
        "bash jobs/submit_pilot_v86.sh", SEED="72", MAX_CONCURRENT="5", DRY_RUN="1"
    )
    assert result.returncode != 0


def full_matrix():
    return pd.concat(
        [rows().assign(dataset=d, seed=s) for d in DATASETS for s in (72, 73, 74)],
        ignore_index=True,
    )


def test_aggregate_requires_every_dataset_seed_method_and_budget():
    frame = full_matrix()
    args = dict(
        seeds=[72, 73, 74],
        methods=["Tri-Fair-v8.6", "NSGAII-PO-Fair"],
        model="qwen-3-30b",
        split="validation",
        budget=1000,
    )
    runs = validate_matrix(frame, **args)
    assert len(runs) == 24
    for broken in [
        frame[frame.seed != 74],
        frame.assign(selection_policy="current_incumbents"),
        frame.assign(actual_budget_tokens=2000),
    ]:
        with pytest.raises(ValueError):
            validate_matrix(broken, **args)


@shell_test
def test_submission_captures_shared_seeds_and_chains_all_dependencies(tmp_path):
    fake = tmp_path / "sbatch"
    fake.write_text(
        '#!/usr/bin/env bash\nset -eu\nn=100; [[ ! -f "$FAKE_COUNTER" ]] || read -r n < "$FAKE_COUNTER"\nn=$((n+1)); echo "$n" > "$FAKE_COUNTER"\nprintf \'%s|%s|%s\\n\' "$PILOT_SEEDS" "$PILOT_RESULTS_ROOT" "$*" >> "$FAKE_CALLS"\necho "$n"\n'
    )
    fake.chmod(0o755)
    result = bash(
        "bash jobs/submit_pilot_v86.sh",
        PILOT_SEEDS="72,73,74",
        MAX_CONCURRENT="5",
        DRY_RUN="0",
        PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
        FAKE_COUNTER=str(tmp_path / "counter"),
        FAKE_CALLS=str(tmp_path / "calls"),
    )
    assert result.returncode == 0, result.stderr
    calls = (tmp_path / "calls").read_text().splitlines()
    assert len(calls) == 3
    assert all(x.startswith("72,73,74|") for x in calls)
    assert len({x.split("|")[1] for x in calls}) == 1
    assert "--array=0-11%5" in calls[0]
    assert "--dependency=afterok:101" in calls[1]
    assert "--dependency=afterok:102" in calls[2]
    assert "--array" not in calls[2]
