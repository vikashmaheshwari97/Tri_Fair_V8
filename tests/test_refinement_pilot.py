"""Evidence bounds, real scoring, and conservative pilot comparison rules."""

from argparse import Namespace
import json
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import pytest
from promptolution.utils.prompt import Prompt
from tests.fakes import optimizer
from tri_fair.optimizers.refined import RefinedTriFairOptimizer
from tri_fair.optimizers.trifair import TriFairOptimizer
from tri_fair.config.registry import METHODS
from tri_fair.config.study import build_optimizer_registry, TRI_FAIR_PARAMETERS
from tri_fair.search.refinement import head_tail, diverse_evidence, refinement_request
from tri_fair.cli.pilot import (
    VARIANT,
    train_command,
    read_result,
    compare_results,
    measured_accuracy,
    main,
)


def test_existing_method_configuration_is_preserved():
    configs = build_optimizer_registry()
    assert configs["Tri-Fair-v8"].optimizer_params == TRI_FAIR_PARAMETERS
    assert METHODS["Tri-Fair-v8"] is TriFairOptimizer
    assert METHODS[VARIANT] is RefinedTriFairOptimizer
    for method in (
        "NSGAII-PO-Fair",
        "SPEA2-PO-Fair",
        "NSGAIII-PO-Fair",
        "RandomSearch-PO-Fair",
        "NSGAII-PO-Fair-Enhanced",
    ):
        params = configs[method].optimizer_params
        assert params["upper_shots"] == 4 and params["crossovers_per_iter"] == 6
        assert configs[method].eval_strategy == "full"


@pytest.mark.parametrize("limit", [0, 1, 7, 8, 9, 40, 560])
def test_head_tail_has_strict_bound(limit):
    value = head_tail("BEGIN " + "x" * 1000 + " END", limit)
    assert len(value) <= limit
    if limit > 9:
        assert value.startswith("BEGIN") and value.endswith(" END")


def test_diverse_evidence_uses_cached_errors_across_groups_and_retains_success():
    frame = pd.DataFrame(
        [
            {
                "text": "start " + "x" * 1000 + " teacher",
                "_review_target": "teacher",
                "_review_prediction": "nurse",
                "gender": gender,
            }
            for gender in (0, 0, 0, 1)
        ]
        + [
            {
                "text": "nurse",
                "_review_target": "nurse",
                "_review_prediction": "nurse",
                "gender": 0,
            }
        ]
    )
    task = SimpleNamespace(
        x_column="text",
        protected_columns=["gender"],
        cached_prediction_frame=lambda prompt, blocks: frame,
    )
    rows = diverse_evidence(task, Prompt("instruction"), [0], limit=3)
    assert len(rows) == 3 and rows[-1]["correct"]
    assert {row["protected_context"]["gender"] for row in rows[:2]} == {"0", "1"}
    assert rows[0]["input"].endswith("teacher")
    assert len(json.dumps(rows, ensure_ascii=False)) <= 3200
    assert diverse_evidence(task, None, [], max_chars=10) == []


def test_request_contains_all_labels_and_no_holdout_access():
    task = SimpleNamespace(
        classes=["nurse", "teacher"],
        fairness_metric_name="bios_tpr_gap_statistical",
        task_description="Occupation",
    )
    prompt = Prompt("Occupation from duties.")
    text = refinement_request(task, prompt, prompt, "quality", [])
    assert "nurse, teacher" in text and "SEARCH examples only" in text
    assert "55 words" in text and "<final_answer>" in text


def test_refined_optimizer_measures_challengers_and_accounts_for_tokens():
    obj = optimizer(
        RefinedTriFairOptimizer, smart_start_count=0, uncertainty_resamples=8
    )
    seeds = obj._build_seed_population()
    assert all(not prompt.few_shots for prompt in seeds)
    obj.optimize(2)
    assert obj.predictor.llm.get_token_count()["total_tokens"] > 0
    assert obj.meta_llm.get_token_count()["total_tokens"] > 0
    assert obj.operator_history
    assert all(
        obj.get_fairness_record(prompt)["fairness_ready"] for prompt in obj.incumbents
    )


def make_run(tmp_path, *, full=True):
    summary = {
        "status": "complete",
        "requested_budget": 100000,
        "final_downstream_tokens": {"total_tokens": 99000},
        "resolved_configuration": {
            "search_validation_partition": {"search_original_blocks": [0, 1]},
            "dataset": {
                "fairness": {
                    "quality_metric": "macro_f1",
                    "fairness_kwargs": {"normalization_cost_upper_bound": 120},
                }
            },
        },
    }
    (tmp_path / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    # The partial prompt looks much better; it MUST NOT win representative selection.
    rows = [
        {
            "step": 1,
            "evaluated_on_blocks": "[0, 1]" if full else "[0]",
            "is_incumbent": True,
            "fairness_ready": True,
            "quality": 0.7,
            "cost": 20.0,
            "fairness": 0.2,
            "instruction": "fully measured",
            "few_shots_json": "[]",
        },
        {
            "step": 1,
            "evaluated_on_blocks": "[0]",
            "is_incumbent": True,
            "fairness_ready": True,
            "quality": 1.0,
            "cost": 0.0,
            "fairness": 0.0,
            "instruction": "misleading partial",
            "few_shots_json": "[]",
        },
    ]
    pd.DataFrame(rows).to_parquet(tmp_path / "step_results.parquet")


def test_partial_fidelity_never_counts_as_a_win(tmp_path):
    make_run(tmp_path)
    record = read_result(tmp_path, VARIANT)
    assert record["quality"] == 0.7 and record["instruction"] == "fully measured"
    make_run(tmp_path, full=False)
    assert not read_result(tmp_path, VARIANT)["comparable"]


def test_failed_initialization_is_visible_not_a_zero_score(tmp_path):
    (tmp_path / "run_summary.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "error_type": "BudgetExhausted",
                "error": "cannot fit next block",
            }
        ),
        encoding="utf-8",
    )
    row = read_result(tmp_path, "NSGAII-PO-Fair")
    assert row["status"] == "insufficient_budget" and not row["comparable"]
    assert "quality" not in row
    comparisons = compare_results([dict(method=VARIANT, comparable=True), row])
    assert comparisons[0]["all_three_strictly_better"] is None


def test_all_three_comparison_requires_every_objective_and_equal_coverage(tmp_path):
    make_run(tmp_path)
    variant = read_result(tmp_path, VARIANT)
    control = dict(
        variant, method="NSGAII-PO-Fair", quality=0.6, cost=25.0, unfairness=0.3
    )
    assert compare_results([variant, control])[0]["all_three_strictly_better"]
    control["unfairness"] = 0.1
    assert not compare_results([variant, control])[0]["all_three_strictly_better"]
    control["search_original_blocks"] = [0]
    assert compare_results([variant, control])[0]["all_three_strictly_better"] is None


def test_pilot_defaults_are_identical_across_methods_and_dry_run_has_no_writes(
    tmp_path, capsys
):
    args = Namespace(
        model="qwen-3-30b",
        dataset="bbq",
        seed=72,
        budget=100000,
        manifest_dir=str(tmp_path),
    )
    a = train_command(args, VARIANT, tmp_path)
    b = train_command(args, "NSGAII-PO-Fair", tmp_path)
    assert [v.replace(VARIANT, "METHOD") for v in a] == [
        v.replace("NSGAII-PO-Fair", "METHOD") for v in b
    ]
    assert main(["--dataset", "bbq", "--output-root", str(tmp_path)]) == 0
    assert "DRY RUN" in capsys.readouterr().out
    assert not list(tmp_path.iterdir())


def test_pilot_keeps_all_methods_when_one_exhausts_budget(tmp_path, monkeypatch):
    from tri_fair.cli import pilot

    calls = []

    def fake_run(command, **kwargs):
        method = command[command.index("--optimizer") + 1]
        calls.append(method)
        directory = Path(command[command.index("--output-dir") + 1]) / "run"
        directory.mkdir(parents=True)
        if method == VARIANT:
            make_run(directory)
            return SimpleNamespace(returncode=0)
        (directory / "run_summary.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": "BudgetExhausted",
                    "error": "budget too small",
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(pilot.subprocess, "run", fake_run)
    assert (
        pilot.main(
            [
                "--execute",
                "--dataset",
                "bbq",
                "--output-root",
                str(tmp_path),
                "--methods",
                f"{VARIANT},NSGAII-PO-Fair,SPEA2-PO-Fair",
            ]
        )
        == 0
    )
    assert calls == [VARIANT, "NSGAII-PO-Fair", "SPEA2-PO-Fair"]
    report = json.loads(
        next(tmp_path.rglob("pilot_report.json")).read_text(encoding="utf-8")
    )
    assert report["status"] == "inconclusive" and not report["benchmark_result"]
    assert report["all_three_better_than_every_control"] is None
    assert len(report["runs"]) == 3


def test_bios_accuracy_is_weighted_recall_and_does_not_replace_macro_f1():
    row = {
        "quality": 0.6,
        "fairness_diagnostics_json": json.dumps(
            {"profession_recall": {"nurse": 0.5, "teacher": 1.0}}
        ),
        "group_support_json": json.dumps(
            {"nurse/male": 2, "nurse/female": 4, "teacher/male": 1, "teacher/female": 1}
        ),
    }
    assert measured_accuracy(row, "macro_f1") == 0.625
    assert row["quality"] == 0.6
