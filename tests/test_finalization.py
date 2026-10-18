"""A final audit uses actual predictions, a frozen shortlist and the original cap."""

import json
import pandas as pd
import pytest
from tests.fakes import optimizer
from tri_fair.optimizers.refined import RefinedTriFairOptimizer
from tri_fair.runtime.budget import BudgetAwarePredictor, TokenBudgetController
from tri_fair.runtime.callbacks import ExperimentCallback
from tri_fair.runtime.finalization import finalize_search_front
from tri_fair.cli.pilot import read_result


def initialized(tmp_path):
    obj = optimizer(
        RefinedTriFairOptimizer, smart_start_count=0, uncertainty_resamples=8
    )
    controller = TokenBudgetController(
        llm=obj.predictor.llm, max_tokens=650000, max_output_tokens=16
    )
    obj.predictor = BudgetAwarePredictor(obj.predictor, controller)
    obj.budget_controller = controller
    obj._pre_optimization_loop()
    ExperimentCallback(str(tmp_path)).on_step_end(obj)
    return obj


def test_audit_adds_real_predictions_without_changing_optimizer_selection(tmp_path):
    obj = initialized(tmp_path)
    prompts = [p.construct_prompt() for p in obj.incumbents]
    scores = [list(s) for s in obj.scores]
    before = obj.budget_controller.consumed_tokens
    meta_before = obj.meta_llm.get_token_count()
    summary = finalize_search_front(obj, tmp_path, count=2, total_budget=1000000)
    frame = pd.read_parquet(tmp_path / "final_search_results.parquet")
    assert summary["status"] == "complete"
    assert obj.budget_controller.consumed_tokens > before
    assert obj.budget_controller.consumed_tokens <= 1000000
    assert (
        summary["search_downstream_tokens"] + summary["audit_downstream_tokens"]
        == summary["total_downstream_tokens"]
    )
    assert [p.construct_prompt() for p in obj.incumbents] == prompts and [
        list(s) for s in obj.scores
    ] == scores
    assert obj.meta_llm.get_token_count() == meta_before
    assert frame["fairness_ready"].all()
    assert all(
        set(json.loads(x)) == set(range(obj.task.n_blocks))
        for x in frame.evaluated_on_blocks
    )
    for _, row in frame.iterrows():
        from promptolution.utils.prompt import Prompt

        prompt = Prompt(row.instruction, json.loads(row.few_shots_json))
        result = obj.task.evaluate([prompt], obj.predictor, eval_strategy="full")
        assert row.quality == pytest.approx(result.agg_scores[0])
        assert row.fairness == pytest.approx(result.fairness_loss[0])
        assert row.cost == pytest.approx(
            obj.cost_per_input_token * result.agg_input_tokens[0]
            + obj.cost_per_output_token * result.agg_output_tokens[0]
        )


def test_audit_cannot_overspend_or_claim_missing_predictions(tmp_path):
    obj = initialized(tmp_path)
    controller = obj.budget_controller
    consumed = controller.consumed_tokens
    controller.max_tokens = consumed
    summary = finalize_search_front(obj, tmp_path, count=2, total_budget=consumed)
    assert summary["status"] == "incomplete" and summary["errors"]
    assert controller.consumed_tokens == consumed
    assert not (tmp_path / "final_search_results.parquet").exists()


def test_audit_cannot_lower_cap(tmp_path):
    obj = initialized(tmp_path)
    with pytest.raises(ValueError):
        finalize_search_front(obj, tmp_path, count=2, total_budget=500000)


def test_reader_cannot_ignore_an_incomplete_audit(tmp_path):
    summary = {
        "status": "complete",
        "requested_budget": 1000000,
        "final_downstream_tokens": {"total_tokens": 650000},
        "final_search_audit": {"status": "incomplete"},
        "resolved_configuration": {
            "search_validation_partition": {"search_original_blocks": [0, 1]},
            "dataset": {},
        },
    }
    (tmp_path / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    row = read_result(tmp_path, "Tri-Fair-v8.2-Experimental")
    assert not row["comparable"] and row["reason"] == "incomplete_final_search_audit"
