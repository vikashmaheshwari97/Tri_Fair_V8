"""Exercise public runners, strict budgeting, checkpointing and split labelling."""

import json
import random
import sys
import numpy as np
import pandas as pd
import pytest
from tri_fair.runtime.checkpointing import save_checkpoint
from tri_fair.optimizers.trifair import TriFairOptimizer
from tri_fair.optimizers.refined import RefinedTriFairOptimizer
from tri_fair.optimizers.baselines import SPEA2POFair, NSGAIIIPOFair
from tests.fakes import optimizer, task_and_pool, FakeLLM, FakePredictor


def signature(obj):
    return (
        [p.construct_prompt() for p in obj.prompts],
        [p.construct_prompt() for p in obj.incumbents],
        np.asarray(obj.scores).tolist(),
        obj.predictor.llm.get_token_count(),
        obj.meta_llm.get_token_count(),
        getattr(obj, "operator_history", None),
    )


@pytest.mark.parametrize("cls", [TriFairOptimizer, RefinedTriFairOptimizer, SPEA2POFair, NSGAIIIPOFair])
def test_interrupted_equals_uninterrupted(cls, tmp_path):
    kwargs = (
        {"smart_start_count": 0, "uncertainty_resamples": 8}
        if issubclass(cls, TriFairOptimizer)
        else {}
    )
    random.seed(52)
    np.random.seed(52)
    full = optimizer(cls, **kwargs)
    full.optimize(3)
    expected = signature(full)
    random.seed(52)
    np.random.seed(52)
    first = optimizer(cls, **kwargs)
    first.optimize(1)
    path = tmp_path / "checkpoint.pkl"
    save_checkpoint(first, path)
    restored = optimizer(cls, **kwargs)
    restored.load_checkpoint(path)
    restored.optimize(3)
    assert signature(restored) == expected


@pytest.mark.parametrize(
    "method",
    [
        "Tri-Fair-v8",
        "Tri-Fair-v8.2-Experimental",
        "Tri-Fair-v8-NoFeedback",
        "Tri-Fair-v8-NoDemo",
        "Tri-Fair-v8-NoUncertainty",
        "Tri-Fair-v8-FullEval",
        "NSGAII-PO-Fair",
        "SPEA2-PO-Fair",
        "NSGAIII-PO-Fair",
        "RandomSearch-PO-Fair",
        "NSGAII-PO-Fair-Enhanced",
    ],
)
def test_complete_public_runner_with_fake_models(method, tmp_path, monkeypatch, audit=False):
    import tri_fair.cli.train as runner
    from tri_fair.cli.common import atomic_write_json

    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    manifest_path = manifest_dir / "bias_in_bios_seed42.json"
    atomic_write_json(manifest_path, {"fixture": True})

    def fake_task(**kwargs):
        task, pool = task_and_pool(eval_strategy=kwargs["eval_strategy"])
        task.manifest_path = str(manifest_path)
        return task, pool

    monkeypatch.setattr(runner, "create_dev_tasks", fake_task)
    monkeypatch.setattr(runner, "create_llm", lambda **kwargs: FakeLLM())

    def fake_predictor(llm, classes):
        p = FakePredictor()
        p.llm = llm
        return p

    monkeypatch.setattr(runner, "MarkerBasedPredictor", fake_predictor)
    argv = [
        "test",
        "--experiment-name",
        "fake",
        "--model",
        "qwen-3-30b",
        "--dataset",
        "bias_in_bios",
        "--optimizer",
        method,
        "--random-seed",
        "42",
        "--budget-per-run",
        "1000000",
        "--study-profile",
        "v8",
        "--max-steps",
        "1",
        "--n-init-prompts",
        "6",
        "--manifest-dir",
        str(manifest_dir),
        "--output-dir",
        str(tmp_path / method),
        "--budget-checkpoints",
        "1000000",
        "--allow-under-budget-exit",
    ]
    if audit:
        argv.extend(["--final-search-candidates", "2", "--final-search-reserve", "0.35"])
    monkeypatch.setattr(sys, "argv", argv)
    args = runner.parse_args()
    result = runner.run(args, logging_dir=tmp_path / method / "logs")
    summary = json.loads((result / "run_summary.json").read_text())
    assert summary["status"] == "complete" and summary["budget_overshoot_tokens"] == 0
    partition = summary["resolved_configuration"]["search_validation_partition"]
    assert set(partition["search_ids"]).isdisjoint(partition["validation_ids"])
    assert (result / "checkpoints/latest.pkl").exists()
    assert (result / "step_results.parquet").exists()
    if audit:
        audit_summary = summary["final_search_audit"]
        assert audit_summary["status"] == "complete"
        assert audit_summary["search_budget_tokens"] == 650000
        assert audit_summary["total_budget_tokens"] == 1000000
        assert not audit_summary["feedback_to_optimizer"]
        assert (result / "final_search_results.parquet").exists()
        from tri_fair.cli.pilot import read_result
        report = read_result(result, method)
        assert report["comparable"] and report["quality_metric"] == "macro_f1"


@pytest.mark.parametrize("method", ["Tri-Fair-v8.2-Experimental", "NSGAII-PO-Fair"])
def test_complete_public_runner_with_reserved_search_audit(method, tmp_path, monkeypatch):
    test_complete_public_runner_with_fake_models(method, tmp_path, monkeypatch, audit=True)


def test_v8_validation_evaluation_labels_and_cache(tmp_path, monkeypatch):
    import tri_fair.cli.evaluate as evaluator
    from tri_fair.cli.common import prompt_id

    task, _ = task_and_pool()
    manifest = tmp_path / "bias_in_bios_seed42.json"
    manifest.write_text("{}")
    task.manifest_path = str(manifest)
    log = tmp_path / "source_log"
    log.mkdir()
    (log / "args.json").write_text(json.dumps({"study_profile": "v8"}))
    # A single prompt shared by two methods is evaluated only once.
    from promptolution.utils.prompt import Prompt

    text = Prompt("Classify occupation using duties.").construct_prompt()
    membership = pd.DataFrame(
        [
            {
                "prompt": text,
                "instruction": "Classify occupation using duties.",
                "few_shots_json": "[]",
                "downstream_template": None,
                "prompt_id": prompt_id(text),
                "optimizer": m,
                "model": "qwen-3-30b",
                "dataset": "bias_in_bios",
                "seed": 42,
                "chosen_step": 1,
                "budget_checkpoint": 100,
                "source_log_path": str(log),
            }
            for m in ["Tri-Fair-v8", "NSGAII-PO-Fair"]
        ]
    )
    monkeypatch.setattr(evaluator, "_logging_dir", lambda *a: log)
    monkeypatch.setattr(
        evaluator,
        "_membership_for_run",
        lambda *a, optimizer, **kw: membership[membership.optimizer == optimizer],
    )
    monkeypatch.setattr(evaluator, "create_llm", lambda **kw: FakeLLM())
    import tri_fair.data.loading as creation

    monkeypatch.setattr(
        creation, "create_dev_tasks", lambda *a, **kw: (task, pd.DataFrame())
    )

    def fake_predictor(llm, classes):
        p = FakePredictor()
        p.llm = llm
        return p

    monkeypatch.setattr(evaluator, "MarkerBasedPredictor", fake_predictor)
    args = type(
        "Args",
        (),
        dict(
            study_root=str(tmp_path),
            study_profile="v8",
            evaluation_split="validation",
            structured_labels=False,
            dataset="bias_in_bios",
            model="qwen-3-30b",
            random_seed=42,
            optimizers="Tri-Fair-v8,NSGAII-PO-Fair",
            minimum_actual_tokens=0,
            maximum_actual_tokens=1000,
            manifest_dir=str(tmp_path),
            max_output_tokens=16,
            output_file=str(tmp_path / "eval.parquet"),
            force=False,
        ),
    )()
    path = evaluator.run(args)
    output = pd.read_parquet(path)
    assert (
        len(output) == 2
        and "validation_quality" in output
        and "test_quality" not in output
    )
    # Same cached output is safe; a changed decoding identity is rejected.
    evaluator.run(args)
    args.max_output_tokens = 17
    with pytest.raises(ValueError, match="fingerprint"):
        evaluator.run(args)


@pytest.mark.parametrize("cls", [TriFairOptimizer, SPEA2POFair, NSGAIIIPOFair])
def test_search_stops_before_exceeding_atomic_budget(cls):
    from tri_fair.runtime.budget import TokenBudgetController, BudgetAwarePredictor

    random.seed(91)
    np.random.seed(91)
    kwargs = (
        {"smart_start_count": 0, "uncertainty_resamples": 8}
        if cls is TriFairOptimizer
        else {}
    )
    obj = optimizer(cls, **kwargs)
    obj.optimize(1)
    llm = obj.predictor.llm
    limit = llm.get_token_count()["total_tokens"] + 100
    controller = TokenBudgetController(llm=llm, max_tokens=limit, max_output_tokens=16)
    obj.budget_controller = controller
    obj.predictor = BudgetAwarePredictor(obj.predictor, controller)
    obj._resume_loaded = True
    obj.optimize(3)
    assert llm.get_token_count()["total_tokens"] <= limit
    assert controller.rejected_operations > 0 or controller.stop_requested


def test_v8_racing_calls_paired_guard_before_dominance_rejection(monkeypatch):
    from promptolution.utils.prompt import Prompt

    obj = optimizer(TriFairOptimizer, smart_start_count=0, uncertainty_resamples=8)
    obj.optimize(1)
    calls = []
    monkeypatch.setattr(obj, "_is_dominated", lambda *a: True)
    monkeypatch.setattr(obj, "_robustly_dominated", lambda *a: True)
    monkeypatch.setattr(obj, "_paired_rejection", lambda *a: calls.append(a) or True)
    challenger = Prompt("New synthetic candidate for paired rejection regression.")
    obj._do_intensification(challenger)
    assert calls and calls[0][0] is challenger
