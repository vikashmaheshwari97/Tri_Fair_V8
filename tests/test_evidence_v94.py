"""Evidence visibility, mixed-fidelity pruning, strict caps and reproducibility."""

import importlib.util
import json
import pickle
import random
from pathlib import Path

import numpy as np
import pytest

from tests.fakes import optimizer
from tri_fair.optimizers.evidence_racing import EvidenceRacingTriFairOptimizer
from tri_fair.optimizers.plain_baselines import PLAIN_BASELINES
from tri_fair.runtime.budget import (
    BudgetAwarePredictor,
    BudgetExhausted,
    TokenBudgetController,
)
from tri_fair.runtime.checkpointing import (
    capture_optimizer_state,
    restore_optimizer_state,
)
from tri_fair.search.evidence_v94 import cached_evidence, edit_request


def make(**kwargs):
    random.seed(72)
    np.random.seed(72)
    return optimizer(EvidenceRacingTriFairOptimizer, **kwargs)


def test_cached_evidence_requires_only_parent_observed_blocks():
    obj = make()
    obj._pre_optimization_loop()
    parent = obj.prompts[0]
    blocks = obj._blocks(parent)
    before = obj.predictor.llm.get_token_count()
    evidence = cached_evidence(obj, parent, blocks)
    assert evidence["split"] == "search" and evidence["blocks"] == blocks
    assert evidence["n_observed"] == sum(len(obj.task.blocks[b]) for b in blocks)
    assert len(evidence["cases"]) <= 4
    assert before == obj.predictor.llm.get_token_count()
    assert all(c["search_id"] in obj.task.ids for c in evidence["cases"])


def test_every_initial_demonstration_is_rendered_and_charged():
    obj = make()
    obj._pre_optimization_loop()
    assert any(p.few_shots for p in obj.prompts)
    for parent in obj.prompts:
        assert all(d in parent.construct_prompt() for d in parent.few_shots)
        if parent.few_shots:
            from tri_fair.search.rendering import rendered_prompt

            bare = rendered_prompt(
                parent.instruction, [], template=parent.downstream_template
            )
            assert obj.token_counter(parent.construct_prompt()) > obj.token_counter(
                bare.construct_prompt()
            )


def test_historical_v93_rendering_is_preserved_but_new_initializer_repairs_it():
    from promptolution.utils.prompt import Prompt

    from tri_fair.optimizers.support_ready import SupportReadyTriFairOptimizer

    old = optimizer(SupportReadyTriFairOptimizer)
    new = make()
    p = Prompt("Choose a class.")
    demo = "Input: a synthetic case. Output: teacher"
    assert demo not in old._build_initial_prompt(p, [demo]).construct_prompt()
    assert demo in new._build_initial_prompt(p, [demo]).construct_prompt()


def test_evidence_does_not_silently_fill_missing_search_predictions():
    obj = make()
    obj._pre_optimization_loop()
    with pytest.raises(KeyError, match="uncached evidence"):
        cached_evidence(obj, obj.prompts[0], range(obj.task.n_blocks))


def test_scoped_request_is_bounded_and_preserves_label_space():
    obj = make()
    obj._pre_optimization_loop()
    p, q = obj.prompts[:2]
    ev = cached_evidence(obj, p, obj._blocks(p))
    request = edit_request(obj, p, q, "fairness", ev)
    assert obj.meta_token_counter(request) <= obj.generation_input_token_limit
    assert all(label in request for label in obj.task.classes)
    assert "not determine the prediction" in request
    assert "same-label controls" in request


def test_evidence_generation_records_novel_candidates():
    obj = make()
    obj._pre_optimization_loop()
    children = obj._generate_challengers()
    assert len(children) == obj.crossovers_per_iter
    assert len({p.construct_prompt() for p in children}) == len(children)
    assert obj.proposal_records
    assert all(
        obj.token_counter(p.instruction) <= obj.instruction_token_limit
        for p in children
    )


def test_duplicate_candidate_is_detected_before_inference():
    obj = make()
    obj._pre_optimization_loop()
    assert not obj._novel(obj.prompts[0])
    child = obj._generate_challengers()[0]
    assert not obj._novel(child)


def test_invalid_meta_responses_have_bounded_retry_cost(monkeypatch):
    obj = make(proposal_retries=3)
    obj._pre_optimization_loop()
    monkeypatch.setattr(obj.meta_llm, "get_response", lambda x: ["invalid"])
    obj._generate_challengers()
    assert obj.algorithm_counts["proposal_attempts"] == 3
    assert obj.algorithm_counts["invalid_meta_responses"] <= 3


def test_compression_removes_complete_demo_unit():
    obj = make()
    obj._pre_optimization_loop()
    p = next(p for p in obj.prompts if p.few_shots)
    obj._tournament_selection = lambda *args: (p, obj.prompts[-1])
    obj.crossovers_per_iter = 1
    obj.proposal_round = 2
    child = obj._generate_challengers()[0]
    assert len(child.few_shots) == len(p.few_shots) - 1
    assert obj.token_counter(child.construct_prompt()) < obj.token_counter(
        p.construct_prompt()
    )


@pytest.mark.parametrize("n", [1, 2])
def test_stability_never_claims_supported_ordering_with_too_few_blocks(n):
    obj = make()
    assert obj._stable_dominance(None, None, list(range(n))) is False


def test_unsupported_delete_one_estimate_defers_early_rejection(monkeypatch):
    obj = make()
    monkeypatch.setattr(obj, "_supported", lambda *a: False)
    assert not obj._stable_dominance(None, None, [0, 1, 2])


def test_stable_point_ordering_is_required_on_every_supported_subset(monkeypatch):
    obj = make()
    monkeypatch.setattr(obj, "_supported", lambda *a: True)
    monkeypatch.setattr(
        obj,
        "_get_block_vectors",
        lambda *a: np.array([[0.9, -10, -0.1], [0.8, -11, -0.2]]),
    )
    assert obj._stable_dominance("a", "b", [0, 1, 2])
    monkeypatch.setattr(
        obj,
        "_get_block_vectors",
        lambda ps, bs: np.array(
            [[0.9, -10, -0.1], [0.95 if 0 not in bs else 0.8, -11, -0.2]]
        ),
    )
    assert not obj._stable_dominance("a", "b", [0, 1, 2])


def test_stability_ablation_is_explicit():
    obj = make(stability_guard=False)
    assert obj._stable_dominance(None, None, [0]) is True
    assert not obj.search_diagnostics()["conservative_early_pruning"]


def test_intensification_keeps_common_basis_not_full_search():
    obj = make()
    obj.optimize(1)
    assert obj.search_diagnostics()["mixed_fidelity_population"]
    assert obj.search_diagnostics()["incumbent_advancement"]
    assert any(len(obj._blocks(p)) < obj.task.n_blocks for p in obj.prompts)
    assert obj.edit_outcomes
    assert not obj.search_diagnostics()["holdout_feedback_used"]


def test_shared_hard_budget_also_applies_to_new_method():
    obj = make()
    obj.budget_controller = TokenBudgetController(
        llm=obj.predictor.llm, max_tokens=4000, max_output_tokens=16
    )
    obj.predictor = BudgetAwarePredictor(obj.predictor, obj.budget_controller)
    try:
        obj.optimize(50)
    except BudgetExhausted:
        pass
    assert obj.budget_controller.consumed_tokens <= 4000


def test_resume_preserves_evidence_and_random_stream():
    obj = make()
    obj.optimize(2)
    state = pickle.loads(pickle.dumps(capture_optimizer_state(obj)))
    obj._step()
    expected = [p.construct_prompt() for p in obj.prompts]
    expected_outcomes = list(obj.edit_outcomes)
    other = make()
    restore_optimizer_state(other, state)
    other._step()
    assert [p.construct_prompt() for p in other.prompts] == expected
    assert other.edit_outcomes == expected_outcomes
    assert other.proposal_round == obj.proposal_round


def test_resume_rejects_changed_experimental_configuration():
    obj = make()
    other = make(stability_guard=False)
    with pytest.raises(ValueError, match="Evidence-racing configuration"):
        other.load_optimizer_state_dict(obj.optimizer_state_dict())


@pytest.mark.parametrize(
    "method", ["Tri-Fair-v9.4", "Tri-Fair-v9.4-NoEvidence", "Tri-Fair-v9.4-NoStability"]
)
def test_public_runner_freezes_audit_with_new_method(method, tmp_path, monkeypatch):
    from tests.test_integration import test_complete_public_runner_with_fake_models

    test_complete_public_runner_with_fake_models(
        method, tmp_path, monkeypatch, audit=True
    )


@pytest.mark.parametrize("profile,runs", [("screen", 20), ("ablation", 8)])
def test_plan_is_explicit_and_shared_reserve_applies_to_all_methods(
    monkeypatch, profile, runs
):
    monkeypatch.setenv("V94_PROFILE", profile)
    monkeypatch.setenv("PILOT_SEEDS", "72")
    p = Path(__file__).resolve().parents[1] / "research/run_pilot_v94.py"
    spec = importlib.util.spec_from_file_location("plan94", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    plan = mod.plan()
    assert plan["optimizer_runs"] == runs
    assert plan["pilot_settings"]["final_search_reserve"] == 0.20
    assert not plan["fairness_thresholds_changed"]
    assert plan["baseline_implementations_unchanged"]
    if profile == "screen":
        assert set(PLAIN_BASELINES).issubset(plan["methods"])
    else:
        assert plan["methods"] == ("Tri-Fair-v9.4", "Tri-Fair-v9.3")


def test_v94_cannot_feed_audit_results_back_into_search():
    from types import SimpleNamespace

    from tri_fair.cli.train import _validate_args

    with pytest.raises(ValueError, match="frozen terminal audit"):
        _validate_args(
            SimpleNamespace(optimizer="Tri-Fair-v9.4", reclaim_audit_reserve=True)
        )


def test_supplementary_figures_preserve_measurements(tmp_path, monkeypatch):
    from tests.test_research_figures import rows

    research = Path(__file__).resolve().parents[1] / "research"
    monkeypatch.syspath_prepend(str(research))
    import plot_v94

    frame = rows().replace(
        {
            "optimizer": {
                "Tri-Fair-v8.6": "Tri-Fair-v9.4",
                "NSGAII-PO-Fair": "NSGAII-PO-Fair-Plain",
            }
        }
    )
    frame["validation_input_tokens"], frame["validation_output_tokens"] = 50.0, 2.0
    dest = tmp_path / "heldout_validation/qwen-3-30b/winobias/seed72"
    dest.mkdir(parents=True)
    frame.to_parquet(dest / "eval_checkpoints_union.parquet", index=False)
    made = plot_v94.render(tmp_path, tmp_path / "figures", budget=1000)
    assert len(made) == 5
    assert (tmp_path / "figures/representative_tradeoffs.csv").is_file()
    assert (tmp_path / "figures/fairness_components.csv").is_file()
    p = json.loads((tmp_path / "figures/figure_provenance.json").read_text())
    assert p["metric_values_modified"] is False
