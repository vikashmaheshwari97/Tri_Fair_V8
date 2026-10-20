"""Real cached task tests plus controlled three-objective race trajectories."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from promptolution.utils.prompt import Prompt
from tests.fakes import optimizer
from tri_fair.optimizers.intensified import IntensifiedTriFairOptimizer, dominates
from tri_fair.runtime.budget import BudgetAwarePredictor, TokenBudgetController, BudgetExhausted
from tri_fair.runtime.checkpointing import capture_optimizer_state, restore_optimizer_state


def configured(**kwargs):
    obj = optimizer(IntensifiedTriFairOptimizer, repeat_check_examples=0, **kwargs)
    obj.budget_controller = TokenBudgetController(
        llm=obj.predictor.llm, max_tokens=100000, max_output_tokens=16)
    obj.predictor = BudgetAwarePredictor(obj.predictor, obj.budget_controller)
    obj._pre_optimization_loop()
    obj._race_order = list(range(obj.task.n_blocks))
    return obj


def test_dominance_never_trades_off_an_objective():
    assert dominates([.9, -20, -.1], [.8, -21, -.2])
    assert not dominates([.9, -22, -.1], [.8, -21, -.2])
    assert not dominates([.9, -20, np.nan], [.8, -21, -.2])
    assert not dominates([.9, -20, -.1], [.9, -20, -.1])


def controlled_vectors(obj, monkeypatch, child_vector, ready=True):
    calls = []
    def vectors(prompts, blocks):
        calls.append((list(prompts), list(blocks)))
        current = child_vector(len(blocks)) if callable(child_vector) else child_vector
        return np.array([[.95, -10, -.1]] * (len(prompts)-1) + [current]), np.array([ready]*len(prompts))
    monkeypatch.setattr(obj, '_shared_vectors', vectors)
    return calls


def test_rejection_on_shared_pooled_blocks_after_minimum(monkeypatch):
    obj = configured()
    calls = controlled_vectors(obj, monkeypatch, [.7, -20, -.3])
    assert not obj._intensify(Prompt('inferior child'))
    assert [c[1] for c in calls] == [[0], [0, 1]]
    assert all(c[0][:-1] == obj.incumbents for c in calls)
    assert obj.race_counts['early_rejections'] == 1
    assert obj.race_counts['unevaluated_blocks_after_rejection'] == 6


@pytest.mark.parametrize('ready,vector', [(False, [.7, -20, -.3]), (True, [.99, -20, -.3])])
def test_no_rejection_without_support_or_with_quality_tradeoff(monkeypatch, ready, vector):
    obj = configured()
    calls = controlled_vectors(obj, monkeypatch, vector, ready)
    assert obj._intensify(Prompt('tradeoff child'))
    assert len(calls) == obj.task.n_blocks
    assert obj.race_counts['early_rejections'] == 0


def test_comparison_only_when_objective_vector_improves(monkeypatch):
    obj = configured()
    # At 2 blocks child trades off quality. Later quality drops, so comparison
    # is deferred until improvement over the previous comparison or completion.
    controlled_vectors(obj, monkeypatch, lambda n: [.99 if n <= 2 else .7, -20, -.3])
    assert obj._intensify(Prompt('trajectory child'))
    assert obj.race_counts['comparisons'] == 1


def test_cache_reuse_and_pooled_metrics():
    obj = configured()
    child = Prompt('new task instruction')
    prompts = obj.incumbents + [child]
    vectors, ready = obj._shared_vectors(prompts, [0, 1])
    count = obj.budget_controller.consumed_tokens
    again, again_ready = obj._shared_vectors(prompts, [0, 1])
    assert obj.budget_controller.consumed_tokens == count
    np.testing.assert_allclose(vectors, again)
    np.testing.assert_array_equal(ready, again_ready)
    assert obj.task.get_evaluated_blocks([child])[child] == [0, 1]
    pooled = obj.task.objectives_from_frame(obj.task.cached_prediction_frame(child, [0,1]),
        obj.cost_per_input_token, obj.cost_per_output_token)
    np.testing.assert_allclose(vectors[-1], pooled)


def test_real_racing_step_and_checkpoint_restore(monkeypatch):
    obj = configured(intensification_margins=(0, 0, 0))
    obj.optimize(3)
    assert obj.budget_controller.consumed_tokens <= 100000
    assert obj.race_counts['challenger_blocks'] > 0
    for prompt in obj.prompts + obj.candidate_archive:
        assert set(obj.task.get_evaluated_blocks([prompt])[prompt]) == set(range(obj.task.n_blocks))
    import pickle
    state = pickle.loads(pickle.dumps(capture_optimizer_state(obj)))
    restored = configured(intensification_margins=(0, 0, 0))
    restore_optimizer_state(restored, state)
    assert restored.race_round == obj.race_round
    assert restored.race_counts == obj.race_counts
    # Both the proposal stream and randomized block schedule must resume.
    obj._step()
    expected = [p.construct_prompt() for p in obj.prompts]
    restore_optimizer_state(restored, state)
    restored._step()
    assert [p.construct_prompt() for p in restored.prompts] == expected
    assert restored._race_order == obj._race_order


def test_partial_rejection_does_not_enter_full_archive(monkeypatch):
    obj = configured()
    original = [p.construct_prompt() for p in obj.prompts]
    controlled_vectors(obj, monkeypatch, [.7, -20, -.3])
    obj._step()
    assert [p.construct_prompt() for p in obj.prompts] == original
    assert obj.race_counts['early_rejections'] > 0
    assert obj.strategy_updates > 0
    assert not obj.v89_outcomes
    assert all(not r.get('full_search_observation', True) for r in obj.race_trace
               if r['status'] == 'strategy_rejection_credit')


def test_budget_interrupt_preserves_measured_population():
    obj = configured()
    before = [p.construct_prompt() for p in obj.prompts]
    controller = obj.budget_controller
    controller.max_tokens = controller.consumed_tokens + 1
    with pytest.raises(BudgetExhausted):
        obj._intensify(Prompt('too expensive to finish'))
    assert controller.consumed_tokens <= controller.max_tokens
    assert [p.construct_prompt() for p in obj.prompts] == before
    assert obj.race_counts['budget_interrupted_challengers'] == 1


def test_first_block_admission_not_full_candidate_reservation():
    obj = configured()
    child = Prompt('a candidate')
    partial = obj._estimate_complete(child)
    obj.enable_intensification = False
    complete = obj._estimate_complete(child)
    assert 0 < partial < complete


def test_checkpoint_rejects_changed_racing_protocol():
    obj = configured()
    other = configured(enable_intensification=False)
    with pytest.raises(ValueError, match='Intensification'):
        other.load_optimizer_state_dict(obj.optimizer_state_dict())


@pytest.mark.parametrize('method', ['Tri-Fair-v9.1', 'Tri-Fair-v9.1-FullEval'])
def test_public_runner(method, tmp_path, monkeypatch):
    from tests.test_integration import test_complete_public_runner_with_fake_models
    test_complete_public_runner_with_fake_models(method, tmp_path, monkeypatch)


@pytest.mark.parametrize('profile,runs', [('ablation', 12), ('screen',24)])
def test_frozen_launch_plan(monkeypatch, profile, runs):
    monkeypatch.setenv('V91_PROFILE', profile)
    monkeypatch.setenv('PILOT_SEEDS','72')
    spec = importlib.util.spec_from_file_location('v91_plan', Path(__file__).resolve().parents[1]/'research/run_pilot_v91.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    plan = mod.plan()
    assert plan['optimizer_runs'] == runs
    assert plan['per_method_cap'] == 2000000
    assert 'Tri-Fair-v9.1' in plan['methods']
    if profile == 'ablation':
        assert 'Tri-Fair-v9.1-FullEval' in plan['methods']

@pytest.mark.parametrize('method', ['Tri-Fair-v9.1', 'Tri-Fair-v9.1-FullEval'])
def test_public_runner_with_frozen_final_audit(method, tmp_path, monkeypatch):
    from tests.test_integration import test_complete_public_runner_with_fake_models
    test_complete_public_runner_with_fake_models(method, tmp_path, monkeypatch, audit=True)


def test_unaffordable_next_block_has_correct_stopping_reason(monkeypatch):
    obj = configured()
    ctrl = obj.budget_controller
    ctrl.max_tokens = ctrl.consumed_tokens + 1
    monkeypatch.setattr(obj, '_generate_challengers', lambda: [Prompt('unaffordable new child')])
    with pytest.raises(BudgetExhausted, match='next_shared_block_exceeds_budget'):
        obj._filled_challengers()
    assert ctrl.stopping_reason == 'next_shared_block_exceeds_budget'
    assert obj.fill_trace[-1]['admission_scope'] == 'next_missing_shared_block'
