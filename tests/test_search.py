from collections import Counter
import random
import numpy as np
import pytest
from promptolution.utils.prompt import Prompt
from tri_fair.search.selection import (
    hypervolume,
    truncate_archive,
    spea2_select,
    nondominated,
)
from tri_fair.search.uncertainty import resample_indices, pooled_bootstrap
from tri_fair.data.partitions import split_search_validation
from tri_fair.optimizers.trifair import TriFairOptimizer
from tri_fair.optimizers.baselines import (
    SPEA2POFair,
    NSGAIIIPOFair,
    RandomSearchPOFair,
    ReflectiveNSGAIIPOFair,
)
from tri_fair.runtime.checkpointing import (
    capture_optimizer_state,
    restore_optimizer_state,
)
from tests.fakes import optimizer, task_and_pool, FakePredictor


def test_pooled_center_is_union_not_average_blocks():
    obj = object.__new__(TriFairOptimizer)
    obj._get_block_vectors = lambda p, b: (
        np.asarray([[0.5, -10.0, 0.0]])
        if len(b) == 2
        else np.asarray([[0.5, -10.0, -0.6666667]])
    )
    assert obj._robust_vectors([object()], [0, 1])[0, 2] == 0.0


def test_archive_hv_stage_runs_and_preserves_extremes(monkeypatch):
    import tri_fair.search.selection as selection

    calls = []
    original = selection.hypervolume
    monkeypatch.setattr(
        selection, "hypervolume", lambda x: calls.append(len(x)) or original(x)
    )
    values = np.column_stack(
        [
            np.linspace(0.1, 0.95, 40),
            np.linspace(0.95, 0.1, 40),
            np.random.default_rng(1).uniform(0.1, 0.95, 40),
        ]
    )
    kept = truncate_archive(values, 12, np.eye(3))
    assert len(set(kept)) == 12 and calls
    assert all(int(np.argmax(values[:, j])) in kept for j in range(3))


def test_hypervolume_known_empty_and_dominated():
    assert hypervolume(np.empty((0, 3))) == 0.0
    assert hypervolume(np.asarray([[0.5, 0.5, 0.5]]), [1, 1, 1]) == pytest.approx(0.125)
    assert hypervolume(
        np.asarray([[0.5, 0.5, 0.5], [0.8, 0.8, 0.8]]), [1, 1, 1]
    ) == pytest.approx(0.125)


def test_adaptive_scheduler_uses_successes_for_single_slot():
    obj = object.__new__(TriFairOptimizer)
    obj.mode_attempts = Counter(
        {m: 100 for m in ["quality", "fairness", "cost", "balanced", "explore"]}
    )
    obj.mode_successes = Counter({"fairness": 100})
    obj._front_deficit_weights = lambda: {m: 0.2 for m in obj.mode_attempts}
    obj.current_step = 1
    obj.exploration_interval = 4
    obj.operator_ucb_scale = 0.18
    assert obj.allocate_modes(1) == ["fairness"]


def test_pair_resampling_is_reproducible_and_preserves_bios_cells():
    task, _ = task_and_pool()
    predictor = FakePredictor()
    p = Prompt("instructions")
    task.evaluate([p], predictor, block_idx=[0, 1])
    frame = task.cached_prediction_frame(p, [0, 1])
    a = resample_indices(frame, metric=task.fairness_metric_name, seed=42, repeats=8)
    b = resample_indices(frame, metric=task.fairness_metric_name, seed=42, repeats=8)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    for idx in a:
        assert len(frame.iloc[idx].groupby(["_review_target", "gender"])) == 4
    before = predictor.llm.get_token_count()
    draws = pooled_bootstrap(
        task, [p, p], [0, 1], cost_in=0.11, cost_out=0.41, seed=42, repeats=8
    )
    assert np.array_equal(draws[:, 0], draws[:, 1]) and draws.shape == (8, 2, 3)
    assert before == predictor.llm.get_token_count()


def test_validation_split_disjoint_and_same_for_every_method():
    t, _ = task_and_pool()
    s, v = split_search_validation(t)
    t2, _ = task_and_pool(eval_strategy="full")
    s2, v2 = split_search_validation(t2)
    assert set(s.ids).isdisjoint(v.ids)
    assert set(s.ids) | set(v.ids) == set(t.ids)
    assert s.ids == s2.ids and v.ids == v2.ids


def test_adaptive_checkpoint_roundtrip_and_wrong_method_rejected():
    first = optimizer(TriFairOptimizer, smart_start_count=0, uncertainty_resamples=8)
    first.mode_attempts["fairness"] = 9
    first.mode_successes["fairness"] = 4
    first.pending_rewards = {"candidate": "fairness"}
    first._archive_stagnation_steps = 7
    state = capture_optimizer_state(first)
    restored = optimizer(TriFairOptimizer, smart_start_count=0, uncertainty_resamples=8)
    restore_optimizer_state(restored, state)
    assert restored.mode_attempts == first.mode_attempts
    assert restored.pending_rewards == first.pending_rewards
    assert restored._archive_stagnation_steps == 7
    bad = restored.optimizer_state_dict()
    bad["method_version"] = "wrong"
    with pytest.raises(ValueError):
        restored.load_optimizer_state_dict(bad)


@pytest.mark.parametrize(
    "cls",
    [
        TriFairOptimizer,
        SPEA2POFair,
        NSGAIIIPOFair,
        RandomSearchPOFair,
        ReflectiveNSGAIIPOFair,
    ],
)
def test_real_optimizer_two_steps_with_fake_models(cls):
    random.seed(42)
    np.random.seed(42)
    kwargs = (
        {"smart_start_count": 0, "uncertainty_resamples": 8}
        if cls is TriFairOptimizer
        else {}
    )
    obj = optimizer(cls, **kwargs)
    obj.optimize(n_steps=2)
    assert obj.prompts and len(obj.prompts) <= obj.population_size
    assert len(obj.scores) == len(obj.prompts)
    assert np.isfinite(np.asarray(obj.scores)).all()
    for p in obj.incumbents:
        assert obj.get_fairness_record(p)["fairness_ready"]


def test_spea2_keeps_tradeoffs_and_removes_dominated():
    values = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, -1.0, -1.0]]
    )
    assert set(spea2_select(values, 3)) == {0, 1, 2}
    assert nondominated(values).tolist() == [True, True, True, False]


def test_checkpoint_rejects_changed_data_and_ablation():
    first = optimizer(TriFairOptimizer, smart_start_count=0, uncertainty_resamples=8)
    state = capture_optimizer_state(first)
    changed = optimizer(TriFairOptimizer, smart_start_count=0, uncertainty_resamples=8)
    changed.task.df.loc[0, "input"] = "changed text with same example ID"
    with pytest.raises(ValueError, match="data or metric"):
        restore_optimizer_state(changed, state)
    changed = optimizer(
        TriFairOptimizer,
        smart_start_count=0,
        uncertainty_resamples=8,
        enable_feedback=False,
    )
    with pytest.raises(ValueError, match="configuration"):
        restore_optimizer_state(changed, state)


def test_demo_insert_renders_from_zero_shot_template(monkeypatch):
    import tri_fair.search.feedback as feedback

    obj = optimizer(TriFairOptimizer, smart_start_count=0, uncertainty_resamples=8)
    anchor = obj.prompts[0]
    obj.task.evaluate([anchor], obj.predictor, block_idx=[0, 1])
    obj.incumbents = [anchor]
    monkeypatch.setattr(
        feedback,
        "generate_portfolio_challengers",
        lambda x: [Prompt("Instruction without demonstrations.")],
    )
    monkeypatch.setattr(feedback, "_anchor_for_mode", lambda *a: (anchor, None))
    monkeypatch.setattr(feedback.random, "choice", lambda options: "insert")
    children = feedback.generate_reflective_challengers(obj)
    assert children and children[0].few_shots
    assert "<few_shots>" in children[0].downstream_template
    assert str(children[0].few_shots[0]) in children[0].construct_prompt()


def test_paired_rejection_accepts_only_jointly_positive_evidence():
    obj = optimizer(TriFairOptimizer, smart_start_count=0, uncertainty_resamples=8)
    challenger, incumbent = object(), object()
    obj._bootstrap = lambda *a: np.asarray(
        [[[0.5, -10.0, -0.1], [0.6, -9.0, -0.05]]] * 8
    )
    assert obj._paired_rejection(challenger, incumbent, [0, 1])
    obj._bootstrap = lambda *a: np.asarray(
        [[[0.5, -10.0, -0.1], [0.6, -11.0, -0.05]]] * 8
    )
    assert not obj._paired_rejection(challenger, incumbent, [0, 1])


@pytest.mark.parametrize(
    "dataset,expected_search,expected_validation",
    [("bbq", 264, 176), ("civil_comments", 384, 192), ("bias_in_bios", 448, 336)],
)
def test_profile_partition_has_worst_case_wilson_support(
    dataset, expected_search, expected_validation
):
    import pandas as pd
    from tri_fair.config.datasets import ALL_DATASETS
    from tri_fair.config.study import apply_study_profile
    from tri_fair.data.task import FairnessTask
    from tri_fair.data.partitions import _supports_all_outcomes

    config = apply_study_profile(ALL_DATASETS[dataset])
    f = config.fairness
    rows = []
    blocks = []
    for block in range(f.dev_size // f.block_size):
        current = []
        if dataset == "bias_in_bios":
            cells = [
                {"target": label, "gender": g, "profession_id": i}
                for i, label in enumerate(f.class_names)
                for g in [0, 1]
            ]
        elif dataset == "bbq":
            cells = [
                {
                    "target": "a",
                    "category": f"cat{i}",
                    "context_condition": ctx,
                    "question_polarity": "neg",
                    "template_id": f"{block}-{i}",
                    "target_loc": 0,
                    "unknown_loc": 2,
                    "label_index": 0,
                }
                for i in range(11)
                for ctx in ["ambig", "disambig"]
                for _ in range(2)
            ]
        else:
            identities = list(f.protected_columns)
            cells = [
                {
                    "target": label,
                    "label_index": int(label == "toxic"),
                    **{k: int(k == identity) for k in identities},
                }
                for identity in identities
                for label in f.class_names
                for _ in range(6)
            ]
        for cell in cells:
            key = f"id-{len(rows)}"
            current.append(key)
            rows.append(dict(cell, _manifest_id=key, input="synthetic input"))
        blocks.append(current)
    assert len(rows) == f.dev_size
    task = FairnessTask(
        df=pd.DataFrame(rows),
        task_description=config.task_description,
        classes=f.class_names,
        fairness_metric=lambda *a, **kw: None,
        fairness_metric_name=f.metric_name,
        quality_metric=f.quality_metric,
        protected_columns=f.protected_columns,
        metadata_columns=f.metadata_columns,
        min_group_count=f.min_group_count,
        fairness_kwargs=f.fairness_kwargs,
        block_ids=blocks,
        seed=72,
    )
    search, validation = split_search_validation(task)
    assert (
        len(search.df) == expected_search and len(validation.df) == expected_validation
    )
    assert _supports_all_outcomes(search, list(range(search.n_blocks)))
    assert _supports_all_outcomes(validation, list(range(validation.n_blocks)))


def test_dominant_two_parent_tournament_cannot_loop_forever(monkeypatch):
    from tri_fair.optimizers.nsga2_core import NSGA2PromptOptimizer

    obj = object.__new__(NSGA2PromptOptimizer)
    obj.prompts = [Prompt("first parent"), Prompt("second parent")]
    calls = []
    obj._tournament_select_one = lambda *a: calls.append(1) or obj.prompts[0]
    first, second = obj._tournament_selection()
    assert first != second and len(calls) == 5


def test_bbq_resampling_preserves_category_clusters():
    import pandas as pd

    records = []
    for category, n_units in [("one", 2), ("two", 5)]:
        for unit in range(n_units):
            for example in range(4):
                records.append(
                    {
                        "category": category,
                        "template_id": f"{category}-{unit}",
                        "_review_id": f"{category}-{unit}-{example}",
                    }
                )
    frame = pd.DataFrame(records)
    draws = resample_indices(frame, metric="bbq_bias_statistical", seed=42, repeats=8)
    original = frame.groupby("category").size().to_dict()
    for indices in draws:
        sampled = frame.iloc[indices]
        assert sampled.groupby("category").size().to_dict() == original
        assert all(count % 4 == 0 for count in sampled.groupby("template_id").size())


def test_unready_bootstrap_draw_blocks_rejection():
    obj = optimizer(TriFairOptimizer, smart_start_count=0, uncertainty_resamples=8)
    prompt = obj.prompts[0]
    obj.task.evaluate([prompt], obj.predictor, block_idx=[0, 1])
    frame = obj.task.cached_prediction_frame(prompt, [0, 1])
    obj.task.min_group_count = 100
    values = obj.task.objectives_from_frame(frame, 0.11, 0.41)
    assert np.isnan(values[2])
    obj._bootstrap = lambda *a: np.full((8, 2, 3), np.nan)
    assert not obj._paired_rejection(object(), object(), [0, 1])
