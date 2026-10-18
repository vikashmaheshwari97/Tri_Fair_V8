"""Explicit current-study optimizer settings and component ablations."""

from copy import deepcopy
from tri_fair.config.schema import OptimizerConfig

STUDY_VERSION = "8.1-clean-search-validation-test"


def apply_study_profile(config):
    return deepcopy(config)


TRI_FAIR_PARAMETERS = {
    "crossovers_per_iter": 6,
    "upper_shots": 4,
    "check_fs_accuracy": True,
    "create_fs_reasoning": False,
    "objective_aware_variation": True,
    "min_initial_ready_blocks": 2,
    "min_racing_blocks": 3,
    "dominance_epsilons": (0.002, 0.08, 0.002),
    "max_confirmation_streak": 1,
    "quality_guard_scale": 0.06,
    "fairness_guard_scale": 0.08,
    "smart_start_count": 4,
    "archive_cap": 15,
    "verified_parent_blocks": 3,
    "stagnation_window": 3,
    "late_phase_crossovers": 2,
    "critical_phase_crossovers": 1,
    "exploration_interval": 4,
    "reference_partitions": 4,
    "operator_ucb_scale": 0.18,
    "quality_floor_start": 0.055,
    "quality_floor_end": 0.02,
    "parent_pool_cap": 10,
    "uncertainty_resamples": 24,
    "uncertainty_alpha": 0.1,
    "feedback_cases": 4,
    "enable_feedback": True,
    "enable_demo_search": True,
}


def build_optimizer_registry():
    configs = {}
    shared = dict(
        crossovers_per_iter=6,
        upper_shots=4,
        check_fs_accuracy=True,
        create_fs_reasoning=False,
        objective_aware_variation=True,
    )
    for name in [
        "NSGAII-PO-Fair",
        "SPEA2-PO-Fair",
        "NSGAIII-PO-Fair",
        "RandomSearch-PO-Fair",
        "NSGAII-PO-Fair-Enhanced",
    ]:
        configs[name] = OptimizerConfig(name, name, dict(shared), "full", 1)
    configs["RandomSearch-PO-Fair"].optimizer_params["objective_aware_variation"] = (
        False
    )
    variants = {
        "Tri-Fair-v8": {},
        "Tri-Fair-v8-NoFeedback": {"enable_feedback": False},
        "Tri-Fair-v8-NoDemo": {"enable_demo_search": False},
        "Tri-Fair-v8-NoUncertainty": {"uncertainty_resamples": 0},
        "Tri-Fair-v8-FullEval": {"full_evaluation": True},
    }
    for name, overrides in variants.items():
        configs[name] = OptimizerConfig(
            name, name, dict(TRI_FAIR_PARAMETERS, **overrides), "sequential_block", 1
        )
    configs["Tri-Fair-v8.2-Experimental"] = OptimizerConfig(
        "Tri-Fair-v8.2-Experimental",
        "Tri-Fair-v8.2-Experimental",
        dict(TRI_FAIR_PARAMETERS, smart_start_count=0, feedback_cases=6),
        "sequential_block",
        1,
    )
    configs["Tri-Fair-v8.3"] = OptimizerConfig(
        "Tri-Fair-v8.3",
        "Tri-Fair-v8.3",
        dict(
            TRI_FAIR_PARAMETERS,
            smart_start_count=0,
            feedback_cases=5,
            max_generation_calls=128,
            max_idle_rounds=3,
            prompt_word_limit=96,
            generation_input_token_limit=2100,
        ),
        "sequential_block",
        1,
    )
    configs["Tri-Fair-v8.4-Hybrid"] = OptimizerConfig(
        "Tri-Fair-v8.4-Hybrid",
        "Tri-Fair-v8.4-Hybrid",
        dict(
            shared,
            max_generation_calls=128,
            prompt_word_limit=96,
            generation_input_token_limit=2100,
        ),
        "full",
        1,
    )
    # Hybrid is retained for historical checkpoint/evaluation compatibility.
    configs["Tri-Fair-v8.5"] = OptimizerConfig(
        "Tri-Fair-v8.5",
        "Tri-Fair-v8.5",
        dict(
            TRI_FAIR_PARAMETERS,
            smart_start_count=0,
            feedback_cases=5,
            min_initial_ready_blocks=3,
            max_generation_calls=128,
            max_idle_rounds=3,
            prompt_word_limit=96,
            generation_input_token_limit=2100,
            stability_strength=1.0,
            parent_quality_margin=0.04,
        ),
        "sequential_block",
        1,
    )
    return configs
