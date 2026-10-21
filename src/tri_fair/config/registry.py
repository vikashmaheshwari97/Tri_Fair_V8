"""Explicit factory and effective metadata for the upgraded study."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import asdict
from pathlib import Path

from tri_fair.optimizers.adaptive import AdaptiveTriFairOptimizer
from tri_fair.optimizers.attributed import AttributedTriFairOptimizer
from tri_fair.optimizers.baselines import (
    NSGAIIIPOFair,
    RandomSearchPOFair,
    ReflectiveNSGAIIPOFair,
    SPEA2POFair,
)
from tri_fair.optimizers.budget_fill import FILLED_BASELINES
from tri_fair.optimizers.directional import DirectionalTriFairOptimizer
from tri_fair.optimizers.efficient import EfficientTriFairOptimizer
from tri_fair.optimizers.evidence_racing import EvidenceRacingTriFairOptimizer
from tri_fair.optimizers.guarded import GuardedTriFairOptimizer
from tri_fair.optimizers.hybrid import HybridTriFairOptimizer
from tri_fair.optimizers.intensified import IntensifiedTriFairOptimizer
from tri_fair.optimizers.mocapo_fair import MOCAPOFairOptimizer
from tri_fair.optimizers.nsga2 import NSGA2Fair
from tri_fair.optimizers.plain_baselines import PLAIN_BASELINES, PlainVariationProtocol
from tri_fair.optimizers.refined import RefinedTriFairOptimizer
from tri_fair.optimizers.repaired import RepairedTriFairOptimizer
from tri_fair.optimizers.stable import StableTriFairOptimizer
from tri_fair.optimizers.support_ready import SupportReadyTriFairOptimizer
from tri_fair.optimizers.trifair import TriFairOptimizer

METHODS = {
    "Tri-Fair-v9.4": EvidenceRacingTriFairOptimizer,
    "Tri-Fair-v9.4-NoEvidence": EvidenceRacingTriFairOptimizer,
    "Tri-Fair-v9.4-NoStability": EvidenceRacingTriFairOptimizer,
    **PLAIN_BASELINES,
    "Tri-Fair-v9.3": SupportReadyTriFairOptimizer,
    "Tri-Fair-v9.3-NoReadinessWarmup": SupportReadyTriFairOptimizer,
    "Tri-Fair-v9.2": MOCAPOFairOptimizer,
    "Tri-Fair-v9.2-NoWeakerDominance": MOCAPOFairOptimizer,
    "Tri-Fair-v9.2-FullEval": MOCAPOFairOptimizer,
    "Tri-Fair-v9.1": IntensifiedTriFairOptimizer,
    "Tri-Fair-v9.1-FullEval": IntensifiedTriFairOptimizer,
    "Tri-Fair-v9.0": DirectionalTriFairOptimizer,
    "Tri-Fair-v9.0-NoDisparity": DirectionalTriFairOptimizer,
    "Tri-Fair-v9.0-NoCoverage": DirectionalTriFairOptimizer,
    "Tri-Fair-v8.9": AdaptiveTriFairOptimizer,
    "Tri-Fair-v8.8": RepairedTriFairOptimizer,
    "Tri-Fair-v8.7": GuardedTriFairOptimizer,
    "Tri-Fair-v8.6": EfficientTriFairOptimizer,
    "Tri-Fair-v8.5": StableTriFairOptimizer,
    "Tri-Fair-v8.4-Hybrid": HybridTriFairOptimizer,
    "Tri-Fair-v8.3": AttributedTriFairOptimizer,
    "Tri-Fair-v8.2-Experimental": RefinedTriFairOptimizer,
    "Tri-Fair-v8": TriFairOptimizer,
    "Tri-Fair-v8-NoFeedback": TriFairOptimizer,
    "Tri-Fair-v8-NoDemo": TriFairOptimizer,
    "Tri-Fair-v8-NoUncertainty": TriFairOptimizer,
    "Tri-Fair-v8-FullEval": TriFairOptimizer,
    "NSGAII-PO-Fair": NSGA2Fair,
    "SPEA2-PO-Fair": SPEA2POFair,
    "NSGAIII-PO-Fair": NSGAIIIPOFair,
    "RandomSearch-PO-Fair": RandomSearchPOFair,
    "NSGAII-PO-Fair-Enhanced": ReflectiveNSGAIIPOFair,
}

# One active proposed method. Earlier versions remain addressable only for
# reproducing existing checkpoints/results; no historical scores are relabeled.
ACTIVE_METHODS = ("Tri-Fair-v9.4", *PLAIN_BASELINES)


def build_optimizer(
    *,
    args,
    optimizer_config,
    model_config,
    predictor,
    task,
    meta_llm,
    initial_prompts,
    df_fewshots,
    callbacks,
):
    cls = METHODS[optimizer_config.name]
    params = dict(optimizer_config.optimizer_params)
    overrides = dict(getattr(args, "optimizer_overrides", None) or {})
    budget_fill = overrides.pop("budget_fill", False)
    if not isinstance(budget_fill, bool):
        raise ValueError("budget_fill must be a boolean")
    if budget_fill and optimizer_config.name in FILLED_BASELINES:
        cls = FILLED_BASELINES[optimizer_config.name]
    elif budget_fill and optimizer_config.name not in {
        *PLAIN_BASELINES,
        "Tri-Fair-v9.4",
        "Tri-Fair-v9.4-NoEvidence",
        "Tri-Fair-v9.4-NoStability",
        "Tri-Fair-v9.3",
        "Tri-Fair-v9.3-NoReadinessWarmup",
        "Tri-Fair-v9.2",
        "Tri-Fair-v9.2-NoWeakerDominance",
        "Tri-Fair-v9.2-FullEval",
        "Tri-Fair-v9.1",
        "Tri-Fair-v9.1-FullEval",
        "Tri-Fair-v8.9",
        "Tri-Fair-v9.0",
        "Tri-Fair-v9.0-NoDisparity",
        "Tri-Fair-v9.0-NoCoverage",
    }:
        raise ValueError(
            "Budget filling requires a registered comparator or explicit Tri-Fair pilot variant"
        )
    if set(overrides) - set(params):
        raise ValueError(
            f"unsupported optimizer overrides: {sorted(set(overrides) - set(params))}"
        )
    params.update(overrides)
    if issubclass(cls, (DirectionalTriFairOptimizer, EvidenceRacingTriFairOptimizer)):
        # Reserve generation and chat-template overhead within the same loaded
        # engine's context; meta and downstream share that engine.
        ceiling = (
            model_config.max_model_len
            - getattr(args, "meta_max_output_tokens", 512)
            - 128
        )
        if ceiling < 512:
            raise ValueError("Insufficient model context for directional search")
        params["generation_input_token_limit"] = min(
            params["generation_input_token_limit"], ceiling
        )
    if args.crossovers_per_iteration is not None:
        params["crossovers_per_iter"] = args.crossovers_per_iteration
    if args.max_few_shot_examples is not None:
        params["upper_shots"] = args.max_few_shot_examples
    if args.disable_objective_aware_variation:
        params["objective_aware_variation"] = False
    if issubclass(cls, TriFairOptimizer):
        bound = (
            args.fixed_cost_upper_bound
            or task.fairness_kwargs["normalization_cost_upper_bound"]
        )
        params["fixed_objective_bounds"] = ((0.0, -float(bound), -1.0), (1.0, 0.0, 0.0))
        n = task.n_blocks
        params["fidelity_utilization_thresholds"] = (0.0, 0.25, 0.5, 0.75, 0.9)
        params["fidelity_block_targets"] = tuple(
            (min(n, max(2, k)) for k in (2, 3, 4, max(4, n - 1), n))
        )
    optimizer = cls(
        predictor=predictor,
        task=task,
        meta_llm=meta_llm,
        initial_prompts=initial_prompts,
        callbacks=callbacks,
        df_few_shots=df_fewshots,
        cost_per_input_token=model_config.input_costs * args.input_cost_multiplier,
        cost_per_output_token=model_config.output_costs * args.output_cost_multiplier,
        **params,
    )
    optimizer.effective_parameters = params
    if budget_fill or issubclass(
        cls, (AdaptiveTriFairOptimizer, DirectionalTriFairOptimizer)
    ):
        optimizer.effective_parameters = dict(
            params,
            budget_fill=True,
            shared_budget_protocol="v89-novel-complete-candidate-retries",
        )
    if issubclass(cls, IntensifiedTriFairOptimizer):
        optimizer.effective_parameters["shared_budget_protocol"] = (
            "v91-next-block-admission-hard-total-cap" if optimizer.enable_intensification
            else "v89-novel-complete-candidate-retries")
    if isinstance(optimizer, MOCAPOFairOptimizer):
        optimizer.effective_parameters = dict(params, budget_fill=False,
            shared_budget_protocol="atomic block admission; hard total cap",
            baseline_budget_fill_requested=budget_fill)
    if isinstance(optimizer, PlainVariationProtocol):
        optimizer.effective_parameters = dict(params, budget_fill=True,
            variation_protocol="original CAPO for evolutionary methods; independent proposals for Random Search",
            shared_budget_protocol="v89-novel-complete-candidate-retries")
    if isinstance(optimizer, EvidenceRacingTriFairOptimizer):
        optimizer.effective_parameters = dict(
            params, budget_fill=True,
            shared_budget_protocol="atomic blocks; bounded novel proposals; frozen terminal audit",
            variation_protocol="cached-search evidence edits plus original CAPO exploration",
        )
    return optimizer


def effective_run_metadata(optimizer, dataset_config, model_config, args):
    root = Path(__file__).resolve().parents[3]
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "source-archive-see-RELEASE_MANIFEST.json"
    digest = hashlib.sha256()
    for path in sorted(
        [*(root / "src").rglob("*.py"), *(root / "scripts").rglob("*.py")]
    ):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    from importlib.metadata import PackageNotFoundError, version

    packages = {}
    for package in [
        "promptolution",
        "numpy",
        "pandas",
        "pymoo",
        "datasets",
        "huggingface-hub",
        "vllm",
        "torch",
    ]:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {
        "package_versions": packages,
        "study_profile": "8.1-clean-search-validation-test",
        "holdout_feedback_used": False,
        "code_revision": revision,
        "source_sha256": digest.hexdigest(),
        "resolved_configuration": {
            "dataset": asdict(dataset_config),
            "model": asdict(model_config),
            "optimizer": type(optimizer).__name__,
            "parameters": optimizer.effective_parameters,
            "search_validation_partition": optimizer.task.study_partition,
            "downstream_decoding": {
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": args.random_seed,
                "structured_labels": args.structured_labels,
                "max_tokens": args.max_output_tokens,
            },
            "meta_decoding": {
                "temperature": 0.55,
                "top_p": 0.95,
                "seed": args.random_seed + 1000003,
                "max_tokens": args.meta_max_output_tokens,
            },
        },
    }
