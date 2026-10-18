"""Explicit factory and effective metadata for the upgraded study."""

from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import hashlib
import subprocess
from tri_fair.optimizers.trifair import TriFairOptimizer
from tri_fair.optimizers.refined import RefinedTriFairOptimizer
from tri_fair.optimizers.attributed import AttributedTriFairOptimizer
from tri_fair.optimizers.stable import StableTriFairOptimizer
from tri_fair.optimizers.nsga2 import NSGA2Fair
from tri_fair.optimizers.hybrid import HybridTriFairOptimizer
from tri_fair.optimizers.baselines import (
    SPEA2POFair,
    NSGAIIIPOFair,
    RandomSearchPOFair,
    ReflectiveNSGAIIPOFair,
)

METHODS = {
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
    overrides = getattr(args, "optimizer_overrides", None) or {}
    if set(overrides) - set(params):
        raise ValueError(
            f"unsupported optimizer overrides: {sorted(set(overrides) - set(params))}"
        )
    params.update(overrides)
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
