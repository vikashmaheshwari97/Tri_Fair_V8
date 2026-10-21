"""Run a current Tri-Fair method with strict budgets and resumable checkpoints."""
# ruff: noqa: E402
# Inference environment bootstrap must precede imports that load vLLM.

from tri_fair.runtime.inference_profile import (
    bootstrap_batch_invariance,
    require_matching_profile,
)

bootstrap_batch_invariance()

from tri_fair.config.study import build_optimizer_registry
import argparse
import json
import logging
import random
import traceback
from pathlib import Path
from typing import Any
from promptolution.optimizers.base_optimizer import BaseOptimizer
from promptolution.predictors import MarkerBasedPredictor
from promptolution.utils import LoggerCallback
from promptolution.utils.callbacks import TokenCountCallback
from tri_fair.runtime.callbacks import ExperimentCallback
from tri_fair.runtime.budget import (
    BudgetAwarePredictor,
    BudgetStopCallback,
    TokenBudgetController,
)
from tri_fair.runtime.checkpointing import OptimizerCheckpointCallback
from tri_fair.config.datasets import ALL_DATASETS
from tri_fair.config.models import ALL_MODELS
from tri_fair.config.setup import SETUP
from tri_fair.runtime.models import create_llm
from tri_fair.data.loading import create_dev_tasks
from tri_fair.runtime.utils import copy_llm, generate_random_hash, seed_everything
from tri_fair.cli.common import (
    append_jsonl,
    atomic_write_json,
    compare_immutable_args,
    configure_logging,
    directory_lock,
    manifest_lock_path,
    parse_positive_int_csv,
    set_generation_limit,
    token_counts,
    utc_now_iso,
)
from tri_fair.config.study import apply_study_profile

CURRENT_OPTIMIZERS = build_optimizer_registry()
LOGGER = logging.getLogger(__name__)
FAIRNESS_OPTIMIZERS = {"Tri-Fair", "NSGAII-PO-Fair", *build_optimizer_registry()}


def _load_optimizer_overrides(value: str) -> dict:
    with Path(value).expanduser().open(encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise argparse.ArgumentTypeError("optimizer overrides must be a JSON object")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one prompt-optimization configuration with resumable checkpoints."
    )
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--random-seed", type=int, required=True)
    parser.add_argument("--budget-per-run", type=int, required=True)
    parser.add_argument("--output-dir", default="results/main_results")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dataset", choices=sorted(ALL_DATASETS), required=True)
    parser.add_argument("--model", choices=sorted(ALL_MODELS), required=True)
    parser.add_argument(
        "--optimizer", choices=sorted(CURRENT_OPTIMIZERS), required=True
    )
    parser.add_argument("--study-profile", choices=["v8"], default="v8")
    parser.add_argument(
        "--optimizer-overrides",
        type=_load_optimizer_overrides,
        default=None,
        help="v8 only: JSON file with explicit optimizer parameter overrides.",
    )
    parser.add_argument("--structured-labels", action="store_true")
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--batch-invariant", action="store_true")
    parser.add_argument("--n-init-prompts", type=int, default=12)
    parser.add_argument("--max-output-tokens", type=int, default=16)
    parser.add_argument("--meta-max-output-tokens", type=int, default=256)
    parser.add_argument(
        "--budget-checkpoints",
        default="250000,500000,750000,1000000",
        help="Comma-separated downstream/evaluation-model token thresholds.",
    )
    parser.add_argument("--manifest-dir", default="data/splits")
    parser.add_argument("--regenerate-manifest", action="store_true")
    parser.add_argument("--manifest-lock-timeout", type=float, default=3600.0)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--max-steps", type=int, default=SETUP.n_steps)
    parser.add_argument("--random-selection", action="store_true")
    parser.add_argument("--no-weaker-dominance", action="store_true")
    parser.add_argument("--crossovers-per-iteration", type=int, default=None)
    parser.add_argument("--max-few-shot-examples", type=int, default=None)
    parser.add_argument("--input-cost-multiplier", type=float, default=1.0)
    parser.add_argument("--output-cost-multiplier", type=float, default=1.0)
    parser.add_argument("--fixed-cost-upper-bound", type=float, default=None)
    parser.add_argument("--disable-objective-aware-variation", action="store_true")
    parser.add_argument(
        "--strict-token-budget",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject downstream calls before they can cross the configured budget.",
    )
    parser.add_argument(
        "--near-budget-fraction",
        type=float,
        default=0.9,
        help="Begin reducing challenger count after this budget fraction.",
    )
    parser.add_argument(
        "--minimum-budget-utilization",
        type=float,
        default=0.95,
        help="Minimum accepted utilization for an atomic under-budget stop.",
    )
    parser.add_argument(
        "--output-token-reserve",
        type=int,
        default=2,
        help="Tokenizer margin reserved per downstream response.",
    )
    parser.add_argument(
        "--allow-under-budget-exit",
        action="store_true",
        help="Permit a run to exit below the requested token budget (debug only).",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--final-search-candidates",
        type=int,
        default=0,
        help="Optional pilot-only frozen-shortlist audit; zero preserves normal training.",
    )
    parser.add_argument("--reclaim-audit-reserve", action="store_true")
    parser.add_argument("--final-search-reserve", type=float, default=0.35)
    parser.add_argument(
        "--max-idle-steps",
        type=int,
        default=0,
        help="Optional shared watchdog; zero preserves ordinary training.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.optimizer.startswith(("Tri-Fair-v9.2", "Tri-Fair-v9.3")) and getattr(args, "reclaim_audit_reserve", False):
        raise ValueError("The mixed-fidelity pilot uses a frozen terminal audit; reserve reclamation would change the search protocol")
    if args.max_idle_steps < 0:
        raise ValueError("max-idle-steps cannot be negative")
    if args.final_search_candidates < 0 or not 0 < args.final_search_reserve < 1:
        raise ValueError("Invalid final search audit candidate count or reserve")
    if args.final_search_candidates and (
        not args.strict_token_budget
        or args.study_profile != "v8"
        or args.resume_from
        or not args.allow_under_budget_exit
    ):
        raise ValueError(
            "Final search audit requires a fresh strict-budget v8 pilot with --allow-under-budget-exit"
        )
    if args.study_profile == "v8" and args.n_init_prompts < 2:
        raise ValueError("v8 search requires at least two initial prompts")
    if args.study_profile != "v8" and args.optimizer_overrides:
        raise ValueError("optimizer-overrides requires study-profile v8")
    if args.study_profile == "v8" and (
        args.random_selection or args.no_weaker_dominance
    ):
        raise ValueError("legacy MO-CAPO switches do not apply to the v8 registry")
    positive_fields = {
        "budget_per_run": args.budget_per_run,
        "n_init_prompts": args.n_init_prompts,
        "max_output_tokens": args.max_output_tokens,
        "meta_max_output_tokens": args.meta_max_output_tokens,
        "max_steps": args.max_steps,
    }
    for name, value in positive_fields.items():
        if int(value) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.crossovers_per_iteration is not None and args.crossovers_per_iteration <= 0:
        raise ValueError("--crossovers-per-iteration must be positive")
    if args.max_few_shot_examples is not None and args.max_few_shot_examples < 0:
        raise ValueError("--max-few-shot-examples cannot be negative")
    if args.input_cost_multiplier < 0 or args.output_cost_multiplier < 0:
        raise ValueError("Cost multipliers cannot be negative")
    if args.fixed_cost_upper_bound is not None and args.fixed_cost_upper_bound <= 0:
        raise ValueError("--fixed-cost-upper-bound must be positive")
    if not 0.0 < args.near_budget_fraction < 1.0:
        raise ValueError("--near-budget-fraction must lie strictly between 0 and 1")
    if not 0.0 < args.minimum_budget_utilization <= 1.0:
        raise ValueError("--minimum-budget-utilization must lie in (0, 1]")
    if args.output_token_reserve < 0:
        raise ValueError("--output-token-reserve cannot be negative")


def _resolve_logging_dir(args: argparse.Namespace) -> Path:
    if args.resume_from:
        checkpoint = Path(args.resume_from).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if checkpoint.parent.name != "checkpoints":
            raise ValueError(
                "--resume-from must point to <logging_dir>/checkpoints/<file>.pkl"
            )
        return checkpoint.parent.parent
    run_id = args.run_id or generate_random_hash()[:8]
    return (Path(args.output_dir).expanduser().resolve() / run_id).resolve()


def _validate_resume_metadata(logging_dir: Path, args: argparse.Namespace) -> None:
    if not args.resume_from:
        return
    prior_path = logging_dir / "args.json"
    if not prior_path.exists():
        raise FileNotFoundError(
            f"Cannot validate resumed run because {prior_path} does not exist"
        )
    with prior_path.open("r", encoding="utf-8") as handle:
        previous = json.load(handle)
    current = vars(args)
    previous.setdefault("max_idle_steps", 0)
    if previous.get("final_search_candidates", 0):
        raise ValueError(
            "Pilot audit checkpoints are search-only; start a fresh pilot instead of resuming"
        )
    compare_immutable_args(
        previous,
        current,
        keys=(
            "optimizer_overrides",
            "study_profile",
            "structured_labels",
            "enable_prefix_caching",
            "batch_invariant",
            "random_seed",
            "dataset",
            "model",
            "optimizer",
            "n_init_prompts",
            "manifest_dir",
            "max_output_tokens",
            "meta_max_output_tokens",
            "input_cost_multiplier",
            "output_cost_multiplier",
            "crossovers_per_iteration",
            "max_few_shot_examples",
            "fixed_cost_upper_bound",
            "disable_objective_aware_variation",
            "strict_token_budget",
            "near_budget_fraction",
            "minimum_budget_utilization",
            "output_token_reserve",
            "max_idle_steps",
        ),
    )


def _optimizer_parameters(
    args: argparse.Namespace, optimizer_config: Any
) -> dict[str, Any]:
    params = dict(optimizer_config.optimizer_params)
    params.update(getattr(args, "optimizer_overrides", None) or {})
    if args.crossovers_per_iteration is not None:
        params["crossovers_per_iter"] = int(args.crossovers_per_iteration)
    if args.max_few_shot_examples is not None:
        params["upper_shots"] = int(args.max_few_shot_examples)
    if (
        optimizer_config.optimizer in FAIRNESS_OPTIMIZERS
        and args.disable_objective_aware_variation
    ):
        params["objective_aware_variation"] = False
    return params


def _build_optimizer(
    *,
    args: argparse.Namespace,
    optimizer_config: Any,
    model_config: Any,
    predictor: MarkerBasedPredictor,
    task: Any,
    meta_llm: Any,
    initial_prompts: list[str],
    df_fewshots: Any,
    callbacks: list[Any],
) -> BaseOptimizer:
    from tri_fair.config.registry import build_optimizer

    return build_optimizer(
        args=args,
        optimizer_config=optimizer_config,
        model_config=model_config,
        predictor=predictor,
        task=task,
        meta_llm=meta_llm,
        initial_prompts=initial_prompts,
        df_fewshots=df_fewshots,
        callbacks=callbacks,
    )


def _write_stage_metadata(
    logging_dir: Path,
    args: argparse.Namespace,
    *,
    status: str,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "status": status,
        "timestamp": utc_now_iso(),
        "experiment_name": args.experiment_name,
        "model": args.model,
        "dataset": args.dataset,
        "optimizer": args.optimizer,
        "seed": args.random_seed,
        "requested_budget": args.budget_per_run,
        "resume_from": args.resume_from,
        "slurm_job_id": None,
        "slurm_array_task_id": None,
    }
    import os

    payload["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    payload["slurm_array_task_id"] = os.environ.get("SLURM_ARRAY_TASK_ID")
    if extra:
        payload.update(extra)
    append_jsonl(logging_dir / "stage_history.jsonl", payload)


def run(args: argparse.Namespace, *, logging_dir: Path | None = None) -> Path:
    _validate_args(args)
    dataset_config = ALL_DATASETS[args.dataset]
    optimizer_config = CURRENT_OPTIMIZERS[args.optimizer]
    model_config = ALL_MODELS[args.model]
    if args.study_profile == "v8":
        dataset_config = apply_study_profile(dataset_config)
        optimizer_config = build_optimizer_registry()[args.optimizer]
    elif args.optimizer in set(build_optimizer_registry()) - {"NSGAII-PO-Fair"}:
        raise ValueError(
            "Select tri_fair.cli.train_v8 or --study-profile v8 for this method"
        )
    if args.enable_prefix_caching:
        from dataclasses import replace

        model_config = replace(
            model_config,
            llm_kwargs=dict(model_config.llm_kwargs, enable_prefix_caching=True),
        )
    require_matching_profile(args.batch_invariant)
    is_fairness_dataset = dataset_config.task_type == "Fairness"
    is_fairness_optimizer = args.optimizer in FAIRNESS_OPTIMIZERS
    if is_fairness_optimizer and (not is_fairness_dataset):
        raise ValueError(f"{args.optimizer} requires a fairness-aware dataset")
    if is_fairness_dataset and (not is_fairness_optimizer):
        raise ValueError(
            "Fairness datasets require a registered three-objective method in the main implementation. This prevents accidental optimization of quality and cost while silently ignoring fairness."
        )
    if args.n_init_prompts > len(dataset_config.initial_prompts):
        raise ValueError(
            f"Requested {args.n_init_prompts} initial prompts, but {args.dataset} contains only {len(dataset_config.initial_prompts)}"
        )
    checkpoints = parse_positive_int_csv(
        args.budget_checkpoints, include=args.budget_per_run
    )
    logging_dir = logging_dir or _resolve_logging_dir(args)
    logging_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _validate_resume_metadata(logging_dir, args)
    pointer = output_dir / "logging_dir.txt"
    pointer.write_text(str(logging_dir), encoding="utf-8")
    seed_everything(args.random_seed)
    atomic_write_json(logging_dir / "args.json", vars(args))
    _write_stage_metadata(
        logging_dir, args, status="starting", extra={"checkpoints": checkpoints}
    )
    downstream_llm = create_llm(model_config=model_config, seed=args.random_seed)
    meta_llm = copy_llm(downstream_llm)
    if args.study_profile == "v8":
        from tri_fair.runtime.decoding import (
            configure_downstream_greedy,
            configure_meta_search,
        )

        configure_downstream_greedy(downstream_llm, seed=args.random_seed)
        configure_meta_search(meta_llm, seed=args.random_seed + 1000003)
    set_generation_limit(downstream_llm, args.max_output_tokens)
    set_generation_limit(meta_llm, args.meta_max_output_tokens)
    lock = manifest_lock_path(args.manifest_dir, args.dataset, args.random_seed)
    with directory_lock(lock, timeout_seconds=args.manifest_lock_timeout):
        dev_task, df_fewshots = create_dev_tasks(
            dataset_config=dataset_config,
            eval_strategy=optimizer_config.eval_strategy,
            n_subsamples=optimizer_config.n_subsamples,
            dev_size=SETUP.dev_size,
            fs_size=SETUP.fs_size,
            seed=args.random_seed,
            manifest_dir=args.manifest_dir,
            regenerate_manifest=args.regenerate_manifest,
        )
    if args.study_profile == "v8":
        from tri_fair.data.partitions import split_search_validation

        dev_task, _ = split_search_validation(dev_task)
    if args.structured_labels:
        from tri_fair.runtime.decoding import configure_structured_labels

        configure_structured_labels(downstream_llm, dev_task.classes)
    initial_prompts = random.Random(args.random_seed).sample(
        list(dataset_config.initial_prompts), args.n_init_prompts
    )
    base_predictor = MarkerBasedPredictor(downstream_llm, dev_task.classes)
    budget_controller = None
    if args.strict_token_budget:
        budget_controller = TokenBudgetController(
            llm=downstream_llm,
            max_tokens=(
                int(args.budget_per_run * (1 - args.final_search_reserve))
                if args.final_search_candidates
                else args.budget_per_run
            ),
            max_output_tokens=args.max_output_tokens,
            near_budget_fraction=args.near_budget_fraction,
            output_token_reserve=args.output_token_reserve,
            event_log_path=logging_dir / "budget_events.jsonl",
        )
        predictor = BudgetAwarePredictor(base_predictor, budget_controller)
    else:
        predictor = base_predictor
    callbacks = [
        LoggerCallback(LOGGER),
        ExperimentCallback(dir=str(logging_dir)),
        OptimizerCheckpointCallback(logging_dir, checkpoints),
    ]
    if args.max_idle_steps:
        from tri_fair.runtime.progress import EvaluationProgressCallback

        callbacks.append(EvaluationProgressCallback(args.max_idle_steps))
    if budget_controller is not None:
        callbacks.append(BudgetStopCallback(budget_controller))
    else:
        callbacks.append(TokenCountCallback(args.budget_per_run, "total_tokens"))
    optimizer = _build_optimizer(
        args=args,
        optimizer_config=optimizer_config,
        model_config=model_config,
        predictor=predictor,
        task=dev_task,
        meta_llm=meta_llm,
        initial_prompts=initial_prompts,
        df_fewshots=df_fewshots,
        callbacks=callbacks,
    )
    if hasattr(optimizer, "set_artifact_directory"):
        optimizer.set_artifact_directory(logging_dir)
    if budget_controller is not None:
        optimizer.budget_controller = budget_controller
    if args.resume_from:
        if not hasattr(optimizer, "load_checkpoint"):
            raise TypeError(
                f"{optimizer.__class__.__name__} does not support checkpoint resume"
            )
        optimizer.load_checkpoint(args.resume_from)
    starting_downstream = token_counts(downstream_llm)
    starting_meta = token_counts(meta_llm)
    LOGGER.info(
        "Starting %s/%s/%s/seed%d at %d downstream tokens; target=%d",
        args.model,
        args.dataset,
        args.optimizer,
        args.random_seed,
        starting_downstream["total_tokens"],
        args.budget_per_run,
    )
    if starting_downstream["total_tokens"] < args.budget_per_run:
        optimizer.optimize(n_steps=args.max_steps)
    else:
        LOGGER.warning(
            "Checkpoint already contains %d downstream tokens, which meets target %d; no additional optimization steps were executed.",
            starting_downstream["total_tokens"],
            args.budget_per_run,
        )
    final_search_audit = None
    if args.final_search_candidates:
        from tri_fair.runtime.finalization import finalize_search_front

        finalizer = finalize_search_front
        options = {}
        if getattr(args, "reclaim_audit_reserve", False):
            from tri_fair.runtime.finalization import reclaim_and_finalize

            finalizer = reclaim_and_finalize
            options["max_steps"] = args.max_steps
        final_search_audit = finalizer(
            optimizer,
            logging_dir,
            count=args.final_search_candidates,
            total_budget=args.budget_per_run,
            **options,
        )
    final_downstream = token_counts(downstream_llm)
    final_meta = token_counts(meta_llm)
    final_total = int(final_downstream["total_tokens"])
    if args.strict_token_budget and final_total > args.budget_per_run:
        raise RuntimeError(
            f"Strict downstream-token budget violation: actual={final_total}, target={args.budget_per_run}"
        )
    utilization = final_total / args.budget_per_run
    atomic_budget_stop = bool(
        budget_controller is not None
        and budget_controller.stopping_reason
        in {
            "next_atomic_operation_exceeds_budget",
            "next_complete_candidate_exceeds_budget",
        }
    )
    accepted_atomic_stop = bool(
        atomic_budget_stop and utilization >= args.minimum_budget_utilization
    )
    if (
        final_total < args.budget_per_run
        and (not args.allow_under_budget_exit)
        and (not accepted_atomic_stop)
    ):
        reason = (
            budget_controller.stopping_reason if budget_controller is not None else None
        )
        raise RuntimeError(
            f"Optimizer exited below the requested evaluation-token budget: actual={final_total}, target={args.budget_per_run}, utilization={utilization:.6f}, stopping_reason={reason!r}. A clean strict-budget stop is accepted only when no complete atomic operation fits and the minimum utilization threshold is met."
        )
    budget_summary = (
        budget_controller.summary()
        if budget_controller is not None
        else {
            "strict_budget_enabled": False,
            "requested_budget_tokens": args.budget_per_run,
            "actual_downstream_tokens": final_total,
            "remaining_tokens": max(0, args.budget_per_run - final_total),
            "budget_difference_tokens": final_total - args.budget_per_run,
            "budget_utilization": utilization,
            "stopping_reason": "legacy_post_step_callback",
        }
    )
    if budget_controller is not None:
        budget_controller.write_summary(logging_dir / "budget_summary.json")
    summary = {
        "status": "complete",
        "completed_at": utc_now_iso(),
        "model": args.model,
        "dataset": args.dataset,
        "optimizer": args.optimizer,
        "seed": args.random_seed,
        "requested_budget": args.budget_per_run,
        "budget_checkpoints": checkpoints,
        "starting_downstream_tokens": starting_downstream,
        "final_downstream_tokens": final_downstream,
        "starting_meta_tokens": starting_meta,
        "final_meta_tokens": final_meta,
        "budget_overshoot_tokens": max(
            0, final_downstream["total_tokens"] - args.budget_per_run
        ),
        "budget_difference_tokens": final_downstream["total_tokens"]
        - args.budget_per_run,
        "budget_utilization": utilization,
        "budget_controller": budget_summary,
        "manifest_path": getattr(dev_task, "manifest_path", None),
        "logging_dir": str(logging_dir),
        "checkpoint_latest": str(logging_dir / "checkpoints" / "latest.pkl"),
        "method_version": getattr(optimizer, "method_version", "8.1-clean-baseline"),
        "fairness_metric": getattr(dev_task, "fairness_metric_name", None),
        "fairness_kwargs": getattr(dev_task, "fairness_kwargs", None),
        "population_size": len(initial_prompts),
        "optimizer_parameters": _optimizer_parameters(args, optimizer_config),
        "final_search_audit": final_search_audit,
        "runtime_stop_reason": getattr(optimizer, "runtime_stop_reason", None),
        "search_diagnostics": optimizer.search_diagnostics()
        if hasattr(optimizer, "search_diagnostics")
        else None,
    }
    if args.study_profile == "v8":
        from tri_fair.config.registry import effective_run_metadata

        summary.update(
            effective_run_metadata(optimizer, dataset_config, model_config, args)
        )
        atomic_write_json(
            logging_dir / "resolved_configuration.json",
            summary["resolved_configuration"],
        )
    atomic_write_json(logging_dir / "run_summary.json", summary)
    _write_stage_metadata(logging_dir, args, status="complete", extra=summary)
    LOGGER.info("Run completed: %s", logging_dir)
    return logging_dir


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    logging_dir = _resolve_logging_dir(args)
    try:
        run(args, logging_dir=logging_dir)
    except Exception as error:
        try:
            logging_dir.mkdir(parents=True, exist_ok=True)
            failure = {
                "status": "failed",
                "failed_at": utc_now_iso(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "model": args.model,
                "dataset": args.dataset,
                "optimizer": args.optimizer,
                "seed": args.random_seed,
                "requested_budget": args.budget_per_run,
            }
            atomic_write_json(logging_dir / "run_summary.json", failure)
            _write_stage_metadata(logging_dir, args, status="failed", extra=failure)
        except Exception:
            LOGGER.exception("Could not persist failure metadata")
        raise


if __name__ == "__main__":
    main()
