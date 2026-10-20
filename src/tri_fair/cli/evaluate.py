"""Evaluate registered methods' unique logged prompts in one held-out job.

A prompt is evaluated exactly once per dataset/model/seed and its held-out
result is joined back to every method/checkpoint membership row. Greedy
downstream decoding and stable prompt-ID ordering reduce sampling and batching
variation; GPU arithmetic can still depend on batch shape.
"""

from __future__ import annotations
# ruff: noqa: E402
# Inference environment bootstrap must precede imports that load vLLM.
from tri_fair.runtime.inference_profile import (
    bootstrap_batch_invariance,
    require_matching_profile,
)

bootstrap_batch_invariance()

import argparse
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from promptolution.predictors import MarkerBasedPredictor
from tri_fair.cli.common import (
    atomic_write_parquet,
    configure_logging,
    prompt_id,
    reconstruct_prompts,
    set_generation_limit,
    sha256_file,
    stable_latest_per_prompt,
    utc_now_iso,
)
from tri_fair.reporting.membership import _step_token_table
from tri_fair.reporting.membership import (
    _rename_development_columns,
    load_prompt_candidates,
)
from tri_fair.config.datasets import ALL_DATASETS
from tri_fair.config.models import ALL_MODELS
from tri_fair.config.setup import SETUP
from tri_fair.runtime.decoding import configure_downstream_greedy
from tri_fair.runtime.models import create_llm
from tri_fair.data.loading import create_test_task
from tri_fair.runtime.utils import seed_everything

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--study-profile", choices=["v8"], default="v8")
    parser.add_argument(
        "--evaluation-split", choices=["validation", "test"], default="validation"
    )
    parser.add_argument("--structured-labels", action="store_true")
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--batch-invariant", action="store_true")
    parser.add_argument("--dataset", choices=sorted(ALL_DATASETS), required=True)
    parser.add_argument("--model", choices=sorted(ALL_MODELS), required=True)
    parser.add_argument("--random-seed", type=int, required=True)
    parser.add_argument("--optimizers", default="Tri-Fair-v8.6,NSGAII-PO-Fair")
    parser.add_argument("--minimum-actual-tokens", type=int, default=0)
    parser.add_argument("--maximum-actual-tokens", type=int, default=5000000)
    parser.add_argument("--manifest-dir", default="data/splits_v8_clean")
    parser.add_argument("--max-output-tokens", type=int, default=16)
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--include-history",
        action="store_true",
        help="With --audited-final, also evaluate real logged checkpoint fronts for trajectories.",
    )
    parser.add_argument(
        "--audited-final",
        action="store_true",
        help="Use the frozen, fully evaluated search-audit shortlist and its total token count.",
    )
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="Evaluate only each method's final logged incumbent set.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _parse_optimizers(raw: str) -> tuple[str, ...]:
    values = tuple(
        dict.fromkeys((piece.strip() for piece in str(raw).split(",") if piece.strip()))
    )
    if len(values) < 2:
        raise ValueError("The shared union requires at least two optimizers")
    return values


def _logging_dir(study_root: Path, dataset: str, optimizer: str, seed: int) -> Path:
    """Resolve a standard study pointer or one explicit/unambiguous pilot."""
    pointer = study_root / dataset / optimizer / f"seed{seed}" / "logging_dir.txt"
    if pointer.is_file():
        resolved = Path(pointer.read_text(encoding="utf-8").strip()).expanduser()
        if not resolved.is_absolute():
            resolved = (Path.cwd() / resolved).resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(resolved)
        return resolved
    direct = study_root / "pilot_report.json"
    reports = (
        [direct]
        if direct.is_file()
        else sorted((study_root / dataset / f"seed{seed}").glob("*/pilot_report.json"))
    )
    if len(reports) > 1:
        raise ValueError(
            f"Multiple pilot reports for {dataset}/seed{seed}; pass --study-root "
            "as the exact timestamped directory containing the intended pilot_report.json"
        )
    if not reports:
        raise FileNotFoundError(f"No standard pointer or pilot report: {pointer}")
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    if report.get("dataset") != dataset or int(report.get("seed", -1)) != int(seed):
        raise ValueError("Pilot report dataset/seed differs from requested evaluation")
    records = [r for r in report.get("runs", []) if r.get("method") == optimizer]
    if len(records) != 1 or records[0].get("status") != "measured":
        raise ValueError(f"Pilot lacks one measured run for {optimizer}")
    resolved = reports[0].parent / optimizer / "run"
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    return resolved.resolve()


def _run_args(log_path: Path) -> dict[str, object]:
    path = log_path / "args.json"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def _membership_for_run(
    log_path: Path,
    *,
    optimizer: str,
    dataset: str,
    model: str,
    seed: int,
    minimum_tokens: int,
    maximum_tokens: int,
    final_only: bool = False,
    audited_final: bool = False,
    include_history: bool = False,
) -> pd.DataFrame:
    if audited_final:
        final = _audited_membership(
            log_path,
            optimizer=optimizer,
            dataset=dataset,
            model=model,
            seed=seed,
            minimum_tokens=minimum_tokens,
            maximum_tokens=maximum_tokens,
        )
        if not include_history:
            return final
        history = _membership_for_run(
            log_path,
            optimizer=optimizer,
            dataset=dataset,
            model=model,
            seed=seed,
            minimum_tokens=minimum_tokens,
            maximum_tokens=maximum_tokens,
        )
        return pd.concat([history, final], ignore_index=True, sort=False)
    step_path = log_path / "step_results.parquet"
    if not step_path.is_file():
        raise FileNotFoundError(step_path)
    frame = pd.read_parquet(step_path)
    per_step = _step_token_table(frame)
    per_step = per_step[
        (per_step >= int(minimum_tokens)) & (per_step <= int(maximum_tokens))
    ]
    if per_step.empty:
        raise RuntimeError(
            f"No real states for {optimizer} in {minimum_tokens}..{maximum_tokens} tokens"
        )
    metadata = _run_args(log_path)
    parts: list[pd.DataFrame] = []
    for step, actual_tokens in per_step.items():
        selected, resolved_step = load_prompt_candidates(
            log_path, step=int(step), selection="current"
        )
        if "is_incumbent" not in selected:
            raise ValueError(
                f"{step_path} lacks is_incumbent; current archive cannot be reconstructed"
            )
        selected = selected[
            selected["is_incumbent"].fillna(False).astype(bool)
        ].reset_index(drop=True)
        if selected.empty:
            continue
        selected = _rename_development_columns(selected)
        selected["optimizer"] = optimizer
        selected["dataset"] = dataset
        selected["model"] = model
        selected["seed"] = int(seed)
        selected["chosen_step"] = int(resolved_step)
        selected["actual_budget_tokens"] = int(actual_tokens)
        selected["budget_checkpoint"] = int(actual_tokens)
        selected["checkpoint_signed_error"] = 0
        selected["checkpoint_relative_error"] = 0.0
        selected["checkpoint_policy"] = "shared_union_exact_real_state"
        selected["selection_policy"] = "current_incumbents"
        selected["configured_budget"] = int(
            metadata.get("budget_per_run", maximum_tokens)
        )
        selected["source_log_path"] = str(log_path.resolve())
        if "prompt_id" not in selected:
            selected["prompt_id"] = selected["prompt"].astype(str).map(prompt_id)
        parts.append(selected)
    if not parts:
        raise RuntimeError(f"No incumbent memberships found for {optimizer}")
    if final_only:
        parts = parts[-1:]
    return pd.concat(parts, ignore_index=True, sort=False)


def _audited_membership(
    log_path, *, optimizer, dataset, model, seed, minimum_tokens, maximum_tokens
):
    """Align held-out outcomes with complete, frozen SEARCH audit scores."""
    summary = json.loads(
        (log_path / "finalization_summary.json").read_text(encoding="utf-8")
    )
    if (
        summary.get("status") != "complete"
        or not summary.get("frozen_before_audit")
        or summary.get("feedback_to_optimizer") is not False
    ):
        raise ValueError(
            "A complete frozen search audit without optimizer feedback is required"
        )
    tokens = int(summary["total_downstream_tokens"])
    if not minimum_tokens <= tokens <= maximum_tokens:
        raise ValueError("Audited run lies outside the requested actual-token range")
    selected = pd.read_parquet(log_path / "final_search_results.parquet")
    if len(selected) != int(summary["completed_candidates"]) or selected.empty:
        raise ValueError("Audited candidate count mismatch")
    if not selected.fairness_ready.eq(True).all():
        raise ValueError("Audited search candidates lack fairness support")
    if "downstream_template" not in selected:
        selected["downstream_template"] = None
    prompts = reconstruct_prompts(selected)
    selected["prompt"] = [p.construct_prompt() for p in prompts]
    selected["prompt_id"] = selected.prompt.map(prompt_id)
    if selected.prompt_id.duplicated().any():
        raise ValueError("Duplicate audited prompt identity")
    selected = _rename_development_columns(selected)
    for name, value in dict(
        optimizer=optimizer,
        dataset=dataset,
        model=model,
        seed=int(seed),
        chosen_step=int(summary["search_steps_logged"]),
        actual_budget_tokens=tokens,
        budget_checkpoint=tokens,
        checkpoint_signed_error=0,
        checkpoint_relative_error=0.0,
        checkpoint_policy="frozen_full_search_audit",
        selection_policy="audited_final_shortlist",
        configured_budget=int(summary["total_budget_tokens"]),
        source_log_path=str(log_path.resolve()),
    ).items():
        selected[name] = value
    return selected


def _test_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column.startswith("test_")
        or column
        in {
            "prompt_id",
            "evaluation_timestamp",
            "manifest_path",
            "manifest_sha256",
            "shared_union_evaluation",
        }
    ]


def run(args: argparse.Namespace) -> Path:
    if getattr(args, "include_history", False) and not getattr(
        args, "audited_final", False
    ):
        raise ValueError("--include-history requires --audited-final")
    if args.study_profile == "base" and args.evaluation_split == "validation":
        raise ValueError("validation partition requires study-profile v8")
    if args.maximum_actual_tokens <= args.minimum_actual_tokens:
        raise ValueError("maximum-actual-tokens must exceed minimum-actual-tokens")
    optimizers = _parse_optimizers(args.optimizers)
    study_root = Path(args.study_root).expanduser().resolve()
    memberships = []
    for optimizer in optimizers:
        log_path = _logging_dir(study_root, args.dataset, optimizer, args.random_seed)
        memberships.append(
            _membership_for_run(
                log_path,
                optimizer=optimizer,
                dataset=args.dataset,
                model=args.model,
                seed=args.random_seed,
                minimum_tokens=args.minimum_actual_tokens,
                maximum_tokens=args.maximum_actual_tokens,
                final_only=bool(getattr(args, "final_only", False)),
                audited_final=bool(getattr(args, "audited_final", False)),
                include_history=bool(getattr(args, "include_history", False)),
            )
        )
    membership = pd.concat(memberships, ignore_index=True, sort=False)
    if set(membership["model"].astype(str)) != {args.model}:
        raise ValueError("requested model does not match optimization runs")
    for log_path in [Path(p) for p in membership["source_log_path"].unique()]:
        recorded = _run_args(log_path)
        for flag in ("enable_prefix_caching", "batch_invariant"):
            if bool(recorded.get(flag, False)) != bool(getattr(args, flag, False)):
                raise ValueError(f"evaluation runtime fingerprint differs: {flag}")
        if recorded.get("model", args.model) != args.model:
            raise ValueError("evaluation model differs from optimization")
        if int(recorded.get("max_output_tokens", 16)) != args.max_output_tokens:
            raise ValueError(
                "evaluation decoding fingerprint differs from optimization"
            )
        if bool(recorded.get("structured_labels", False)) != args.structured_labels:
            raise ValueError("evaluation decoding differs from optimization")
        if recorded.get("study_profile", "base") != args.study_profile:
            raise ValueError("evaluation profile differs from optimization")
        if (
            recorded.get("input_cost_multiplier", 1.0) != 1.0
            or recorded.get("output_cost_multiplier", 1.0) != 1.0
        ):
            raise ValueError("shared evaluation requires unit cost multipliers")
    unique_candidates = (
        stable_latest_per_prompt(membership)
        .sort_values("prompt_id", kind="stable")
        .reset_index(drop=True)
    )
    output_path = (
        Path(args.output_file).expanduser().resolve()
        if args.output_file
        else study_root
        / ("shared_union_" + args.evaluation_split)
        / args.model
        / args.dataset
        / f"seed{args.random_seed}"
        / "eval_checkpoints_union.parquet"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    from dataclasses import asdict, replace
    from hashlib import sha256

    model_config = ALL_MODELS[args.model]
    if getattr(args, "enable_prefix_caching", False):
        model_config = replace(
            model_config,
            llm_kwargs=dict(model_config.llm_kwargs, enable_prefix_caching=True),
        )
    require_matching_profile(getattr(args, "batch_invariant", False))
    code_root = Path(__file__).resolve().parents[3]
    evaluator_hash = sha256()
    for path in sorted((code_root / "src").rglob("*.py")):
        evaluator_hash.update(str(path.relative_to(code_root)).encode())
        evaluator_hash.update(path.read_bytes())
    evaluator_hash.update(Path(__file__).read_bytes())
    metadata_path = output_path.with_suffix(".metadata.json")
    identity = {
        "evaluator_sha256": evaluator_hash.hexdigest(),
        "batch_invariant": bool(getattr(args, "batch_invariant", False)),
        "enable_prefix_caching": bool(getattr(args, "enable_prefix_caching", False)),
        "model_configuration": json.loads(
            json.dumps(asdict(model_config), default=str)
        ),
        "model": args.model,
        "dataset": args.dataset,
        "seed": args.random_seed,
        "profile": args.study_profile,
        "split": args.evaluation_split,
        "max_output_tokens": args.max_output_tokens,
        "structured_labels": args.structured_labels,
        "candidate_policy": "audited_final_plus_history"
        if getattr(args, "include_history", False)
        else "audited_final_shortlist"
        if getattr(args, "audited_final", False)
        else "final_incumbents"
        if getattr(args, "final_only", False)
        else "all_logged_incumbents",
        "manifest_sha256": sha256_file(
            Path(args.manifest_dir) / f"{args.dataset}_seed{args.random_seed}.json"
        ),
    }
    if output_path.is_file() and (not args.force):
        if (
            not metadata_path.exists()
            or json.loads(metadata_path.read_text(encoding="utf-8")) != identity
        ):
            raise ValueError(
                "evaluation fingerprint differs; use a new path or --force"
            )
    cached = pd.DataFrame()
    if output_path.is_file() and (not args.force):
        cached = pd.read_parquet(output_path)
        if args.evaluation_split == "validation":
            cached = cached.rename(
                columns={
                    c: c.replace("validation_", "test_", 1)
                    for c in cached
                    if c.startswith("validation_")
                }
            )
    cached_ids = set(cached.get("prompt_id", pd.Series(dtype=str)).astype(str))
    pending = unique_candidates[
        ~unique_candidates["prompt_id"].astype(str).isin(cached_ids)
    ].reset_index(drop=True)
    evaluated = pd.DataFrame()
    evaluation_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    if not pending.empty:
        seed_everything(args.random_seed)
        if args.study_profile == "v8":
            from tri_fair.config.study import apply_study_profile

            dataset_config = apply_study_profile(ALL_DATASETS[args.dataset])
        else:
            dataset_config = apply_study_profile(ALL_DATASETS[args.dataset])
        llm = create_llm(model_config=model_config, seed=args.random_seed)
        configure_downstream_greedy(llm, seed=args.random_seed)
        set_generation_limit(llm, args.max_output_tokens)
        if args.evaluation_split == "validation":
            from tri_fair.data.loading import create_dev_tasks
            from tri_fair.data.partitions import split_search_validation

            development, _ = create_dev_tasks(
                dataset_config,
                eval_strategy="full",
                seed=args.random_seed,
                manifest_dir=args.manifest_dir,
            )
            _, test_task = split_search_validation(development)
        else:
            test_task = create_test_task(
                dataset_config=dataset_config,
                eval_strategy="full",
                n_subsamples=0,
                test_size=SETUP.test_size,
                seed=args.random_seed,
                manifest_dir=args.manifest_dir,
                regenerate_manifest=False,
            )
        if args.structured_labels:
            from tri_fair.runtime.decoding import configure_structured_labels

            configure_structured_labels(llm, test_task.classes)
        prompts = reconstruct_prompts(pending)
        result = test_task.evaluate(
            prompts=prompts,
            predictor=MarkerBasedPredictor(llm, test_task.classes),
            eval_strategy="full",
        )
        evaluation_usage = dict(llm.get_token_count())
        evaluated = pending[["prompt_id"]].copy()
        quality = np.asarray(result.agg_scores, dtype=float)
        input_tokens = np.asarray(result.agg_input_tokens, dtype=float)
        output_tokens = np.asarray(result.agg_output_tokens, dtype=float)
        evaluated["test_quality"] = quality
        evaluated["test_cost"] = (
            float(model_config.input_costs) * input_tokens
            + float(model_config.output_costs) * output_tokens
        )
        evaluated["test_input_tokens"] = input_tokens
        evaluated["test_output_tokens"] = output_tokens
        evaluated["test_fairness"] = np.asarray(result.fairness_loss, dtype=float)
        evaluated["test_fairness_ready"] = np.asarray(result.fairness_ready, dtype=bool)
        evaluated["test_fairness_diagnostics_json"] = [
            json.dumps(value, sort_keys=True, default=str)
            for value in result.fairness_diagnostics
        ]
        evaluated["test_group_support_json"] = [
            json.dumps(value, sort_keys=True, default=str)
            for value in result.fairness_support
        ]
        evaluated["test_objective_vector"] = [
            [float(q), -float(c), -float(f)]
            for q, c, f in zip(
                evaluated["test_quality"],
                evaluated["test_cost"],
                evaluated["test_fairness"],
            )
        ]
        evaluated["evaluation_timestamp"] = utc_now_iso()
        evaluated["manifest_path"] = str(getattr(test_task, "manifest_path", ""))
        evaluated["manifest_sha256"] = sha256_file(evaluated["manifest_path"].iloc[0])
        evaluated["shared_union_evaluation"] = True
    lookup_parts: list[pd.DataFrame] = []
    if not cached.empty:
        lookup_parts.append(stable_latest_per_prompt(cached[_test_columns(cached)]))
    if not evaluated.empty:
        lookup_parts.append(evaluated)
    if not lookup_parts:
        raise RuntimeError("No cached or newly evaluated held-out rows")
    lookup = stable_latest_per_prompt(
        pd.concat(lookup_parts, ignore_index=True, sort=False)
    )
    test_only = [column for column in lookup.columns if column != "prompt_id"]
    output = membership.merge(
        lookup[["prompt_id", *test_only]],
        on="prompt_id",
        how="left",
        validate="many_to_one",
    )
    if output["test_quality"].isna().any():
        raise RuntimeError("Some membership rows lack shared held-out results")
    dedup = ["optimizer", "prompt_id", "chosen_step", "budget_checkpoint"]
    if "selection_policy" in output:
        dedup.append("selection_policy")
    output = (
        output.sort_values(dedup, kind="stable")
        .drop_duplicates(dedup, keep="last")
        .reset_index(drop=True)
    )
    output["evaluation_split"] = args.evaluation_split
    if args.evaluation_split == "validation":
        output = output.rename(
            columns={
                c: c.replace("test_", "validation_", 1)
                for c in output
                if c.startswith("test_")
            }
        )
    atomic_write_parquet(output, output_path)
    from tri_fair.cli.common import atomic_write_json

    atomic_write_json(metadata_path, identity)
    with output_path.with_suffix(".usage.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "timestamp": utc_now_iso(),
                    "evaluation_split": args.evaluation_split,
                    "new_unique_prompts": len(pending),
                    "cached_unique_prompts": len(unique_candidates) - len(pending),
                    "downstream_tokens": evaluation_usage,
                    "budget_scope": "held-out evaluation, outside the optimization token cap",
                },
                sort_keys=True,
            )
            + "\n"
        )
    LOGGER.info(
        "Wrote %d memberships for %d unique prompts to %s",
        len(output),
        output["prompt_id"].nunique(),
        output_path,
    )
    return output_path


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    run(args)


if __name__ == "__main__":
    main()
