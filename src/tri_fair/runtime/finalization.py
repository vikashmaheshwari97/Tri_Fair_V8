"""Budgeted, frozen-shortlist search audit, identical for every pilot method."""

from __future__ import annotations
import json
import math
import logging
from pathlib import Path
import pandas as pd
from tri_fair.cli.common import atomic_write_json
from tri_fair.runtime.budget import BudgetExhausted


def finalize_search_front(
    optimizer, directory, *, count, total_budget, candidate_pool=None, anchors=False
):
    """Complete search coverage after freezing candidates; never feed back results."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    controller = optimizer.budget_controller
    search_summary = controller.summary()
    bound = float(optimizer.task.fairness_kwargs["normalization_cost_upper_bound"])
    candidates, seen = [], set()
    for prompt in optimizer.incumbents if candidate_pool is None else candidate_pool:
        key = prompt.construct_prompt()
        if key in seen:
            continue
        seen.add(key)
        record = optimizer.get_fairness_record(prompt)
        values = [float(record[name]) for name in ("quality", "cost", "fairness")]
        if not all(math.isfinite(value) for value in values):
            continue
        quality, cost, unfairness = values
        distance = (1 - quality) ** 2 + (cost / bound) ** 2 + unfairness**2
        candidates.append(
            (distance, key, prompt, (1 - quality, cost / bound, unfairness))
        )
    ordered = sorted(candidates, key=lambda item: (item[0], item[1]))
    if anchors:
        # Same Pareto filtering and objective anchors for every method.
        front = [
            c
            for c in ordered
            if not any(
                all(x <= y for x, y in zip(d[3], c[3]))
                and any(x < y for x, y in zip(d[3], c[3]))
                for d in ordered
            )
        ]
        selected = []
        if front:
            for item in [front[0]] + [
                min(front, key=lambda c: (c[3][j], c[0], c[1])) for j in range(3)
            ]:
                if item[1] not in {v[1] for v in selected}:
                    selected.append(item)
            selected += [c for c in front if c[1] not in {v[1] for v in selected}]
        ordered = selected
    shortlist = [item[2] for item in ordered[:count]]
    atomic_write_json(
        directory / "final_search_shortlist.json",
        {
            "split": "search",
            "frozen_before_audit": True,
            "selection": "Pareto balanced and objective anchors"
            if anchors
            else "lowest squared normalized distance among final search incumbents",
            "prompts": [
                {"instruction": p.instruction, "few_shots": list(p.few_shots)}
                for p in shortlist
            ],
        },
    )
    if (
        total_budget < controller.max_tokens
        or total_budget < controller.consumed_tokens
    ):
        raise ValueError("Final search cap must include all previously consumed tokens")
    # Same counter and admission checks: reserved tokens are part of the cap,
    # never an additional unreported evaluation budget.
    controller.max_tokens = int(total_budget)
    controller.stop_requested = False
    controller.stopping_reason = None
    controller.stopping_operation = None
    controller.estimated_next_operation_tokens = None
    path = directory / "step_results.parquet"
    step = int(pd.read_parquet(path)["step"].max()) if path.is_file() else 0
    rows, errors = [], []
    for prompt in shortlist:
        try:
            result = optimizer.task.evaluate(
                [prompt], optimizer.predictor, eval_strategy="full"
            )
        except BudgetExhausted as error:
            errors.append({"instruction": prompt.instruction, "reason": str(error)})
            continue
        quality = float(result.agg_scores[0])
        cost = float(
            optimizer.cost_per_input_token * result.agg_input_tokens[0]
            + optimizer.cost_per_output_token * result.agg_output_tokens[0]
        )
        unfairness = float(result.fairness_loss[0])
        rows.append(
            {
                "instruction": prompt.instruction,
                "few_shots_json": json.dumps(
                    list(prompt.few_shots), ensure_ascii=False
                ),
                "downstream_template": prompt.downstream_template,
                "quality": quality,
                "cost": cost,
                "fairness": unfairness,
                "fairness_ready": bool(result.fairness_ready[0]),
                "fairness_diagnostics_json": json.dumps(
                    result.fairness_diagnostics[0], ensure_ascii=False
                ),
                "group_support_json": json.dumps(
                    result.fairness_support[0], ensure_ascii=False
                ),
                "is_incumbent": True,
                "evaluated_on_blocks": json.dumps(list(range(optimizer.task.n_blocks))),
                "step": step,
                "split": "search",
                "audit_only": True,
            }
        )
    if rows:
        temporary = directory / "final_search_results.parquet.tmp"
        pd.DataFrame(rows).to_parquet(temporary, index=False)
        temporary.replace(directory / "final_search_results.parquet")
    summary = {
        "status": "complete"
        if shortlist and len(rows) == len(shortlist)
        else "incomplete",
        "split": "search",
        "frozen_before_audit": True,
        "feedback_to_optimizer": False,
        "requested_candidates": count,
        "shortlisted_candidates": len(shortlist),
        "completed_candidates": len(rows),
        "search_steps_logged": step,
        "search_budget_tokens": search_summary["requested_budget_tokens"],
        "search_downstream_tokens": search_summary["actual_downstream_tokens"],
        "audit_downstream_tokens": controller.consumed_tokens
        - search_summary["actual_downstream_tokens"],
        "total_budget_tokens": total_budget,
        "total_downstream_tokens": controller.consumed_tokens,
        "search_controller": search_summary,
        "errors": errors,
    }
    atomic_write_json(directory / "finalization_summary.json", summary)
    return summary


def reclaim_and_finalize(optimizer, directory, *, count, total_budget, max_steps):
    """Complete protected search candidates, reuse spare cap, then freeze final audit.

    Coverage completion is SEARCH evidence and may feed further search. Only the
    final shortlist is frozen. Validation/test never enter this function.
    """
    controller = optimizer.budget_controller
    initial = controller.summary()
    stopped_for_budget = controller.stopping_reason in {
        "next_atomic_operation_exceeds_budget",
        "next_complete_candidate_exceeds_budget",
    }
    logging.getLogger(__name__).info(
        "Starting protected SEARCH coverage at %d tokens, total cap %d",
        controller.consumed_tokens,
        total_budget,
    )
    coverage = finalize_search_front(
        optimizer,
        Path(directory) / "coverage_checkpoint",
        count=count,
        total_budget=total_budget,
        anchors=True,
    )
    coverage["feedback_to_optimizer"] = True
    coverage["phase"] = "intermediate_search_coverage"
    atomic_write_json(
        Path(directory) / "coverage_checkpoint" / "finalization_summary.json", coverage
    )
    after_coverage = controller.consumed_tokens
    continued = (
        coverage["status"] == "complete"
        and stopped_for_budget
        and not getattr(optimizer, "runtime_stop_reason", None)
        and optimizer._completed_steps < max_steps
        and after_coverage < total_budget
    )
    logging.getLogger(__name__).info(
        "Reserve reclamation: coverage=%s, used=%d, remaining=%d, continue=%s",
        coverage["status"],
        after_coverage,
        total_budget - after_coverage,
        continued,
    )
    if continued:
        # Retain population, caches, RNG, operator credit and total step limit.
        optimizer._resume_loaded = True
        optimizer.optimize(n_steps=max_steps)
    full_blocks = set(range(optimizer.task.n_blocks))
    candidates = [
        p
        for p, blocks in optimizer.task.prompt_evaluated_blocks.items()
        if full_blocks.issubset(set(blocks))
    ]
    candidates = [
        p for p in candidates if optimizer.get_fairness_record(p)["fairness_ready"]
    ]
    before_final = controller.consumed_tokens
    stop_state = {
        k: getattr(controller, k)
        for k in (
            "stop_requested",
            "stopping_reason",
            "stopping_operation",
            "estimated_next_operation_tokens",
        )
    }
    final = finalize_search_front(
        optimizer,
        directory,
        count=count,
        total_budget=total_budget,
        candidate_pool=candidates,
        anchors=True,
    )
    for key, value in stop_state.items():
        setattr(controller, key, value)
    if controller.consumed_tokens != before_final:
        raise RuntimeError(
            "Cached-only final audit unexpectedly used downstream tokens"
        )
    final["reserve_reclamation"] = {
        "protocol": "protected search coverage; continued search; cached-only frozen final audit",
        "initial_search_cap": initial["requested_budget_tokens"],
        "initial_search_tokens": initial["actual_downstream_tokens"],
        "coverage_tokens": after_coverage - initial["actual_downstream_tokens"],
        "continued_search_tokens": before_final - after_coverage,
        "continued": bool(continued),
        "coverage_status": coverage["status"],
        "fully_evaluated_candidates": len(candidates),
        "same_policy_for_all_methods": True,
    }
    if coverage["status"] != "complete":
        final["status"] = "incomplete"
        final["errors"].append({"reason": "protected search coverage incomplete"})
    atomic_write_json(Path(directory) / "finalization_summary.json", final)
    return final
