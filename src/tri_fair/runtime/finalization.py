"""Budgeted, frozen-shortlist search audit, identical for every pilot method."""

from __future__ import annotations
import json
import math
from pathlib import Path
import pandas as pd
from tri_fair.cli.common import atomic_write_json
from tri_fair.runtime.budget import BudgetExhausted


def finalize_search_front(optimizer, directory, *, count, total_budget):
    """Complete search coverage after freezing candidates; never feed back results."""
    directory = Path(directory)
    controller = optimizer.budget_controller
    search_summary = controller.summary()
    bound = float(optimizer.task.fairness_kwargs["normalization_cost_upper_bound"])
    candidates, seen = [], set()
    for prompt in optimizer.incumbents:
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
        candidates.append((distance, key, prompt))
    shortlist = [
        item[2]
        for item in sorted(candidates, key=lambda item: (item[0], item[1]))[:count]
    ]
    atomic_write_json(
        directory / "final_search_shortlist.json",
        {
            "split": "search",
            "frozen_before_audit": True,
            "selection": "lowest squared normalized distance among final search incumbents",
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
