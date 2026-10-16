"""Experiment logging callbacks for MO-CAPO and Tri-Fair."""

from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd
from promptolution.utils.callbacks import BaseCallback
from tri_fair.optimizers.pareto import ParetoPromptOptimizer
from tri_fair.fairness.base import json_safe


class ExperimentCallback(BaseCallback):
    """Persist prompt populations, objective vectors, and fairness diagnostics."""

    def __init__(self, dir: str) -> None:
        self.directory = Path(dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "step_results.parquet"
        self.step_dir = self.directory / "step_logs"
        self.step_dir.mkdir(parents=True, exist_ok=True)
        self.runhistory_path = self.directory / "runhistory.json"
        self.step = (
            int(pd.read_parquet(self.path)["step"].max()) if self.path.exists() else 0
        )

    def on_step_end(self, optimizer) -> bool:
        self.step += 1
        rows = []
        timestamp = datetime.now(timezone.utc).timestamp()
        downstream_counts = optimizer.predictor.llm.get_token_count()
        meta_counts = (
            optimizer.meta_llm.get_token_count()
            if hasattr(optimizer, "meta_llm")
            else {}
        )
        controller = getattr(optimizer, "budget_controller", None)
        budget = controller.summary() if controller is not None else {}
        for prompt, score in zip(optimizer.prompts, optimizer.scores):
            blocks = _get_blocks(optimizer, prompt)
            record = _get_fairness_record(optimizer, prompt, blocks)
            vector = np.asarray(score, dtype=float).reshape(-1).tolist()
            quality = record.get("quality")
            cost = record.get("cost")
            fairness = record.get("fairness")
            if quality is None and vector:
                quality = vector[0]
            if cost is None and len(vector) >= 2:
                cost = -vector[1]
            if fairness is None and len(vector) >= 3:
                fairness = -vector[2]
            rows.append(
                {
                    "prompt": prompt.construct_prompt(),
                    "instruction": prompt.instruction,
                    "few_shots_json": json.dumps(
                        list(prompt.few_shots), ensure_ascii=False
                    ),
                    "downstream_template": prompt.downstream_template,
                    "objective_vector": vector,
                    "score": vector
                    if len(vector) > 1
                    else vector[0]
                    if vector
                    else None,
                    "quality": quality,
                    "cost": cost,
                    "fairness": fairness,
                    "fairness_ready": record.get("fairness_ready"),
                    "fairness_diagnostics_json": json.dumps(
                        json_safe(record.get("diagnostics", {})),
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "group_support_json": json.dumps(
                        json_safe(record.get("support", {})),
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "is_incumbent": _get_incumbancy(optimizer, prompt),
                    "evaluated_on_blocks": json.dumps(blocks),
                    "input_tokens_downstream": int(
                        downstream_counts.get("input_tokens", 0)
                    ),
                    "output_tokens_downstream": int(
                        downstream_counts.get("output_tokens", 0)
                    ),
                    "total_tokens_downstream": int(
                        downstream_counts.get("total_tokens", 0)
                    ),
                    "input_tokens_meta": int(meta_counts.get("input_tokens", 0)),
                    "output_tokens_meta": int(meta_counts.get("output_tokens", 0)),
                    "total_tokens_meta": int(meta_counts.get("total_tokens", 0)),
                    "requested_budget_downstream": budget.get(
                        "requested_budget_tokens"
                    ),
                    "remaining_budget_downstream": budget.get("remaining_tokens"),
                    "budget_utilization": budget.get("budget_utilization"),
                    "strict_budget_enabled": budget.get("strict_budget_enabled", False),
                    "near_budget_mode": bool(
                        controller is not None and controller.near_budget_mode
                    ),
                    "budget_stopping_reason": budget.get("stopping_reason"),
                    "time": timestamp,
                    "step": self.step,
                }
            )
        frame = pd.DataFrame(rows)
        step_path = self.step_dir / f"step_{self.step:06d}.parquet"
        frame.to_parquet(step_path, index=False)
        if self.path.exists():
            existing = pd.read_parquet(self.path)
            frame = pd.concat([existing, frame], ignore_index=True)
        temporary = self.path.with_suffix(".parquet.tmp")
        frame.to_parquet(temporary, index=False)
        temporary.replace(self.path)
        if hasattr(optimizer, "runhistory"):
            _dump_runhistory(optimizer.runhistory, self.runhistory_path, self.step)
        return True


def _get_incumbancy(optimizer, prompt) -> bool:
    if isinstance(optimizer, ParetoPromptOptimizer) or hasattr(optimizer, "incumbents"):
        return prompt in getattr(optimizer, "incumbents", [])
    return bool(optimizer.prompts and prompt == optimizer.prompts[0])


def _get_blocks(optimizer, prompt) -> list[int]:
    if hasattr(optimizer, "_get_evaluated_blocks"):
        blocks = optimizer._get_evaluated_blocks([prompt])[prompt]
    elif hasattr(optimizer.task, "get_evaluated_blocks"):
        blocks = optimizer.task.get_evaluated_blocks([prompt])[prompt]
    else:
        blocks = []
    if not blocks and getattr(optimizer.task, "eval_strategy", None) == "full":
        blocks = list(range(int(getattr(optimizer.task, "n_blocks", 0))))
    return sorted({int(value) for value in blocks})


def _get_fairness_record(optimizer, prompt, blocks: list[int]) -> Dict[str, Any]:
    if hasattr(optimizer, "get_fairness_record"):
        return dict(optimizer.get_fairness_record(prompt, blocks=blocks or None))
    if hasattr(optimizer.task, "get_last_record"):
        return dict(optimizer.task.get_last_record(prompt))
    return {}


def _dump_runhistory(runhistory, path: Path, step: int) -> None:
    payload: Dict[str, Dict[str, Any]] = {}
    for (prompt_str, block_idx), (objective_vector, eval_step) in runhistory.items():
        block_key = f"block_{block_idx}"
        payload.setdefault(block_key, {})[prompt_str] = {
            "objectives": np.asarray(objective_vector, dtype=float).tolist(),
            "eval_step": int(eval_step),
        }
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            {"last_callback_step": step, "blocks": payload},
            handle,
            indent=2,
            sort_keys=True,
        )
    temporary.replace(path)
