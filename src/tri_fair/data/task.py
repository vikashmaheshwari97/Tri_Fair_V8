"""Group-aware task implementation for Tri-Fair.

Unlike ordinary classification metrics, fairness objectives are non-decomposable:
they must be computed over a set containing protected groups.  ``FairnessTask``
therefore caches predictions per example, then recomputes quality and fairness
exactly over any requested union of blocks without additional LLM calls.
"""

from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from promptolution.tasks.base_task import BaseTask, EvalStrategy
from promptolution.utils.prompt import Prompt
from promptolution.utils.token_counter import get_token_counter
from tri_fair.fairness.base import FairnessMetric, FairnessMetricResult, json_safe
from tri_fair.fairness.bbq import compute_bbq_fairness
from tri_fair.fairness.bios import compute_bios_fairness
from tri_fair.fairness.civilcomments import compute_civilcomments_fairness
from tri_fair.fairness.statistical_metrics import (
    compute_bbq_statistical_fairness,
    compute_bios_statistical_tpr_gap,
    compute_civil_equalized_odds,
)


@dataclass
class FairnessEvalResult:
    scores: np.ndarray
    agg_scores: np.ndarray
    sequences: np.ndarray
    input_tokens: np.ndarray
    output_tokens: np.ndarray
    agg_input_tokens: np.ndarray
    agg_output_tokens: np.ndarray
    fairness_loss: np.ndarray
    fairness_ready: np.ndarray
    fairness_diagnostics: List[Dict[str, Any]]
    fairness_support: List[Dict[str, int]]


_METRICS: Dict[str, FairnessMetric] = {
    "bbq_bias": compute_bbq_fairness,
    "bios_tpr_gap": compute_bios_fairness,
    "civil_worst_group": compute_civilcomments_fairness,
    "bbq_bias_statistical": compute_bbq_statistical_fairness,
    "bios_tpr_gap_statistical": compute_bios_statistical_tpr_gap,
    "civil_equalized_odds": compute_civil_equalized_odds,
}


def resolve_fairness_metric(name: str) -> FairnessMetric:
    try:
        return _METRICS[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown fairness metric {name!r}; available: {sorted(_METRICS)}"
        ) from error


class FairnessTask(BaseTask):
    """Classification task with a set-level fairness objective."""

    def __init__(
        self,
        *,
        df: pd.DataFrame,
        task_description: str,
        classes: Sequence[str],
        quality_metric: str,
        fairness_metric: FairnessMetric,
        fairness_metric_name: str,
        protected_columns: Sequence[str],
        metadata_columns: Sequence[str],
        min_group_count: int,
        fairness_kwargs: Optional[Dict[str, Any]] = None,
        block_ids: Optional[Sequence[Sequence[str]]] = None,
        eval_strategy: EvalStrategy = "full",
        seed: int = 42,
        id_column: str = "_manifest_id",
        x_column: str = "input",
        y_column: str = "target",
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.task_description = task_description
        self.classes = [str(value).casefold() for value in classes]
        if any((value != value.lower() for value in self.classes)):
            raise ValueError("FairnessTask classes must be lower-case")
        self.x_column = x_column
        self.y_column = y_column
        self.id_column = id_column
        self.quality_metric_name = quality_metric
        self.fairness_metric = fairness_metric
        self.fairness_metric_name = fairness_metric_name
        self.protected_columns = list(protected_columns)
        self.metadata_columns = list(dict.fromkeys(metadata_columns))
        self.min_group_count = int(min_group_count)
        self.fairness_kwargs = dict(fairness_kwargs or {})
        self.eval_strategy = eval_strategy
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.task_type = "fairness"
        self.has_y = True
        required = {
            x_column,
            y_column,
            id_column,
            *self.metadata_columns,
            *self.protected_columns,
        }
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(
                f"FairnessTask dataframe is missing columns: {sorted(missing)}"
            )
        if self.df[id_column].astype(str).duplicated().any():
            duplicates = (
                self.df.loc[self.df[id_column].astype(str).duplicated(), id_column]
                .head()
                .tolist()
            )
            raise ValueError(
                f"FairnessTask IDs must be unique; duplicates include {duplicates}"
            )
        self.xs = self.df[x_column].astype(str).tolist()
        self.ys = self.df[y_column].astype(str).str.casefold().tolist()
        self.ids = self.df[id_column].astype(str).tolist()
        self._id_to_position = {value: index for index, value in enumerate(self.ids)}
        if block_ids is None:
            self.blocks: list[list[int]] = [list(range(len(self.df)))]
        else:
            self.blocks = []
            for block_number, ids in enumerate(block_ids):
                positions: list[int] = []
                for value in ids:
                    value = str(value)
                    if value not in self._id_to_position:
                        raise ValueError(
                            f"Block {block_number} references unknown manifest ID {value!r}"
                        )
                    positions.append(self._id_to_position[value])
                if not positions:
                    raise ValueError(f"Block {block_number} is empty")
                self.blocks.append(positions)
            flattened = [position for block in self.blocks for position in block]
            if len(flattened) != len(set(flattened)):
                raise ValueError("Fairness blocks overlap")
            if set(flattened) != set(range(len(self.df))):
                raise ValueError(
                    "Fairness blocks must exactly cover the task dataframe"
                )
        self.n_blocks = len(self.blocks)
        self.n_subsamples = max((len(block) for block in self.blocks))
        self.block_idx = 0
        self.token_count_cache: Dict[Tuple[str, str], Tuple[float, float]] = {}
        self.pred_cache: Dict[Tuple[str, str], str] = {}
        self.sequence_cache: Dict[Tuple[str, str], str] = {}
        self.prompt_evaluated_blocks: Dict[Prompt, List[int]] = {}
        self.last_records: Dict[str, Dict[str, Any]] = {}
        self.eval_cache: Dict[Tuple[str, str], float] = {}
        self.seq_cache = self.sequence_cache

    def _evaluate(self, xs: List[str], ys: List[str], preds: List[str]) -> np.ndarray:
        return np.asarray(
            [
                float(str(y).casefold() == str(pred).casefold())
                for y, pred in zip(ys, preds)
            ]
        )

    def _positions_for_blocks(self, block_idx: Union[int, Sequence[int]]) -> list[int]:
        indices = [block_idx] if isinstance(block_idx, int) else list(block_idx)
        if not indices:
            raise ValueError("At least one block must be requested")
        positions: list[int] = []
        seen: set[int] = set()
        for index in indices:
            if index < 0 or index >= self.n_blocks:
                raise IndexError(f"Block index {index} outside [0, {self.n_blocks})")
            for position in self.blocks[index]:
                if position not in seen:
                    positions.append(position)
                    seen.add(position)
        return positions

    def _resolve_positions(
        self,
        *,
        eval_strategy: Optional[EvalStrategy],
        block_idx: Optional[Union[int, Sequence[int]]],
    ) -> tuple[list[int], list[int]]:
        strategy = eval_strategy or self.eval_strategy
        if block_idx is not None:
            block_indices = (
                [block_idx] if isinstance(block_idx, int) else list(block_idx)
            )
            return (
                self._positions_for_blocks(block_indices),
                [int(value) for value in block_indices],
            )
        if strategy in {"full", "evaluated"}:
            return (list(range(len(self.df))), list(range(self.n_blocks)))
        if strategy == "sequential_block":
            return (list(self.blocks[self.block_idx]), [self.block_idx])
        if strategy == "random_block":
            selected = int(self.rng.integers(0, self.n_blocks))
            return (list(self.blocks[selected]), [selected])
        if strategy == "subsample":
            size = min(self.n_subsamples, len(self.df))
            positions = (
                self.rng.choice(len(self.df), size=size, replace=False)
                .astype(int)
                .tolist()
            )
            return (positions, [])
        raise ValueError(f"Unsupported evaluation strategy {strategy!r}")

    def _quality(self, y_true: Sequence[str], y_pred: Sequence[str]) -> float:
        if self.quality_metric_name == "accuracy":
            return float(accuracy_score(y_true, y_pred))
        if self.quality_metric_name == "macro_f1":
            return float(
                f1_score(
                    y_true,
                    y_pred,
                    labels=self.classes,
                    average="macro",
                    zero_division=0,
                )
            )
        raise ValueError(f"Unsupported quality metric {self.quality_metric_name!r}")

    def evaluate(
        self,
        prompts: Union[Prompt, Sequence[Prompt]],
        predictor,
        system_prompts: Optional[Union[str, List[str]]] = None,
        eval_strategy: Optional[EvalStrategy] = None,
        block_idx: Optional[Union[int, Sequence[int]]] = None,
    ) -> FairnessEvalResult:
        prompt_list = [prompts] if isinstance(prompts, Prompt) else list(prompts)
        positions, evaluated_blocks = self._resolve_positions(
            eval_strategy=eval_strategy, block_idx=block_idx
        )
        if not prompt_list:
            raise ValueError("prompts cannot be empty")
        prompts_to_evaluate: list[str] = []
        xs_to_evaluate: list[str] = []
        missing_keys: list[tuple[str, str]] = []
        for prompt in prompt_list:
            prompt_string = prompt.construct_prompt()
            for position in positions:
                key = (prompt_string, self.ids[position])
                if key in self.pred_cache:
                    continue
                prompts_to_evaluate.append(prompt_string)
                xs_to_evaluate.append(self.xs[position])
                missing_keys.append(key)
        if missing_keys:
            preds, sequences = predictor.predict(
                prompts=prompts_to_evaluate,
                xs=xs_to_evaluate,
                system_prompts=system_prompts,
            )
            if len(preds) != len(missing_keys) or len(sequences) != len(missing_keys):
                raise RuntimeError(
                    "Predictor returned a different number of outputs than requested"
                )
            for key, pred, sequence in zip(missing_keys, preds, sequences):
                self.pred_cache[key] = str(pred).casefold()
                self.sequence_cache[key] = str(sequence)
                self.eval_cache[key] = float("nan")
        token_counter = get_token_counter(predictor.llm)
        y_true = [self.ys[position] for position in positions]
        metadata = self.df.iloc[positions][self.metadata_columns].reset_index(drop=True)
        score_rows: list[np.ndarray] = []
        sequence_rows: list[list[str]] = []
        input_rows: list[np.ndarray] = []
        output_rows: list[np.ndarray] = []
        quality_values: list[float] = []
        fairness_values: list[float] = []
        fairness_ready: list[bool] = []
        fairness_diagnostics: list[Dict[str, Any]] = []
        fairness_support: list[Dict[str, int]] = []
        for prompt in prompt_list:
            prompt_string = prompt.construct_prompt()
            preds = [
                self.pred_cache[prompt_string, self.ids[position]]
                for position in positions
            ]
            sequences = [
                self.sequence_cache[prompt_string, self.ids[position]]
                for position in positions
            ]
            scores = self._evaluate(
                [self.xs[position] for position in positions], y_true, preds
            )
            quality = self._quality(y_true, preds)
            fairness_result: FairnessMetricResult = self.fairness_metric(
                y_true,
                preds,
                metadata,
                min_group_count=self.min_group_count,
                **self.fairness_kwargs,
            )
            input_counts = np.asarray(
                [
                    token_counter(f"{prompt_string}\n{self.xs[position]}")
                    for position in positions
                ],
                dtype=float,
            )
            raw_outputs: list[str] = []
            for sequence, position in zip(sequences, positions):
                prefix = f"{self.xs[position]}\n"
                raw_outputs.append(
                    str(sequence)[len(prefix) :]
                    if str(sequence).startswith(prefix)
                    else str(sequence)
                )
            output_counts = np.asarray(
                [token_counter(output) for output in raw_outputs], dtype=float
            )
            for position, n_in, n_out in zip(positions, input_counts, output_counts):
                self.token_count_cache[prompt_string, self.ids[position]] = (
                    float(n_in),
                    float(n_out),
                )
            score_rows.append(scores)
            sequence_rows.append(sequences)
            input_rows.append(input_counts)
            output_rows.append(output_counts)
            quality_values.append(quality)
            fairness_values.append(float(fairness_result.loss))
            fairness_ready.append(bool(fairness_result.ready))
            diagnostics = json_safe(fairness_result.diagnostics)
            support = {
                str(key): int(value) for key, value in fairness_result.support.items()
            }
            fairness_diagnostics.append(diagnostics)
            fairness_support.append(support)
            self.last_records[prompt_string] = {
                "quality": quality,
                "fairness": float(fairness_result.loss),
                "fairness_ready": bool(fairness_result.ready),
                "diagnostics": diagnostics,
                "support": support,
                "blocks": evaluated_blocks,
                "n_examples": len(positions),
                "mean_input_tokens": float(np.mean(input_counts)),
                "mean_output_tokens": float(np.mean(output_counts)),
            }
            if evaluated_blocks:
                current = self.prompt_evaluated_blocks.setdefault(prompt, [])
                current.extend(
                    (index for index in evaluated_blocks if index not in current)
                )
                current.sort()
        score_array = np.vstack(score_rows)
        input_array = np.vstack(input_rows)
        output_array = np.vstack(output_rows)
        return FairnessEvalResult(
            scores=score_array,
            agg_scores=np.asarray(quality_values, dtype=float),
            sequences=np.asarray(sequence_rows, dtype=object),
            input_tokens=input_array,
            output_tokens=output_array,
            agg_input_tokens=np.mean(input_array, axis=1),
            agg_output_tokens=np.mean(output_array, axis=1),
            fairness_loss=np.asarray(fairness_values, dtype=float),
            fairness_ready=np.asarray(fairness_ready, dtype=bool),
            fairness_diagnostics=fairness_diagnostics,
            fairness_support=fairness_support,
        )

    def cached_prediction_frame(
        self, prompt: Prompt, blocks: Sequence[int]
    ) -> pd.DataFrame:
        """Read cached evidence only; missing predictions are an explicit error."""
        positions = self._positions_for_blocks(sorted(set((int(b) for b in blocks))))
        key = prompt.construct_prompt()
        rows = []
        for position in positions:
            cache_key = (key, self.ids[position])
            if cache_key not in self.pred_cache:
                raise KeyError(f"uncached evidence for {self.ids[position]}")
            row = self.df.iloc[position].to_dict()
            n_in, n_out = self.token_count_cache[cache_key]
            row.update(_review_input_tokens=n_in, _review_output_tokens=n_out)
            row.update(
                _review_id=self.ids[position],
                _review_target=self.ys[position],
                _review_prediction=self.pred_cache[cache_key],
            )
            rows.append(row)
        return pd.DataFrame(rows).reset_index(drop=True)

    def objectives_from_frame(
        self, frame: pd.DataFrame, cost_in: float, cost_out: float
    ) -> np.ndarray:
        """Recompute nonlinear quality/fairness on a complete resampled union.

        Resampling never invokes a predictor or invents experimental observations.
        Cost uses the same cached raw-text token convention as task.evaluate.
        """
        truth = frame["_review_target"].astype(str).tolist()
        predictions = frame["_review_prediction"].astype(str).tolist()
        metric = self.fairness_metric(
            truth,
            predictions,
            frame[self.metadata_columns].reset_index(drop=True),
            min_group_count=self.min_group_count,
            **self.fairness_kwargs,
        )
        return np.asarray(
            [
                self._quality(truth, predictions),
                -float(
                    cost_in * frame["_review_input_tokens"].mean()
                    + cost_out * frame["_review_output_tokens"].mean()
                ),
                -float(metric.loss) if metric.ready else float("nan"),
            ]
        )

    def _fairness_memberships_for_position(self, position: int) -> list[str]:
        """Return metadata cells whose uncertainty is reduced by one row."""
        row = self.df.iloc[int(position)]
        label = str(self.ys[int(position)])
        metric = str(self.fairness_metric_name)
        if metric.startswith("bbq_bias"):
            return [f"{row['category']}/{row['context_condition']}"]
        if metric.startswith("bios_tpr_gap"):
            gender = "female" if int(row["gender"]) == 1 else "male"
            return [f"{label}/{gender}"]
        if metric in {"civil_worst_group", "civil_equalized_odds"}:
            memberships: list[str] = []
            for identity in self.protected_columns:
                present = int(row.get(identity, 0) or 0) == 1
                side = "present" if present else "absent"
                memberships.append(f"{identity}/{side}/{label}")
            return memberships
        return []

    def fairness_support_profile(self, block_indices: Sequence[int]) -> Dict[str, int]:
        """Count protected-group evidence supplied by a set of blocks."""
        profile: Dict[str, int] = {}
        if not block_indices:
            return profile
        for position in self._positions_for_blocks(list(block_indices)):
            for key in self._fairness_memberships_for_position(position):
                profile[key] = profile.get(key, 0) + 1
        return profile

    def select_fairness_block(
        self,
        candidate_blocks: Sequence[int],
        *,
        current_blocks: Sequence[int] = (),
        diagnostics: Optional[Mapping[str, Any]] = None,
    ) -> int:
        """Choose the block with the largest expected uncertainty reduction.

        The estimate is prediction-free for unseen blocks: it uses their protected
        group composition and upweights cells whose current Wilson intervals are
        widest. Ties are broken by the task RNG for reproducible seeds.
        """
        candidates = sorted({int(value) for value in candidate_blocks})
        if not candidates:
            raise ValueError("candidate_blocks cannot be empty")
        current = self.fairness_support_profile(list(current_blocks))
        diag = dict(diagnostics or {})
        raw_widths = diag.get("uncertainty_by_group") or {}
        widths = {
            str(key): float(value)
            for key, value in raw_widths.items()
            if value is not None
        }
        scores: Dict[int, float] = {}
        for block in candidates:
            added = self.fairness_support_profile([block])
            score = 0.0
            for key, increment in added.items():
                before = int(current.get(key, 0))
                after = before + int(increment)
                uncertainty = max(1e-06, float(widths.get(key, 1.0)))
                reduction = 1.0 / np.sqrt(before + 1.0) - 1.0 / np.sqrt(after + 1.0)
                score += uncertainty * reduction
                if before < self.min_group_count <= after:
                    score += uncertainty
            scores[block] = float(score)
        best = max(scores.values())
        tied = [block for block, score in scores.items() if np.isclose(score, best)]
        return int(self.rng.choice(tied))

    def get_evaluated_blocks(
        self, prompts: Union[Prompt, Sequence[Prompt]]
    ) -> Dict[Prompt, List[int]]:
        prompt_list = [prompts] if isinstance(prompts, Prompt) else list(prompts)
        return {
            prompt: list(self.prompt_evaluated_blocks.get(prompt, []))
            for prompt in prompt_list
        }

    def get_last_record(self, prompt: Prompt | str) -> Dict[str, Any]:
        key = prompt.construct_prompt() if isinstance(prompt, Prompt) else str(prompt)
        return dict(self.last_records.get(key, {}))

    def set_block_idx(self, idx: int) -> None:
        if idx < 0 or idx >= self.n_blocks:
            raise IndexError(idx)
        self.block_idx = idx

    def increment_block_idx(self) -> None:
        self.block_idx = (self.block_idx + 1) % self.n_blocks

    def reset_block_idx(self) -> None:
        self.block_idx = 0

    def pop_datapoints(
        self, n: Optional[int] = None, frac: Optional[float] = None
    ) -> pd.DataFrame:
        raise RuntimeError(
            "FairnessTask does not pop datapoints because that would invalidate fixed manifests. Pass the manifest-defined few-shot dataframe to the optimizer."
        )

    def activate_scalarized_objective(self) -> None:
        raise RuntimeError(
            "FairnessTask is intended for explicit multi-objective optimization"
        )
