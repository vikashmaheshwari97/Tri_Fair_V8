"""Objectives responsibilities for the current Tri-Fair search."""

from __future__ import annotations
from typing import Dict, List, Optional, Union
import numpy as np
from promptolution.utils.prompt import Prompt
from tri_fair.data.task import FairnessEvalResult
from collections.abc import Sequence

MODES = ("quality", "fairness", "cost", "balanced", "explore")


class PooledObjectiveMixin:
    def _update_global_bounds(self, vectors: np.ndarray) -> None:
        """Track only fully finite objective vectors for stable normalization."""
        if self.fixed_objective_bounds is not None:
            lower, upper = self.fixed_objective_bounds
            self.global_min_bounds = lower.copy()
            self.global_max_bounds = upper.copy()
            return
        values = np.atleast_2d(np.asarray(vectors, dtype=float))
        values = values[np.all(np.isfinite(values), axis=1)]
        if values.size == 0:
            return
        current_min = np.min(values, axis=0)
        current_max = np.max(values, axis=0)
        if self.global_min_bounds is None:
            self.global_min_bounds = current_min
            self.global_max_bounds = current_max
        else:
            self.global_min_bounds = np.minimum(self.global_min_bounds, current_min)
            self.global_max_bounds = np.maximum(self.global_max_bounds, current_max)

    def _get_objective_vectors(self, result: FairnessEvalResult) -> np.ndarray:
        quality = np.asarray(result.agg_scores, dtype=float).reshape(-1, 1)
        cost = (
            self.cost_per_input_token * np.asarray(result.agg_input_tokens, dtype=float)
            + self.cost_per_output_token
            * np.asarray(result.agg_output_tokens, dtype=float)
        ).reshape(-1, 1)
        unfairness = np.asarray(result.fairness_loss, dtype=float).reshape(-1, 1)
        return np.hstack([quality, -cost, -unfairness])

    def _store_records(
        self,
        prompts: Sequence[Prompt],
        blocks: Sequence[int],
        result: FairnessEvalResult,
        vectors: np.ndarray,
    ) -> None:
        key_blocks = tuple(sorted(set((int(value) for value in blocks))))
        for index, prompt in enumerate(prompts):
            key = (prompt.construct_prompt(), key_blocks)
            self.aggregate_records[key] = {
                "quality": float(result.agg_scores[index]),
                "cost": float(-vectors[index, 1]),
                "fairness": float(result.fairness_loss[index]),
                "fairness_ready": bool(result.fairness_ready[index]),
                "diagnostics": result.fairness_diagnostics[index],
                "support": result.fairness_support[index],
                "blocks": list(key_blocks),
                "mean_input_tokens": float(result.agg_input_tokens[index]),
                "mean_output_tokens": float(result.agg_output_tokens[index]),
            }

    def _get_block_vectors(
        self, prompts: List[Prompt], block_idx: Union[int, List[int]]
    ) -> np.ndarray:
        """Evaluate requested blocks and aggregate fairness exactly over their union."""
        block_indices = [block_idx] if isinstance(block_idx, int) else list(block_idx)
        if not block_indices:
            raise ValueError("block_idx cannot be empty")
        block_indices = sorted(set((int(value) for value in block_indices)))
        prompt_strings = [prompt.construct_prompt() for prompt in prompts]
        for block in block_indices:
            missing_prompts: list[Prompt] = []
            missing_positions: list[int] = []
            for index, prompt_string in enumerate(prompt_strings):
                if (prompt_string, block) not in self.runhistory:
                    missing_prompts.append(prompts[index])
                    missing_positions.append(index)
            if missing_prompts:
                result = self.task.evaluate(
                    missing_prompts, self.predictor, block_idx=block
                )
                vectors = self._get_objective_vectors(result)
                self._update_global_bounds(vectors)
                self._store_records(missing_prompts, [block], result, vectors)
                for local_index, original_index in enumerate(missing_positions):
                    key = (prompt_strings[original_index], block)
                    self.runhistory[key] = (vectors[local_index], self.current_step)
        aggregate = self.task.evaluate(prompts, self.predictor, block_idx=block_indices)
        vectors = self._get_objective_vectors(aggregate)
        self._update_global_bounds(vectors)
        self._store_records(prompts, block_indices, aggregate, vectors)
        return vectors

    def _record_for(self, prompt: Prompt, blocks: Sequence[int]) -> Dict:
        key = (
            prompt.construct_prompt(),
            tuple(sorted(set((int(value) for value in blocks)))),
        )
        if key not in self.aggregate_records:
            self._get_block_vectors([prompt], list(blocks))
        return dict(self.aggregate_records[key])

    def get_fairness_record(
        self, prompt: Prompt, blocks: Optional[Sequence[int]] = None
    ) -> Dict:
        if blocks is None:
            blocks = self._get_evaluated_blocks([prompt])[prompt]
        if not blocks:
            return {}
        return self._record_for(prompt, blocks)

    def _ready_incumbent_vectors(
        self, incumbents: Sequence[Prompt], blocks: Sequence[int]
    ) -> tuple[list[Prompt], np.ndarray]:
        vectors = self._get_block_vectors(list(incumbents), list(blocks))
        ready_prompts: list[Prompt] = []
        ready_vectors: list[np.ndarray] = []
        for prompt, vector in zip(incumbents, vectors):
            if self._record_for(prompt, blocks)["fairness_ready"]:
                ready_prompts.append(prompt)
                ready_vectors.append(vector)
        if not ready_vectors:
            return ([], np.empty((0, self.n_objectives), dtype=float))
        return (ready_prompts, np.vstack(ready_vectors))
