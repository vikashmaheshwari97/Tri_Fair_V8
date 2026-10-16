"""Full-evaluation three-objective NSGA-II prompt-optimization baseline."""

from __future__ import annotations
from collections.abc import Sequence
from typing import Dict, Optional
import numpy as np
from promptolution.utils.prompt import Prompt
from tri_fair.optimizers.nsga2_core import NSGA2PromptOptimizer
from tri_fair.runtime.checkpointing import ResumableOptimizerMixin
from tri_fair.data.task import FairnessEvalResult, FairnessTask
from tri_fair.search.mutation import (
    DEFAULT_MUTATION_WEIGHTS,
    generate_objective_aware_challengers,
)


class NSGA2Fair(ResumableOptimizerMixin, NSGA2PromptOptimizer):
    """NSGA-II-PO with quality, inference cost, and unfairness objectives.

    Unlike Tri-Fair, every offspring is evaluated on the complete development
    set before environmental selection.  It therefore isolates the benefit of
    fairness-aware intensification and budget allocation.
    """

    method_version = "8.0-nsga2-full-evaluation-po-fair"
    supports_multi_objective = True

    def __init__(
        self,
        *args,
        objective_aware_variation: bool = True,
        objective_mutation_weights: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> None:
        self.objective_aware_variation = bool(objective_aware_variation)
        self.objective_mutation_weights = dict(
            objective_mutation_weights or DEFAULT_MUTATION_WEIGHTS
        )
        super().__init__(*args, **kwargs)
        if not isinstance(self.task, FairnessTask):
            raise TypeError("NSGA2Fair requires FairnessTask")
        self.n_objectives = 3
        self.current_step = 0

    def _generate_challengers(self):
        if not self.objective_aware_variation:
            return super()._generate_challengers()
        return generate_objective_aware_challengers(self)

    def _get_objective_vectors(self, result: FairnessEvalResult) -> np.ndarray:
        quality = np.asarray(result.agg_scores, dtype=float).reshape(-1, 1)
        cost = (
            self.cost_per_input_token * np.asarray(result.agg_input_tokens, dtype=float)
            + self.cost_per_output_token
            * np.asarray(result.agg_output_tokens, dtype=float)
        ).reshape(-1, 1)
        unfairness = np.asarray(result.fairness_loss, dtype=float).reshape(-1, 1)
        return np.hstack([quality, -cost, -unfairness])

    def _pre_optimization_loop(self) -> None:
        self.current_step = 0
        super()._pre_optimization_loop()
        self._assert_fairness_ready(self.prompts)

    def _step(self):
        self.current_step += 1
        prompts = super()._step()
        self._assert_fairness_ready(prompts)
        return prompts

    def _assert_fairness_ready(self, prompts: Sequence[Prompt]) -> None:
        result = self.task.evaluate(list(prompts), self.predictor, eval_strategy="full")
        not_ready = [
            prompt.construct_prompt()[:80]
            for prompt, ready in zip(prompts, result.fairness_ready)
            if not bool(ready)
        ]
        if not_ready:
            raise RuntimeError(
                f"Full development evaluation did not meet fairness support requirements for {len(not_ready)} prompts. Check the manifest/block sampler."
            )

    def get_fairness_record(
        self, prompt: Prompt, blocks: Optional[Sequence[int]] = None
    ) -> Dict:
        result = self.task.evaluate([prompt], self.predictor, eval_strategy="full")
        cost = float(
            self.cost_per_input_token * result.agg_input_tokens[0]
            + self.cost_per_output_token * result.agg_output_tokens[0]
        )
        record = self.task.get_last_record(prompt)
        record["cost"] = cost
        return record
