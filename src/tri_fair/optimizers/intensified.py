"""MO-CAPO-inspired progressive screening for the three-objective Tri-Fair search.

Incumbents are fully measured anchors. Challengers visit random shared blocks;
only survivors receive complete coverage and enter the measured archive. This
is an adaptation of Algorithm 2, not a reproduction of MO-CAPO's mixed-fidelity
population, weaker-dominance survival or incumbent-advancement policy.
"""
from __future__ import annotations

import json
from collections import Counter
import numpy as np

from tri_fair.optimizers.directional import DirectionalTriFairOptimizer
from tri_fair.optimizers.repaired import _digest
from tri_fair.runtime.budget import BudgetExhausted


def dominates(left, right, margins=(0.0, 0.0, 0.0)):
    """Maximization Pareto dominance, optionally requiring practical margins."""
    left, right = np.asarray(left), np.asarray(right)
    margins = np.asarray(margins)
    return bool(np.isfinite([left, right]).all()
                and np.all(left >= right + margins)
                and np.any(left > right + margins))


class IntensifiedTriFairOptimizer(DirectionalTriFairOptimizer):
    method_version = "9.1-shared-block-progressive-intensification"
    _adaptive_fields = DirectionalTriFairOptimizer._adaptive_fields + (
        "race_round", "race_counts", "race_trace",
    )

    def __init__(self, *args, enable_intensification=True,
                 intensification_min_blocks=2,
                 intensification_margins=(0.01, 0.5, 0.01), **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_intensification = bool(enable_intensification)
        self.intensification_min_blocks = int(intensification_min_blocks)
        self.intensification_margins = tuple(float(x) for x in intensification_margins)
        if (self.intensification_min_blocks < 1
                or len(self.intensification_margins) != 3
                or not np.isfinite(self.intensification_margins).all()
                or min(self.intensification_margins) < 0):
            raise ValueError("Invalid progressive intensification settings")
        self.race_round = 0
        self.race_counts = Counter()
        self.race_trace = []
        self._race_order = []

    def _estimate_complete(self, prompt):
        # The inherited retry/admission loop reserves only the next block for
        # this method. The same hard controller guards every subsequent block.
        if not self.enable_intensification:
            return super()._estimate_complete(prompt)
        controller = getattr(self, "budget_controller", None)
        if controller is None:
            return None
        key = prompt.construct_prompt()
        for block in self._race_order:
            missing = [i for i in self.task.blocks[block]
                       if (key, str(self.task.ids[i])) not in self.task.pred_cache]
            if missing:
                return controller.estimate_call(
                    inputs=[f"{key}\n{self.task.xs[i]}" for i in missing])
        return 0

    def _fill_log(self, row):
        row = dict(row)
        if self.enable_intensification:
            row["admission_scope"] = "next_missing_shared_block"
            if row.get("reason") == "next_complete_candidate_exceeds_budget":
                row["reason"] = "next_shared_block_exceeds_budget"
        super()._fill_log(row)

    def _filled_challengers(self):
        try:
            return super()._filled_challengers()
        except BudgetExhausted as error:
            if not self.enable_intensification or error.reason != "next_complete_candidate_exceeds_budget":
                raise
            controller = getattr(self, "budget_controller", None)
            if controller is not None and controller.stopping_reason == error.reason:
                controller.stopping_reason = "next_shared_block_exceeds_budget"
                controller.stopping_operation = "v91_observed_shared_blocks"
            raise BudgetExhausted(operation="v91_observed_shared_blocks",
                estimated_tokens=error.estimated_tokens,
                remaining_tokens=error.remaining_tokens,
                reason="next_shared_block_exceeds_budget") from error

    def _race_event(self, **row):
        row = dict(step=self.current_step, split="search", **row)
        self.race_trace = (self.race_trace + [row])[-128:]
        if self._fill_directory is not None:
            with (self._fill_directory / "intensification.jsonl").open(
                    "a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")

    def _shared_vectors(self, prompts, blocks):
        # Recompute metrics on the UNION of rows, never average block F1/gaps.
        # FairnessTask reuses predictions and token counts from its row cache.
        result = self.task.evaluate(prompts, self.predictor, block_idx=blocks)
        return self._get_objective_vectors(result), np.asarray(result.fairness_ready)

    def _intensify(self, child):
        anchors = list(self.incumbents)
        if not anchors:
            raise RuntimeError("Intensification requires measured incumbents")
        for prompt in anchors:
            if set(self.task.get_evaluated_blocks([prompt])[prompt]) != set(range(self.task.n_blocks)):
                raise RuntimeError("A v9.1 incumbent lacks complete search coverage")
        blocks, last_comparison = [], np.full(3, -np.inf)
        scale = np.array([1.0, self.task.fairness_kwargs[
            "normalization_cost_upper_bound"], 1.0])
        try:
            for block in self._race_order:
                blocks.append(int(block))
                vectors, ready = self._shared_vectors(anchors + [child], blocks)
                self.race_counts["challenger_blocks"] += 1
                complete = len(blocks) == self.task.n_blocks
                enough = len(blocks) >= min(self.intensification_min_blocks, self.task.n_blocks)
                current = vectors[-1]
                # Algorithm 2 compares when the challenger improves since the
                # last comparison; full coverage always proceeds to survival.
                improved = bool(np.all(current >= last_comparison)
                                and np.any(current > last_comparison))
                if complete:
                    self._race_event(status="complete", child_sha256=_digest(child),
                                     blocks=blocks.copy(), objectives=current.tolist())
                    return True
                if not enough or not ready[-1] or not np.isfinite(current).all():
                    self.race_counts["deferred_support_checks"] += 1
                    continue
                if not improved:
                    continue
                last_comparison = current.copy()
                eligible = np.flatnonzero(ready[:-1] & np.isfinite(vectors[:-1]).all(axis=1))
                if not len(eligible):
                    continue
                nearest = int(eligible[np.argmin(np.linalg.norm(
                    (vectors[eligible] - current) / scale, axis=1))])
                reject = dominates(vectors[nearest], current, self.intensification_margins)
                self.race_counts["comparisons"] += 1
                self._race_event(status="rejected" if reject else "continue",
                                 child_sha256=_digest(child), blocks=blocks.copy(),
                                 incumbent_sha256=_digest(anchors[nearest]),
                                 objectives=current.tolist(),
                                 incumbent_objectives=vectors[nearest].tolist(),
                                 fairness_ready=True, confidence_interval=False)
                if reject:
                    self.race_counts["early_rejections"] += 1
                    self.race_counts["unevaluated_blocks_after_rejection"] += self.task.n_blocks - len(blocks)
                    return False
        except BudgetExhausted:
            self.race_counts["budget_interrupted_challengers"] += 1
            self._race_event(status="budget_interrupted", child_sha256=_digest(child),
                             requested_blocks=blocks.copy(), promoted=False)
            raise
        raise RuntimeError("Empty evaluation block schedule")

    def _credit_rejection(self, child, tokens):
        # A partial vector is NOT a full-search HV observation. Rejections get
        # zero strategy credit, with their scope explicit in a separate log.
        _, parent, mode, _, _ = self._pending_v89.pop(child.construct_prompt())
        self.strategy_state[mode]["evaluations"] += 1
        self.strategy_updates += 1
        self.operator_evaluations[mode] += 1
        self._race_event(status="strategy_rejection_credit", mode=mode,
                         child_sha256=_digest(child), parent_sha256=_digest(parent),
                         strategy_reward=0.0, evaluation_tokens=tokens,
                         full_search_observation=False)

    def _step(self):
        if not self.enable_intensification:
            return super()._step()
        self.current_step += 1
        self._race_order = np.random.default_rng(
            self.task.seed + 1000003 + self.race_round).permutation(self.task.n_blocks).tolist()
        self.race_round += 1
        children = self._filled_challengers()
        self._last_candidate_token_deltas = {}
        # Commit each completed child immediately; an interrupted later child
        # cannot remove an already measured improvement from the population.
        for child in children:
            key = child.construct_prompt()
            start = self.predictor.llm.get_token_count()["total_tokens"]
            survived = self._intensify(child)
            spent = int(self.predictor.llm.get_token_count()["total_tokens"] - start)
            self._last_candidate_token_deltas[key] = spent
            if not survived:
                self._credit_rejection(child, spent)
                continue
            entry = self._pending_v89[key]
            self._pending_v89[key] = (*entry[:4], np.array([
                self._loss(self._guard_record(p)) for p in self.candidate_archive]))
            self._assert_fairness_ready([child])
            candidates = list({p.construct_prompt(): p for p in self.prompts + [child]}.values())
            result = self.task.evaluate(candidates, self.predictor, eval_strategy="evaluated")
            self.prompts = self._environmental_selection(candidates, self._get_objective_vectors(result))
            result = self.task.evaluate(self.prompts, self.predictor, eval_strategy="evaluated")
            vectors = self._get_objective_vectors(result)
            fronts = self._non_dominated_sort(vectors)
            self.incumbents = [self.prompts[i] for i in fronts[0]]
            self.non_incumbents = [self.prompts[i] for f in fronts[1:] for i in f]
            self.scores = vectors.tolist()
            self._precompute_selection_metrics()
            self.fill_counts["evaluated_candidates"] += 1
            self.race_counts["completed_challengers"] += 1
            # Preserve pending metadata for the other candidates in this batch.
            pending = self._pending_v89.copy()
            self._on_filled_evaluation([child], spent)
            self._pending_v89 = {k: v for k, v in pending.items() if k != key}
        self._pending_v89 = {}
        return self.prompts

    def search_diagnostics(self):
        return dict(super().search_diagnostics(), active_method="Tri-Fair-v9.1",
                    progressive_intensification=self.enable_intensification,
                    shared_budget_protocol=("v91-next-block-admission-hard-total-cap"
                        if self.enable_intensification else "v89-novel-complete-candidate-retries"),
                    intensification_counts=dict(self.race_counts),
                    recent_intensification=self.race_trace,
                    comparison="nearest incumbent on identical pooled search blocks",
                    normalization="fixed quality/cost/unfairness scales",
                    initial_population="complete search coverage",
                    partial_candidates_enter_population=False,
                    incumbent_advancement="unnecessary: all incumbents fully evaluated",
                    confidence_interval=False, exact_mocapo_reproduction=False)

    def optimizer_state_dict(self):
        state = super().optimizer_state_dict()
        state["intensification_configuration"] = [self.enable_intensification,
            self.intensification_min_blocks, list(self.intensification_margins)]
        return state

    def load_optimizer_state_dict(self, state):
        if state.get("intensification_configuration") != self.optimizer_state_dict()["intensification_configuration"]:
            raise ValueError("Intensification checkpoint configuration mismatch")
        super().load_optimizer_state_dict(state)
        self._guard_cache = {}
        self._race_order = []
