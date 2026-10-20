"""Opt-in v8.9 proposal admission, shared by every comparator and Tri-Fair.

Selection and variation are delegated to the named optimizer. Empty, cached,
and unaffordable proposals are different outcomes. No inference is repeated to
pad expenditure. All admitted candidates retain complete search coverage.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from tri_fair.runtime.budget import BudgetExhausted
from tri_fair.optimizers.nsga2 import NSGA2Fair
from tri_fair.optimizers.baselines import NSGAIIIPOFair, SPEA2POFair, RandomSearchPOFair


class BudgetFillMixin:
    _fill_fields = ("fill_attempts", "fill_counts", "fill_trace")

    def __init__(self, *args, proposal_retry_limit=8, **kwargs):
        self.proposal_retry_limit = int(proposal_retry_limit)
        if not 1 <= self.proposal_retry_limit <= 16:
            raise ValueError("proposal_retry_limit must be 1-16")
        self.fill_attempts = 0
        self.fill_counts = Counter()
        self.fill_trace = []
        self._fill_directory = None
        super().__init__(*args, **kwargs)

    def set_artifact_directory(self, directory):
        self._fill_directory = Path(directory)
        self._fill_directory.mkdir(parents=True, exist_ok=True)
        parent = getattr(super(), "set_artifact_directory", None)
        if parent is not None:
            parent(directory)

    def _fill_log(self, row):
        row = dict(row, attempt=self.fill_attempts, step=self.current_step)
        self.fill_trace.append(row)
        self.fill_trace = self.fill_trace[-32:]
        if self._fill_directory is not None:
            with (self._fill_directory / "proposal_admission.jsonl").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")

    def _estimate_complete(self, prompt):
        """The same raw-input/output reserve convention as the hard controller."""
        controller = getattr(self, "budget_controller", None)
        key = prompt.construct_prompt()
        missing = [
            i
            for i, identifier in enumerate(self.task.ids)
            if (key, str(identifier)) not in self.task.pred_cache
        ]
        if controller is None:
            return 0 if not missing else None
        return (
            controller.estimate_call(
                inputs=[f"{key}\n{self.task.xs[i]}" for i in missing]
            )
            if missing
            else 0
        )

    def _filled_challengers(self):
        controller = getattr(self, "budget_controller", None)
        rejected_estimates = []
        for _ in range(self.proposal_retry_limit):
            # A reproducible stream of proposals, rather than resetting every
            # identical meta request to the same fixed sampling seed.
            params = getattr(self.meta_llm, "sampling_params", None)
            seed = int((self.task.seed + 1000003 + self.fill_attempts) % (2**31 - 1))
            if params is not None and hasattr(params, "seed"):
                params.seed = seed
            self.fill_attempts += 1
            proposed = list(self._generate_challengers())
            if not proposed:
                self.fill_counts["empty_proposal_batches"] += 1
                self._fill_log(dict(status="empty", meta_seed=seed))
                continue
            accepted, seen = [], set()
            available = controller.remaining_tokens if controller is not None else None
            for prompt in proposed:
                key = prompt.construct_prompt()
                estimate = self._estimate_complete(prompt)
                identity = hashlib.sha256(key.encode()).hexdigest()
                if key in seen or estimate == 0:
                    self.fill_counts["cached_or_duplicate_candidates"] += 1
                    self._fill_log(
                        dict(status="cached_or_duplicate", prompt_sha256=identity)
                    )
                    continue
                seen.add(key)
                if controller is not None and estimate > available:
                    rejected_estimates.append(estimate)
                    self.fill_counts["unaffordable_candidates"] += 1
                    self._fill_log(
                        dict(
                            status="unaffordable",
                            prompt_sha256=identity,
                            estimated_tokens=estimate,
                            remaining_tokens=available,
                        )
                    )
                    # Continue to later affordable children, not a costly prefix.
                    continue
                accepted.append(prompt)
                if controller is not None:
                    available -= estimate
                self.fill_counts["admitted_candidates"] += 1
                self._fill_log(
                    dict(
                        status="admitted",
                        prompt_sha256=identity,
                        estimated_tokens=estimate,
                        meta_seed=seed,
                    )
                )
            if accepted:
                return accepted
        if rejected_estimates:
            estimate = min(rejected_estimates)
            reason = "next_complete_candidate_exceeds_budget"
            if controller is not None:
                controller.request_stop(
                    reason=reason,
                    operation="v89_observed_complete_candidates",
                    estimated_tokens=estimate,
                )
        else:
            # This is a proposal failure, not a proof that the budget is exhausted.
            estimate, reason = 0, "proposal_retry_limit_no_novel_candidate"
            self.runtime_stop_reason = reason
        self._fill_log(
            dict(
                status="retry_limit",
                reason=reason,
                minimum_observed_rejected_estimate=min(
                    rejected_estimates, default=None
                ),
                global_infeasibility_proved=False,
            )
        )
        raise BudgetExhausted(
            operation="v89_proposal_retry_limit",
            estimated_tokens=estimate,
            remaining_tokens=controller.remaining_tokens if controller else 0,
            reason=reason,
        )

    def _step(self):
        """Full-generation evaluation, with original method-specific survival."""
        self.current_step += 1
        children = self._filled_challengers()
        before = self.predictor.llm.get_token_count()["total_tokens"]
        self._last_candidate_token_deltas = {}
        for child in children:
            start = self.predictor.llm.get_token_count()["total_tokens"]
            self.task.evaluate([child], self.predictor, eval_strategy="full")
            self._last_candidate_token_deltas[child.construct_prompt()] = int(
                self.predictor.llm.get_token_count()["total_tokens"] - start
            )
        candidates = list(
            {p.construct_prompt(): p for p in self.prompts + children}.values()
        )
        result = self.task.evaluate(
            candidates, self.predictor, eval_strategy="evaluated"
        )
        self._assert_fairness_ready(children)
        vectors = self._get_objective_vectors(result)
        self.prompts = self._environmental_selection(candidates, vectors)
        result = self.task.evaluate(
            self.prompts, self.predictor, eval_strategy="evaluated"
        )
        vectors = self._get_objective_vectors(result)
        fronts = self._non_dominated_sort(vectors)
        self.incumbents = [self.prompts[i] for i in fronts[0]]
        self.non_incumbents = [self.prompts[i] for f in fronts[1:] for i in f]
        self.scores = vectors.tolist()
        self._precompute_selection_metrics()
        self.fill_counts["evaluated_candidates"] += len(children)
        after = self.predictor.llm.get_token_count()["total_tokens"]
        hook = getattr(self, "_on_filled_evaluation", None)
        if hook is not None:
            hook(children, int(after - before))
        return self.prompts

    def search_diagnostics(self):
        method = getattr(super(), "search_diagnostics", None)
        result = method() if method is not None else {}
        return dict(
            result,
            shared_budget_protocol="v89-novel-complete-candidate-retries",
            proposal_retry_limit=self.proposal_retry_limit,
            proposal_attempts=self.fill_attempts,
            proposal_admission_counts=dict(self.fill_counts),
            recent_admission_events=self.fill_trace,
            holdout_feedback_used=False,
            exact_full_expenditure_guaranteed=False,
        )

    def optimizer_state_dict(self):
        state = super().optimizer_state_dict()
        state["budget_fill_configuration"] = self.proposal_retry_limit
        state["budget_fill_state"] = {k: getattr(self, k) for k in self._fill_fields}
        return state

    def load_optimizer_state_dict(self, state):
        if state.get("budget_fill_configuration") != self.proposal_retry_limit:
            raise ValueError("Budget-fill checkpoint configuration mismatch")
        super().load_optimizer_state_dict(state)
        for key in self._fill_fields:
            setattr(self, key, state["budget_fill_state"][key])


class BudgetFilledNSGAII(BudgetFillMixin, NSGA2Fair):
    method_version = "8.9-shared-budget-fill/" + NSGA2Fair.method_version


class BudgetFilledNSGAIII(BudgetFillMixin, NSGAIIIPOFair):
    method_version = "8.9-shared-budget-fill/" + NSGAIIIPOFair.method_version


class BudgetFilledSPEA2(BudgetFillMixin, SPEA2POFair):
    method_version = "8.9-shared-budget-fill/" + SPEA2POFair.method_version


class BudgetFilledRandomSearch(BudgetFillMixin, RandomSearchPOFair):
    method_version = "8.9-shared-budget-fill/" + RandomSearchPOFair.method_version


FILLED_BASELINES = {
    "NSGAII-PO-Fair": BudgetFilledNSGAII,
    "NSGAIII-PO-Fair": BudgetFilledNSGAIII,
    "SPEA2-PO-Fair": BudgetFilledSPEA2,
    "RandomSearch-PO-Fair": BudgetFilledRandomSearch,
}
