"""One Tri-Fair optimizer: attributed reflection + stable edits + merge search.

All proposals share one measured archive, prediction cache and token controller.
Historical implementations remain importable solely for reproduction.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import math
from promptolution.utils.prompt import Prompt
from promptolution.utils.token_counter import get_token_counter
from tri_fair.optimizers.attributed import AttributedTriFairOptimizer
from tri_fair.optimizers.stable import StableTriFairOptimizer
from tri_fair.search.attribution import (
    attribute_search,
    parse_proposal,
    revision_request,
)
from tri_fair.search.stability import canonical_instruction, paired_search_evidence
from tri_fair.search.portfolio import (
    ARMS,
    select_arm,
    cached_front_gain,
    complementary_parents,
)


class ConsolidatedTriFairOptimizer(StableTriFairOptimizer):
    method_version = "8.6-shared-budget-adaptive-reflection-stability-merge"
    _adaptive_fields = StableTriFairOptimizer._adaptive_fields + (
        "portfolio_trials",
        "portfolio_rewards",
        "portfolio_trace",
    )

    def __init__(
        self,
        *args,
        portfolio_exploration=0.20,
        portfolio_interval=5,
        portfolio_reward_scale=1000000.0,
        enabled_strategies=ARMS,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.portfolio_exploration = float(portfolio_exploration)
        self.portfolio_interval = int(portfolio_interval)
        self.portfolio_reward_scale = float(portfolio_reward_scale)
        self.enabled_strategies = tuple(enabled_strategies)
        if (
            not math.isfinite(self.portfolio_exploration)
            or self.portfolio_exploration < 0
            or self.portfolio_interval < 3
            or not math.isfinite(self.portfolio_reward_scale)
            or self.portfolio_reward_scale <= 0
        ):
            raise ValueError("Invalid portfolio settings")
        if (
            not self.enabled_strategies
            or len(set(self.enabled_strategies)) != len(self.enabled_strategies)
            or not set(self.enabled_strategies) <= set(ARMS)
        ):
            raise ValueError("Unknown or duplicate proposal strategies")
        self.portfolio_trials = Counter({a: 0 for a in self.enabled_strategies})
        self.portfolio_rewards = Counter({a: 0.0 for a in self.enabled_strategies})
        self.portfolio_trace = []
        self._active_arm = None

    def _generate_challengers(self):
        if self.generation_calls >= self.max_generation_calls:
            self.search_stop_reason = "generation_call_limit"
            return []
        arm = select_arm(
            self.portfolio_trials,
            self.portfolio_rewards,
            exploration=self.portfolio_exploration,
            interval=self.portfolio_interval,
            arms=self.enabled_strategies,
        )
        self._active_arm = arm
        self.portfolio_trials[arm] += 1
        previous_entries = {id(e) for e in self.proposal_memory}
        if arm == "attributed_reflection":
            candidates = AttributedTriFairOptimizer._generate_challengers(self)
        elif arm == "stable_components":
            candidates = StableTriFairOptimizer._generate_challengers(self)
        else:
            candidates = self._merge_challengers()
        # Attach strategy identity to the common proposal history. Old v8.3
        # and v8.5 state, mode selection and progress safeguards remain active.
        for entry in self.proposal_memory:
            if id(entry) not in previous_entries:
                entry["proposal_strategy"] = arm
        self._credit_keys = {
            hashlib.sha256(p.construct_prompt().encode()).hexdigest()
            for p in candidates
        }
        return candidates

    def _merge_challengers(self):
        pair = complementary_parents(self)
        if pair is None:
            self.proposal_rejections["merge_requires_two_observed_parents"] += 1
            return []
        mother, father, evidence = pair
        blocks = evidence["blocks"]
        attribution = attribute_search(
            self.task,
            mother,
            blocks,
            offset=sum(self.portfolio_trials.values()),
            limit=self.feedback_cases,
            targeted=getattr(self, "targeted_feedback", False),
        )
        if self.enable_feedback:
            attribution["complementarity"] = evidence
            attribution["observed_paired_errors"] = paired_search_evidence(
                self.task, mother, blocks
            )
        else:
            attribution = {
                "confusions": [],
                "evidence": [],
                "conditional_error_clusters": [],
                "observed_paired_errors": [],
            }
        attribution["measured_strategy_history"] = [
            {k: r.get(k) for k in ("strategy", "hv_gain", "reward", "blocks")}
            for r in self.portfolio_trace[-3:]
        ]
        prefix = (
            "Combine complementary successful decision rules from the two parents. "
            "Repair only observed failures. Preserve evidence needed for the task, "
            "including indicated pronouns in coreference and professional training in biographies. "
            "Avoid demographic shortcuts. Do not concatenate redundant instructions.\n"
        )
        counter = get_token_counter(self.meta_llm)
        limit = self.generation_input_token_limit
        self.generation_input_token_limit = max(512, limit - counter(prefix) - 8)
        try:
            request = prefix + revision_request(
                self,
                mother,
                father,
                "balanced",
                attribution,
                attempt=sum(self.portfolio_trials.values()),
            )
        finally:
            self.generation_input_token_limit = limit
        if counter(request) > limit:
            raise ValueError("Merge request exceeds generator context limit")
        params = getattr(self.meta_llm, "sampling_params", None)
        if params is not None and hasattr(params, "seed"):
            params.seed = int(self.task.seed + 1000003 + self.generation_calls)
        self.generation_calls += 1
        response = self._responses(self.meta_llm.get_response([request]), 1)[0]
        instruction, diagnosis, reason = parse_proposal(
            response, self.task.classes, word_limit=self.prompt_word_limit
        )
        output, mapping = [], {}
        self.mode_attempts["balanced"] += 1
        if reason:
            self.proposal_rejections[reason] += 1
        else:
            instruction = canonical_instruction(
                instruction,
                self.task.classes,
                catalog="Allowed labels:" in mother.instruction,
            )
            self._register(
                Prompt(instruction, list(mother.few_shots)), "balanced", output, mapping
            )
        self.proposal_memory.append(
            dict(
                mode="balanced",
                diagnosis=diagnosis[:180],
                instruction=instruction[:480],
                status="proposed" if output else reason or "duplicate_candidate",
                complementarity=evidence,
                candidates=[
                    hashlib.sha256(p.construct_prompt().encode()).hexdigest()
                    for p in output
                ],
            )
        )
        self.proposal_memory = self.proposal_memory[-24:]
        self.generated_modes = mapping
        self.operator_history.append(
            dict(
                step=self.current_step,
                modes=list(mapping.values()),
                generated=len(output),
                calls=self.generation_calls,
            )
        )
        return output

    def _step(self):
        before = list(self.incumbents)
        tokens_before = int(self.downstream_llm.get_token_count()["total_tokens"])
        self._active_arm = None
        self._credit_keys = set()
        try:
            return super()._step()
        finally:
            # A budget-stop can occur after some children were evaluated; record
            # only what was actually measured, even on that final partial step.
            if self._active_arm is not None:
                spent = (
                    int(self.downstream_llm.get_token_count()["total_tokens"])
                    - tokens_before
                )
                gain = cached_front_gain(self, before, list(self.incumbents))
                reward = min(
                    1.0,
                    max(0.0, float(gain.get("hv_gain", 0.0)))
                    * self.portfolio_reward_scale
                    / max(1, spent),
                )
                self.portfolio_rewards[self._active_arm] += reward
                record = dict(
                    strategy=self._active_arm,
                    step=self.current_step,
                    downstream_tokens_spent=spent,
                    reward=reward,
                    **gain,
                )
                self.portfolio_trace.append(record)
                self.portfolio_trace = self.portfolio_trace[-256:]
                for entry in self.proposal_memory:
                    if self._credit_keys.intersection(
                        entry.get("candidates", [])
                    ) and entry.get("measurements"):
                        entry["strategy_credit"] = dict(
                            hv_gain=gain.get("hv_gain"),
                            reward=reward,
                            blocks=gain["blocks"],
                        )
                self._active_arm = None

    def search_diagnostics(self):
        return dict(
            super().search_diagnostics(),
            active_method="Tri-Fair-v8.6",
            enabled_strategies=list(self.enabled_strategies),
            proposal_strategy_trials=dict(self.portfolio_trials),
            proposal_strategy_rewards=dict(self.portfolio_rewards),
            portfolio_trace=self.portfolio_trace,
            credit_scope="same cached search blocks; no validation feedback",
            shared_budget=True,
        )

    def optimizer_state_dict(self):
        payload = super().optimizer_state_dict()
        payload["portfolio_configuration"] = [
            self.portfolio_exploration,
            self.portfolio_interval,
            self.portfolio_reward_scale,
            self.enabled_strategies,
        ]
        return payload

    def load_optimizer_state_dict(self, payload):
        expected = [
            self.portfolio_exploration,
            self.portfolio_interval,
            self.portfolio_reward_scale,
            self.enabled_strategies,
        ]
        if payload.get("portfolio_configuration") != expected:
            raise ValueError("Portfolio checkpoint configuration mismatch")
        super().load_optimizer_state_dict(payload)
        self._active_arm = None
