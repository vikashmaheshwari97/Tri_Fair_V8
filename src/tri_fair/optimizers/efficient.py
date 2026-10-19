"""Tri-Fair v8.6 efficiency revision: compact edits and evidence-gated merges.

This changes proposal search, never the measured objectives or control methods.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import math

from promptolution.utils.prompt import Prompt
from promptolution.utils.token_counter import get_token_counter
from tri_fair.optimizers.attributed import AttributedTriFairOptimizer
from tri_fair.optimizers.stable import StableTriFairOptimizer
from tri_fair.optimizers.consolidated import ConsolidatedTriFairOptimizer
from tri_fair.search.attribution import (
    CONTRACT,
    attribute_search,
    parse_proposal,
    revision_request,
)
from tri_fair.search.portfolio import ARMS, complementary_parents, select_arm

EFFICIENT_ARMS = (
    "attributed_reflection",
    "stable_components",
    "compact_edit",
    "complementary_merge",
)
SHORT_CONTRACT = "Return one allowed label in <final_answer> tags; no explanation."


def compact_contract(instruction):
    text = " ".join(str(instruction).split())
    for known in (
        CONTRACT.strip(),
        SHORT_CONTRACT,
        "Return only one allowed label inside <final_answer> tags.",
        "Output exactly one allowed label inside <final_answer> tags.",
    ):
        text = text.replace(known, "")
    return " ".join(text.split()).strip() + " " + SHORT_CONTRACT


def scaled_credit_arm(trials, rewards, arms, *, exploration, interval):
    """Put empirical reward and exploration on compatible scales.

    This only selects operators. The original credit and objective values are
    retained verbatim in diagnostics, including zero/ineligible observations.
    """
    scale = max((rewards[a] / max(1, trials[a]) for a in arms), default=0.0)
    adjusted = {a: rewards[a] / scale if scale > 0 else 0.0 for a in arms}
    return select_arm(
        trials, adjusted, exploration=exploration, interval=interval, arms=tuple(arms)
    )


class EfficientTriFairOptimizer(ConsolidatedTriFairOptimizer):
    method_version = "8.6.1-compact-targeted-supported-stability-gated-merge"
    _adaptive_fields = ConsolidatedTriFairOptimizer._adaptive_fields + (
        "compact_attempts",
        "last_merge_round",
        "merge_gate_rejections",
    )

    def __init__(
        self,
        *args,
        enabled_strategies=EFFICIENT_ARMS,
        merge_interval=5,
        merge_min_repairs=2,
        compression_target_ratio=0.80,
        **kwargs,
    ):
        super().__init__(*args, enabled_strategies=ARMS, **kwargs)
        self.enabled_strategies = tuple(enabled_strategies)
        if (
            not self.enabled_strategies
            or len(set(self.enabled_strategies)) != len(self.enabled_strategies)
            or not set(self.enabled_strategies) <= set(EFFICIENT_ARMS)
        ):
            raise ValueError("Unknown or duplicate proposal strategies")
        self.merge_interval = int(merge_interval)
        self.merge_min_repairs = int(merge_min_repairs)
        self.compression_target_ratio = float(compression_target_ratio)
        if (
            self.merge_interval < 3
            or self.merge_min_repairs < 1
            or not 0.4 <= self.compression_target_ratio < 1
        ):
            raise ValueError("Invalid efficient proposal settings")
        self.require_supported_stability = True
        self.targeted_feedback = True
        self.portfolio_trials = Counter({a: 0 for a in self.enabled_strategies})
        self.portfolio_rewards = Counter({a: 0.0 for a in self.enabled_strategies})
        self.compact_attempts = 0
        self.last_merge_round = -self.merge_interval
        self.merge_gate_rejections = Counter()

    def _merge_eligible(self):
        round_id = sum(self.portfolio_trials.values())
        if round_id - self.last_merge_round < self.merge_interval:
            return False, "cooldown"
        pair = complementary_parents(self)
        if pair is None:
            return False, "no_observed_pair"
        mother, father, evidence = pair
        if len(evidence["blocks"]) < self.min_racing_blocks:
            return False, "insufficient_shared_blocks"
        if (
            evidence["father_repairs"] < self.merge_min_repairs
            or evidence["mother_repairs"] < 1
        ):
            return False, "insufficient_bidirectional_repairs"
        for parent in (mother, father):
            record = self.get_fairness_record(parent, blocks=evidence["blocks"])
            if not record["fairness_ready"]:
                return False, "unsupported_parent_fairness"
        previous = [
            r for r in self.portfolio_trace if r["strategy"] == "complementary_merge"
        ]
        if len(previous) >= 2 and all(
            not r.get("eligible") or r.get("hv_gain", 0) <= 0 for r in previous[-2:]
        ):
            if round_id - self.last_merge_round < 2 * self.merge_interval:
                return False, "unproductive_merge_backoff"
        return True, None

    def _generate_challengers(self):
        if self.generation_calls >= self.max_generation_calls:
            self.search_stop_reason = "generation_call_limit"
            return []
        arms = list(self.enabled_strategies)
        if "complementary_merge" in arms:
            eligible, reason = self._merge_eligible()
            if not eligible:
                arms.remove("complementary_merge")
                self.merge_gate_rejections[reason] += 1
        if not arms:
            return []
        arm = scaled_credit_arm(
            self.portfolio_trials,
            self.portfolio_rewards,
            arms,
            exploration=self.portfolio_exploration,
            interval=self.portfolio_interval,
        )
        self._active_arm = arm
        self.portfolio_trials[arm] += 1
        previous_entries = {id(e) for e in self.proposal_memory}
        if arm == "attributed_reflection":
            candidates = AttributedTriFairOptimizer._generate_challengers(self)
        elif arm == "stable_components":
            candidates = StableTriFairOptimizer._generate_challengers(self)
        elif arm == "compact_edit":
            candidates = self._compact_challengers()
        else:
            self.last_merge_round = sum(self.portfolio_trials.values())
            candidates = super()._merge_challengers()
        for entry in self.proposal_memory:
            if id(entry) not in previous_entries:
                entry["proposal_strategy"] = arm
        self._credit_keys = {
            hashlib.sha256(p.construct_prompt().encode()).hexdigest()
            for p in candidates
        }
        return candidates

    def _compact_challengers(self):
        pool = list(self.parent_pool())
        maps = self.task.get_evaluated_blocks(pool)
        common = sorted(set.intersection(*(set(maps[p]) for p in pool))) if pool else []
        if not common:
            return []
        records = [(p, self.get_fairness_record(p, blocks=common)) for p in pool]
        records = [
            (p, r)
            for p, r in records
            if r["fairness_ready"] and math.isfinite(r["quality"])
        ]
        if not records:
            return []
        floor = max(r["quality"] for _, r in records) - self.parent_quality_margin
        eligible = [(p, r) for p, r in records if r["quality"] >= floor]
        # Work on the expensive end of the high-quality pool; do not spend
        # compression attempts repeatedly editing an already shortest prompt.
        eligible.sort(key=lambda item: (-item[1]["cost"], item[0].construct_prompt()))
        anchor, _ = eligible[self.compact_attempts % len(eligible)]
        attempt = self.compact_attempts
        self.compact_attempts += 1
        self.mode_attempts["cost"] += 1
        output, mapping = [], {}
        if anchor.few_shots:
            demos = (
                []
                if self._metric_family() == "winobias"
                else list(anchor.few_shots[:-1])
            )
            self._register(
                Prompt(
                    anchor.instruction,
                    demos,
                    downstream_template=anchor.downstream_template,
                ),
                "cost",
                output,
                mapping,
            )
        shorter = compact_contract(anchor.instruction)
        if not output and self.token_counter(shorter) < self.token_counter(
            anchor.instruction
        ):
            self._register(
                Prompt(
                    shorter,
                    list(anchor.few_shots),
                    downstream_template=anchor.downstream_template,
                ),
                "cost",
                output,
                mapping,
            )
        diagnosis, reason, instruction = "exact component compression", None, shorter
        if not output:
            evidence = attribute_search(
                self.task,
                anchor,
                common,
                offset=attempt,
                limit=self.feedback_cases,
                targeted=self.enable_feedback,
            )
            if not self.enable_feedback:
                evidence = {
                    "evidence": [],
                    "confusions": [],
                    "conditional_error_clusters": [],
                }
            maximum_words = max(
                12,
                min(
                    64,
                    int(
                        len(anchor.instruction.split()) * self.compression_target_ratio
                    ),
                ),
            )
            prefix = (
                f"Compress to at most {maximum_words} words. Preserve the decision rule, negation, "
                "label mapping and necessary task evidence. Remove redundant prose, not task evidence. "
                "Keep successful cases correct while repairing the observed mistakes.\n"
            )
            counter = get_token_counter(self.meta_llm)
            old_limit = self.generation_input_token_limit
            self.generation_input_token_limit = max(
                512, old_limit - counter(prefix) - 8
            )
            try:
                request = prefix + revision_request(
                    self, anchor, anchor, "cost", evidence, attempt=attempt
                )
            finally:
                self.generation_input_token_limit = old_limit
            if counter(request) > old_limit:
                raise ValueError("Compact edit request exceeds generator context limit")
            params = getattr(self.meta_llm, "sampling_params", None)
            if params is not None and hasattr(params, "seed"):
                params.seed = int(self.task.seed + 1000003 + self.generation_calls)
            self.generation_calls += 1
            response = self._responses(self.meta_llm.get_response([request]), 1)[0]
            instruction, diagnosis, reason = parse_proposal(
                response, self.task.classes, word_limit=maximum_words
            )
            if not reason:
                instruction = compact_contract(instruction)
                if self.token_counter(
                    instruction
                ) <= self.compression_target_ratio * self.token_counter(
                    anchor.instruction
                ):
                    self._register(
                        Prompt(
                            instruction,
                            list(anchor.few_shots),
                            downstream_template=anchor.downstream_template,
                        ),
                        "cost",
                        output,
                        mapping,
                    )
                else:
                    reason = "compression_target_not_met"
            if reason:
                self.proposal_rejections[reason] += 1
        self.proposal_memory.append(
            dict(
                mode="cost",
                diagnosis=diagnosis[:180],
                instruction=instruction[:480],
                status="proposed" if output else reason or "duplicate_candidate",
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

    def search_diagnostics(self):
        return dict(
            super().search_diagnostics(),
            efficiency_revision=self.method_version,
            merge_gate_rejections=dict(self.merge_gate_rejections),
            compact_attempts=self.compact_attempts,
            supported_stability_required=True,
            targeted_feedback=True,
            merge_interval=self.merge_interval,
            merge_min_repairs=self.merge_min_repairs,
            compression_target_ratio=self.compression_target_ratio,
            credit_selection="empirical reward rescaled before UCB; raw credit retained",
        )

    def optimizer_state_dict(self):
        payload = super().optimizer_state_dict()
        payload["efficiency_configuration"] = [
            self.merge_interval,
            self.merge_min_repairs,
            self.compression_target_ratio,
        ]
        return payload

    def load_optimizer_state_dict(self, payload):
        if payload.get("efficiency_configuration") != [
            self.merge_interval,
            self.merge_min_repairs,
            self.compression_target_ratio,
        ]:
            raise ValueError("Efficiency checkpoint configuration mismatch")
        super().load_optimizer_state_dict(payload)
