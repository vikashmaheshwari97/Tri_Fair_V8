"""Tri-Fair v8.3: bounded, attributed, memory-aware prompt search."""

from __future__ import annotations
from collections import Counter
import hashlib
import logging
from promptolution.utils.prompt import Prompt
from tri_fair.optimizers.trifair import TriFairOptimizer
from tri_fair.search.contrastive import _anchor_for_mode, _diverse_partner
from tri_fair.search.attribution import (
    attribute_search,
    contract_instruction,
    gold_demonstration,
    parse_proposal,
    revision_request,
)

LOGGER = logging.getLogger(__name__)


class AttributedTriFairOptimizer(TriFairOptimizer):
    method_version = "8.3-attributed-memory-bounded-search"
    _adaptive_fields = TriFairOptimizer._adaptive_fields + (
        "generation_calls",
        "idle_rounds",
        "proposal_memory",
        "proposal_rejections",
        "search_stop_reason",
    )

    def __init__(
        self,
        *args,
        max_generation_calls=128,
        max_idle_rounds=3,
        prompt_word_limit=96,
        generation_input_token_limit=2100,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_generation_calls = int(max_generation_calls)
        self.max_idle_rounds = int(max_idle_rounds)
        self.prompt_word_limit = int(prompt_word_limit)
        self.generation_input_token_limit = int(generation_input_token_limit)
        if (
            min(self.max_generation_calls, self.max_idle_rounds, self.prompt_word_limit)
            < 1
            or self.generation_input_token_limit < 512
        ):
            raise ValueError("Invalid attributed-search resource limits")
        self.generation_calls = 0
        self.idle_rounds = 0
        self.proposal_memory = []
        self.proposal_rejections = Counter()
        self.search_stop_reason = None
        self._round_proposals = []

    def _build_seed_population(self):
        return [
            Prompt(contract_instruction(p.instruction, self.task.classes), [])
            for p in self.prompts
        ]

    def _generate_challengers(self):
        if not self.objective_aware_variation:
            return super()._generate_challengers()
        if self.generation_calls >= self.max_generation_calls:
            self.search_stop_reason = "generation_call_limit"
            return []
        count = max(1, min(int(self.challenger_count()), 3))
        modes = self.allocate_modes(count)
        common = self._common_blocks()
        requests, parents, evidence = [], [], []
        for index, mode in enumerate(modes):
            # Every proposal counts, including invalid and duplicate responses.
            self.mode_attempts[mode] += 1
            anchor, _ = _anchor_for_mode(self, mode, common)
            partner = _diverse_partner(self, anchor)
            blocks = self.task.get_evaluated_blocks([anchor])[anchor]
            attribution = attribute_search(
                self.task,
                anchor,
                blocks,
                offset=self.generation_calls * count + index,
                limit=self.feedback_cases,
            )
            if not self.enable_feedback:
                attribution = {
                    "confusions": [],
                    "evidence": [],
                    "conditional_error_clusters": [],
                }
            requests.append(
                revision_request(
                    self,
                    anchor,
                    partner,
                    mode,
                    attribution,
                    attempt=self.generation_calls * count + index,
                )
            )
            parents.append(anchor)
            evidence.append(attribution)
        params = getattr(self.meta_llm, "sampling_params", None)
        if params is not None and hasattr(params, "seed"):
            # Reproducible per-batch seeds avoid identical stochastic retries.
            params.seed = int(self.task.seed + 1000003 + self.generation_calls)
        self.generation_calls += 1
        responses = self._responses(self.meta_llm.get_response(requests), len(requests))
        output, mapping = [], {}

        def add(candidate, mode):
            key = candidate.construct_prompt()
            if key in self.seen_candidates:
                self.proposal_rejections["duplicate_candidate"] += 1
                return False
            self.seen_candidates.add(key)
            output.append(candidate)
            mapping[key] = mode
            self.pending_rewards[key] = mode
            self._round_proposals.append(candidate)
            return True

        for response, anchor, mode, attribution in zip(
            responses, parents, modes, evidence
        ):
            instruction, diagnosis, reason = parse_proposal(
                response, self.task.classes, word_limit=self.prompt_word_limit
            )
            accepted = False
            before_candidates = len(output)
            if reason:
                self.proposal_rejections[reason] += 1
            else:
                base = contract_instruction(
                    instruction, self.task.classes, catalog=mode == "quality"
                )
                accepted = add(Prompt(base, []), mode)
                # A short gold-labeled demonstration is a distinct, measured
                # candidate; no model-generated labels or free prediction calls.
                if (
                    self.enable_demo_search
                    and self.upper_shots > 0
                    and mode in ("quality", "fairness")
                ):
                    confusion = attribution["confusions"]
                    target = confusion[0]["gold"] if confusion else None
                    demo = gold_demonstration(
                        self, target=target, offset=self.generation_calls - 1
                    )
                    if demo:
                        accepted = add(Prompt(base, [demo]), mode) or accepted
                if not accepted:
                    reason = "duplicate_candidate"
            if anchor.few_shots and mode == "cost":
                add(Prompt(anchor.instruction, []), "cost")
            self.proposal_memory.append(
                {
                    "mode": mode,
                    "diagnosis": diagnosis[:180],
                    "instruction": instruction[:480],
                    "status": "proposed" if accepted else reason,
                    "parent_quality": float(
                        self.get_fairness_record(anchor)["quality"]
                    ),
                    "candidates": [
                        hashlib.sha256(p.construct_prompt().encode()).hexdigest()
                        for p in output[before_candidates:]
                    ],
                }
            )
        self.proposal_memory = self.proposal_memory[-24:]
        self.generated_modes = mapping
        self.operator_history.append(
            {
                "step": self.current_step,
                "modes": list(mapping.values()),
                "generated": len(output),
                "calls": self.generation_calls,
            }
        )
        return output

    def _select_survivors(self):
        super()._select_survivors()
        # One reflective proposal may create both zero- and one-shot variants;
        # do not count more bandit successes than proposal attempts.
        for mode in self.mode_attempts:
            self.mode_successes[mode] = min(
                self.mode_successes[mode], self.mode_attempts[mode]
            )

    def _step(self):
        before = len(self.task.pred_cache)
        self._round_proposals = []
        prompts = super()._step()
        for candidate in self._round_proposals:
            blocks = self.task.get_evaluated_blocks([candidate])[candidate]
            if not blocks:
                continue
            key = hashlib.sha256(candidate.construct_prompt().encode()).hexdigest()
            record = self.get_fairness_record(candidate, blocks=blocks)
            observation = {
                "candidate_id": key,
                "quality": record["quality"],
                "cost": record["cost"],
                "unfairness": record["fairness"],
                "blocks": list(blocks),
                "fairness_ready": record["fairness_ready"],
                "on_search_front": candidate in self.incumbents,
            }
            for item in self.proposal_memory:
                if key in item.get("candidates", []):
                    item.setdefault("measurements", []).append(observation)
                    item["status"] = "measured_on_search_blocks"
        self._round_proposals = []
        if len(self.task.pred_cache) == before and not self.search_stop_reason:
            # Spend remaining search effort on unresolved coverage when possible.
            self._advance_one_incumbent()
            self.scores = self._get_evaluated_vectors(self.prompts).tolist()
            self.idle_rounds += int(len(self.task.pred_cache) == before)
            if len(self.task.pred_cache) != before:
                self.idle_rounds = 0
            if self.idle_rounds >= self.max_idle_rounds:
                self.search_stop_reason = "no_new_evaluations"
        else:
            self.idle_rounds = 0
        LOGGER.info(
            "Tri-Fair v8.3 progress: step=%d new_predictions=%d generation_calls=%d idle_rounds=%d stop=%s",
            self.current_step,
            len(self.task.pred_cache) - before,
            self.generation_calls,
            self.idle_rounds,
            self.search_stop_reason,
        )
        return prompts

    def _on_step_end(self):
        return bool(super()._on_step_end()) and not self.search_stop_reason

    def search_diagnostics(self):
        return {
            "generation_calls": self.generation_calls,
            "idle_rounds": self.idle_rounds,
            "rejected_proposals": dict(self.proposal_rejections),
            "stop_reason": self.search_stop_reason,
            "recent_proposals": self.proposal_memory,
            "candidate_identity": "sha256 of exact instruction and demonstrations",
            "seen_candidates": len(self.seen_candidates),
        }

    def optimizer_state_dict(self):
        payload = super().optimizer_state_dict()
        payload["attributed_configuration"] = self._attributed_configuration()
        return payload

    def _attributed_configuration(self):
        return {
            key: getattr(self, key)
            for key in (
                "max_generation_calls",
                "max_idle_rounds",
                "prompt_word_limit",
                "generation_input_token_limit",
            )
        }

    def load_optimizer_state_dict(self, payload):
        if payload.get("attributed_configuration") != self._attributed_configuration():
            raise ValueError("Attributed-search checkpoint configuration mismatch")
        super().load_optimizer_state_dict(payload)
