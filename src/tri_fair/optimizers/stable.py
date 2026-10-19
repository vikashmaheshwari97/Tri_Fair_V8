"""Tri-Fair v8.5: stable parents and component-preserving prompt search.

Original implementation informed by GEPA reflection and fairness literature;
no external optimizer implementation or model-weight training is embedded.
"""

from __future__ import annotations

import hashlib
from promptolution.utils.prompt import Prompt
from tri_fair.optimizers.attributed import AttributedTriFairOptimizer
from tri_fair.search.attribution import (
    attribute_search,
    parse_proposal,
    revision_request,
)
from tri_fair.search.stability import (
    balanced_gold_demonstrations,
    canonical_instruction,
    choose_stable_parent,
    paired_search_evidence,
)


class StableTriFairOptimizer(AttributedTriFairOptimizer):
    method_version = "8.5-search-block-stability-component-preserving"
    _adaptive_fields = AttributedTriFairOptimizer._adaptive_fields + ("proposal_round",)
    _cycle = ("quality", "fairness", "quality", "balanced", "cost", "explore")

    def __init__(
        self, *args, stability_strength=1.0, parent_quality_margin=0.04, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.stability_strength = float(stability_strength)
        self.parent_quality_margin = float(parent_quality_margin)
        if (
            not 0 <= self.stability_strength <= 3
            or not 0 <= self.parent_quality_margin <= 0.2
        ):
            raise ValueError("Invalid search stability settings")
        self.stability_cache = {}
        self.proposal_round = 0

    def _metric_family(self):
        if str(self.task.fairness_metric_name).startswith("winobias"):
            return "winobias"
        return super()._metric_family()

    def _build_seed_population(self):
        # Keep original compact seeds; expose schema-complete alternatives
        # without appending a catalog to every candidate during every mutation.
        return [
            Prompt(
                canonical_instruction(
                    p.instruction,
                    self.task.classes,
                    catalog=(i % 2 == 1 and len(self.task.classes) > 3),
                ),
                [],
            )
            for i, p in enumerate(self.prompts)
        ]

    def _register(self, candidate, mode, output, mapping):
        key = candidate.construct_prompt()
        if key in self.seen_candidates:
            self.proposal_rejections["duplicate_candidate"] += 1
            return False
        self.seen_candidates.add(key)
        self.pending_rewards[key] = mode
        self._round_proposals.append(candidate)
        mapping[key] = mode
        output.append(candidate)
        return True

    def _generate_challengers(self):
        if not self.objective_aware_variation:
            return super()._generate_challengers()
        if self.generation_calls >= self.max_generation_calls:
            self.search_stop_reason = "generation_call_limit"
            return []
        # A two-component proposal is raced one at a time. Keep useful demos
        # during instruction edits; test demo edits with the original instruction.
        output, mapping = [], {}
        count = max(1, min(int(self.challenger_count()), 2))
        plans = []
        for _ in range(count):
            attempt = self.proposal_round
            mode = self._cycle[attempt % len(self._cycle)]
            self.proposal_round += 1
            self.mode_attempts[mode] += 1
            anchor, stability = choose_stable_parent(self, mode, attempt)
            blocks = self.task.get_evaluated_blocks([anchor])[anchor]
            evidence = attribute_search(
                self.task,
                anchor,
                blocks,
                offset=attempt,
                limit=self.feedback_cases,
                targeted=getattr(self, "targeted_feedback", False),
            )
            if not self.enable_feedback:
                evidence = {
                    "confusions": [],
                    "evidence": [],
                    "conditional_error_clusters": [],
                }
            if mode == "cost" and anchor.few_shots:
                # Exact component deletion needs no generation, but is still
                # evaluated and charged through the normal search budget.
                self._register(
                    Prompt(anchor.instruction, list(anchor.few_shots[:-1])),
                    mode,
                    output,
                    mapping,
                )
            partner = self.incumbents[attempt % len(self.incumbents)]
            augmented = dict(evidence)
            augmented["search_block_sensitivity"] = stability
            if self.enable_feedback:
                augmented["observed_paired_errors"] = paired_search_evidence(
                    self.task, anchor, blocks
                )
            guidance = {
                "bios": " Preserve valid occupational training and explicit current duties. Do not discard education when it establishes qualification; names and gender are not occupation evidence.",
                "civil": " Treat identity mentions symmetrically. Separate directed abuse from quotations and counterspeech without blanket removal of identity terms.",
                "bbq": " Preserve explicit passage evidence and negation. Use the unknown option only when the passage does not determine the answer; map options explicitly.",
                "winobias": " Preserve pronouns and grammatical gender needed for reference resolution. Use syntactic roles and causal context; occupational gender stereotypes are not evidence. Do not mask or remove the indicated pronoun.",
            }.get(self._metric_family(), "")
            # Bound the COMPLETE request, including task-specific guidance.
            from promptolution.utils.token_counter import get_token_counter

            counter = get_token_counter(self.meta_llm)
            prefix = (
                "Edit one failed rule, retain successful rules and the label schema. Internally compare task evidence under irrelevant identity changes; do not emit reasoning."
                + guidance
                + "\n"
            )
            if mode == "cost":
                prefix += "Shorten the instruction while preserving its exact decision rule and all labels.\n"
            # revision_request has its own pruning; reserve room for this prefix.
            original_limit = self.generation_input_token_limit
            self.generation_input_token_limit = max(
                512, original_limit - counter(prefix) - 8
            )
            try:
                request = prefix + revision_request(
                    self, anchor, partner, mode, augmented, attempt=attempt
                )
            finally:
                self.generation_input_token_limit = original_limit
            if counter(request) > original_limit:
                raise ValueError(
                    "Complete stable proposal exceeds generator input limit"
                )
            plans.append((request, anchor, mode, evidence, stability))
        params = getattr(self.meta_llm, "sampling_params", None)
        if params is not None and hasattr(params, "seed"):
            params.seed = int(self.task.seed + 1000003 + self.generation_calls)
        self.generation_calls += 1
        responses = self._responses(
            self.meta_llm.get_response([p[0] for p in plans]), len(plans)
        )
        for response, (_, anchor, mode, evidence, stability) in zip(responses, plans):
            before = len(output)
            instruction, diagnosis, reason = parse_proposal(
                response, self.task.classes, word_limit=self.prompt_word_limit
            )
            if reason:
                self.proposal_rejections[reason] += 1
            else:
                # Preserve the parent's catalog only when it had one.
                instruction = canonical_instruction(
                    instruction,
                    self.task.classes,
                    catalog="Allowed labels:" in anchor.instruction,
                )
                if mode == "cost" and self.token_counter(
                    instruction
                ) >= self.token_counter(anchor.instruction):
                    self.proposal_rejections["compression_not_shorter"] += 1
                else:
                    self._register(
                        Prompt(instruction, list(anchor.few_shots)),
                        mode,
                        output,
                        mapping,
                    )
            if (
                self.enable_demo_search
                and self.upper_shots > 0
                and mode in ("quality", "fairness", "balanced")
            ):
                demos = balanced_gold_demonstrations(
                    self, evidence, offset=self.proposal_round - 1
                )
                if demos:
                    self._register(
                        Prompt(anchor.instruction, demos), mode, output, mapping
                    )
            self.proposal_memory.append(
                dict(
                    mode=mode,
                    diagnosis=diagnosis[:180],
                    instruction=instruction[:480],
                    status="proposed"
                    if len(output) > before
                    else (reason or "no_novel_component"),
                    parent_sensitivity=stability,
                    candidates=[
                        hashlib.sha256(p.construct_prompt().encode()).hexdigest()
                        for p in output[before:]
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
            proposal_round=self.proposal_round,
            stability_scope="cached search blocks only; auxiliary parent selection",
            objective_scores="unchanged pooled measured scores",
            component_policy="instruction edits preserve demonstrations; demo edits preserve instruction",
        )

    def optimizer_state_dict(self):
        payload = super().optimizer_state_dict()
        payload["stability_configuration"] = [
            self.stability_strength,
            self.parent_quality_margin,
        ]
        return payload

    def load_optimizer_state_dict(self, payload):
        if payload.get("stability_configuration") != [
            self.stability_strength,
            self.parent_quality_margin,
        ]:
            raise ValueError("Stability checkpoint configuration mismatch")
        super().load_optimizer_state_dict(payload)
        self.stability_cache.clear()
