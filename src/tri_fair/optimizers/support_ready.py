"""Explicit fairness-readiness initialization ablation of three-objective MO-CAPO.

v9.2 remains the structural reproduction. v9.3 keeps its variation, progressive
race, heterogeneous population, subset selection and incumbent advancement, but
expands the common initial basis until the unchanged fairness gates are met.
Unseen block acquisition uses only search metadata and observed uncertainty.
"""

from __future__ import annotations

import numpy as np

from tri_fair.optimizers.mocapo_fair import MOCAPOFairOptimizer


class SupportReadyTriFairOptimizer(MOCAPOFairOptimizer):
    method_version = "9.3-mocapo-fairness-ready-initialization"
    _adaptive_fields = MOCAPOFairOptimizer._adaptive_fields + (
        "supported_initial_blocks",
        "warmup_records",
        "warmup_downstream_tokens",
    )

    def __init__(self, *args, bootstrap_readiness=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.bootstrap_readiness = bool(bootstrap_readiness)
        self.supported_initial_blocks = []
        self.warmup_records = []
        self.warmup_downstream_tokens = 0

    def _support_diagnostics(self, population, blocks):
        widths = {}
        records = []
        for p in population:
            record = self._record_for(p, blocks)
            diag = record.get("diagnostics", {})
            for key, value in (diag.get("uncertainty_by_group") or {}).items():
                if value is not None and np.isfinite(float(value)):
                    widths[key] = max(widths.get(key, 0.0), float(value))
            records.append(
                dict(
                    fairness_ready=record["fairness_ready"],
                    reasons=diag.get("readiness_reasons", []),
                    max_ci_width=diag.get("max_ci_width"),
                    valid_units=diag.get(
                        "valid_professions",
                        diag.get("ready_groups", diag.get("valid_identities")),
                    ),
                )
            )
        return dict(uncertainty_by_group=widths), records

    def _pre_optimization_loop(self):
        super()._pre_optimization_loop()
        population = list(self.prompts)
        blocks = list(self.initial_common_blocks)
        if self.bootstrap_readiness:
            start = self.predictor.llm.get_token_count()["total_tokens"]
            while not self._supported(population, blocks):
                diag, records = self._support_diagnostics(population, blocks)
                self.warmup_records.append(
                    dict(blocks=sorted(blocks), population=records)
                )
                missing = sorted(set(range(self.task.n_blocks)) - set(blocks))
                if not missing:
                    self._event(
                        "support_warmup_unattainable",
                        blocks=sorted(blocks),
                        population=records,
                    )
                    raise RuntimeError(
                        "Full search coverage cannot meet the unchanged fairness-readiness requirements; inspect support diagnostics before another GPU run"
                    )
                new = self.task.select_fairness_block(
                    missing, current_blocks=blocks, diagnostics=diag
                )
                blocks.append(new)
                self._get_block_vectors(population, blocks)
                self.algorithm_counts["warmup_added_blocks"] += 1
                self._event(
                    "support_warmup_block", block=int(new), blocks=sorted(blocks)
                )
            self.warmup_downstream_tokens = int(
                self.predictor.llm.get_token_count()["total_tokens"] - start
            )
            self.incumbents, self.non_incumbents = population, []
            self._update_incumbent_front(blocks)
            self.initial_common_blocks = sorted(blocks)
            self._sync()
        self.supported_initial_blocks = sorted(blocks)
        self._event(
            "support_initialization_finished",
            enabled=self.bootstrap_readiness,
            blocks=sorted(blocks),
            fairness_ready=self._supported(population, blocks),
            extra_downstream_tokens=self.warmup_downstream_tokens,
            required_all_search_blocks=len(blocks) == self.task.n_blocks,
        )

    def search_diagnostics(self):
        diag = super().search_diagnostics()
        diag.update(
            active_method="Tri-Fair-v9.3"
            if self.bootstrap_readiness
            else "Tri-Fair-v9.3-NoReadinessWarmup",
            method_version=self.method_version,
            readiness_warmup=self.bootstrap_readiness,
            supported_initial_blocks=self.supported_initial_blocks,
            warmup_downstream_tokens=self.warmup_downstream_tokens,
            readiness_required_full_initial_coverage=len(self.supported_initial_blocks)
            == self.task.n_blocks,
            initialization="one random shared block, then metadata-directed common coverage until unchanged fairness gates pass"
            if self.bootstrap_readiness
            else "v9.2 single random shared block",
            exact_mocapo_initialization=False if self.bootstrap_readiness else True,
            mathematical_thresholds_changed=False,
            holdout_feedback_used=False,
        )
        return diag

    def optimizer_state_dict(self):
        state = super().optimizer_state_dict()
        state["support_configuration"] = self.bootstrap_readiness
        return state

    def load_optimizer_state_dict(self, state):
        if state.get("support_configuration") != self.bootstrap_readiness:
            raise ValueError("Fairness-readiness initialization configuration differs")
        super().load_optimizer_state_dict(state)
