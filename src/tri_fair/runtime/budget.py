"""Strict downstream-token budget control for Tri-Fair v2.3.

The upstream promptolution TokenCountCallback checks the token count only after a
complete optimizer step.  A complete step may evaluate several prompts or fairness
blocks, so the requested budget can be exceeded by one large step.

This module adds:

* a pre-call hard budget admission check;
* exact accounting in the same raw-input convention used by promptolution;
* a conservative output-token reserve;
* deterministic near-budget challenger reduction;
* an adapter that protects every downstream predictor call, including few-shot
  reasoning calls made outside FairnessTask.evaluate;
* checkpointable reporting state.

Meta-LLM tokens remain separately logged and are not part of the downstream
optimization budget, matching the MO-CAPO convention.
"""

from __future__ import annotations
from collections.abc import Sequence
from dataclasses import asdict, dataclass
import json
import logging
import math
from pathlib import Path
from typing import Any, Mapping
from promptolution.utils.callbacks import BaseCallback
from promptolution.utils.templates import DEFAULT_SYS_PROMPT
from promptolution.utils.token_counter import get_token_counter

LOGGER = logging.getLogger(__name__)


class BudgetExhausted(RuntimeError):
    """Raised before an atomic downstream call that cannot fit in the budget."""

    def __init__(
        self,
        *,
        operation: str,
        estimated_tokens: int,
        remaining_tokens: int,
        reason: str = "next_atomic_operation_exceeds_budget",
    ) -> None:
        self.operation = str(operation)
        self.estimated_tokens = int(estimated_tokens)
        self.remaining_tokens = int(remaining_tokens)
        self.reason = str(reason)
        super().__init__(
            f"{self.reason}: operation={self.operation!r}, estimated={self.estimated_tokens}, remaining={self.remaining_tokens}"
        )


@dataclass(frozen=True)
class BudgetEvent:
    """One budget admission or completed-call event."""

    event: str
    operation: str
    before_tokens: int
    after_tokens: int
    estimated_tokens: int
    actual_tokens: int
    remaining_tokens: int
    admitted: bool


class TokenBudgetController:
    """Pre-call hard controller for downstream/evaluation-model tokens."""

    STATE_VERSION = 1

    def __init__(
        self,
        *,
        llm: Any,
        max_tokens: int,
        max_output_tokens: int,
        near_budget_fraction: float = 0.9,
        output_token_reserve: int = 2,
        event_log_path: str | Path | None = None,
    ) -> None:
        max_tokens = int(max_tokens)
        max_output_tokens = int(max_output_tokens)
        output_token_reserve = int(output_token_reserve)
        near_budget_fraction = float(near_budget_fraction)
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if output_token_reserve < 0:
            raise ValueError("output_token_reserve cannot be negative")
        if not 0.0 < near_budget_fraction < 1.0:
            raise ValueError("near_budget_fraction must lie strictly between 0 and 1")
        self.llm = llm
        self.max_tokens = max_tokens
        self.max_output_tokens = max_output_tokens
        self.near_budget_fraction = near_budget_fraction
        self.output_token_reserve = output_token_reserve
        self.event_log_path = Path(event_log_path) if event_log_path else None
        if self.event_log_path is not None:
            self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._count_text = get_token_counter(llm)
        self.stop_requested = False
        self.stopping_reason: str | None = None
        self.stopping_operation: str | None = None
        self.estimated_next_operation_tokens: int | None = None
        self.entered_near_budget_at_tokens: int | None = None
        self.rejected_operations = 0
        self.reduced_iterations = 0
        self.trimmed_candidates = 0
        self.completed_downstream_calls = 0
        self.maximum_admitted_estimate = 0
        self.maximum_actual_call_tokens = 0

    def token_counts(self) -> dict[str, int]:
        """Return the LLM counters in canonical promptolution form."""
        if hasattr(self.llm, "get_token_count"):
            raw = self.llm.get_token_count()
            input_tokens = int(raw.get("input_tokens", 0))
            output_tokens = int(raw.get("output_tokens", 0))
        else:
            input_tokens = int(getattr(self.llm, "input_token_count", 0))
            output_tokens = int(getattr(self.llm, "output_token_count", 0))
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

    @property
    def consumed_tokens(self) -> int:
        return int(self.token_counts()["total_tokens"])

    @property
    def remaining_tokens(self) -> int:
        return max(0, int(self.max_tokens - self.consumed_tokens))

    @property
    def utilization(self) -> float:
        return float(self.consumed_tokens / self.max_tokens)

    @property
    def near_budget_mode(self) -> bool:
        return self.utilization >= self.near_budget_fraction

    def _normalise_system_prompts(
        self, system_prompts: str | Sequence[str] | None, n: int
    ) -> list[str]:
        if n < 0:
            raise ValueError("n cannot be negative")
        if system_prompts is None:
            return [str(DEFAULT_SYS_PROMPT)] * n
        if isinstance(system_prompts, str):
            return [system_prompts] * n
        values = [str(value) for value in system_prompts]
        if len(values) != n:
            raise ValueError(
                f"system prompt count {len(values)} does not match input count {n}"
            )
        return values

    def estimate_call(
        self,
        *,
        inputs: Sequence[str],
        system_prompts: str | Sequence[str] | None = None,
        max_output_tokens: int | None = None,
    ) -> int:
        """Conservative upper bound in the exact promptolution accounting convention.

        promptolution's BaseLLM updates the input counter by tokenizing the raw user
        inputs and system prompts separately.  The output counter is updated by
        tokenizing generated response strings.  Reserving max_tokens plus a small
        tokenizer margin per response therefore guarantees that an admitted call
        cannot cross the configured hard budget under the current VLLM backend.
        """
        raw_inputs = [str(value) for value in inputs]
        systems = self._normalise_system_prompts(system_prompts, len(raw_inputs))
        output_cap = (
            self.max_output_tokens
            if max_output_tokens is None
            else int(max_output_tokens)
        )
        if output_cap <= 0:
            raise ValueError("max_output_tokens must be positive")
        input_tokens = sum((int(self._count_text(value)) for value in raw_inputs))
        input_tokens += sum((int(self._count_text(value)) for value in systems))
        output_tokens = len(raw_inputs) * (output_cap + self.output_token_reserve)
        return int(input_tokens + output_tokens)

    def _append_event(self, event: BudgetEvent) -> None:
        if self.event_log_path is None:
            return
        with self.event_log_path.open("a", encoding="utf-8") as handle:
            json.dump(asdict(event), handle, sort_keys=True)
            handle.write("\n")

    def request_stop(
        self, *, reason: str, operation: str, estimated_tokens: int
    ) -> None:
        """Record the first scientifically meaningful stopping reason."""
        self.stop_requested = True
        if self.stopping_reason is None:
            self.stopping_reason = str(reason)
            self.stopping_operation = str(operation)
            self.estimated_next_operation_tokens = int(estimated_tokens)

    def admit_or_raise(
        self,
        *,
        estimated_tokens: int,
        operation: str,
        reason: str = "next_atomic_operation_exceeds_budget",
    ) -> None:
        """Admit one atomic call or raise before any model work begins."""
        estimated_tokens = int(estimated_tokens)
        before = self.consumed_tokens
        remaining = max(0, self.max_tokens - before)
        admitted = estimated_tokens <= remaining
        self._append_event(
            BudgetEvent(
                event="admission",
                operation=str(operation),
                before_tokens=before,
                after_tokens=before,
                estimated_tokens=estimated_tokens,
                actual_tokens=0,
                remaining_tokens=remaining,
                admitted=admitted,
            )
        )
        if not admitted:
            self.rejected_operations += 1
            self.request_stop(
                reason=reason, operation=operation, estimated_tokens=estimated_tokens
            )
            raise BudgetExhausted(
                operation=operation,
                estimated_tokens=estimated_tokens,
                remaining_tokens=remaining,
                reason=reason,
            )
        self.maximum_admitted_estimate = max(
            self.maximum_admitted_estimate, estimated_tokens
        )

    def observe_completed_call(
        self, *, before_tokens: int, estimated_tokens: int, operation: str
    ) -> None:
        """Record actual use and fail loudly if the hard invariant is violated."""
        after = self.consumed_tokens
        actual = int(after - int(before_tokens))
        if actual < 0:
            raise RuntimeError("Downstream token counter decreased during a model call")
        if after > self.max_tokens:
            raise RuntimeError(
                f"Strict downstream-token budget was violated despite pre-call admission: actual={after}, budget={self.max_tokens}, operation={operation!r}, estimate={estimated_tokens}"
            )
        self.completed_downstream_calls += 1
        self.maximum_actual_call_tokens = max(self.maximum_actual_call_tokens, actual)
        self._append_event(
            BudgetEvent(
                event="completed_call",
                operation=str(operation),
                before_tokens=int(before_tokens),
                after_tokens=after,
                estimated_tokens=int(estimated_tokens),
                actual_tokens=actual,
                remaining_tokens=max(0, self.max_tokens - after),
                admitted=True,
            )
        )

    def suggested_candidate_count(self, configured_count: int) -> int:
        """Smoothly reduce challenger count after the near-budget threshold."""
        configured_count = int(configured_count)
        if configured_count <= 0:
            raise ValueError("configured_count must be positive")
        if not self.near_budget_mode:
            return configured_count
        if self.entered_near_budget_at_tokens is None:
            self.entered_near_budget_at_tokens = self.consumed_tokens
        phase_width = 1.0 - self.near_budget_fraction
        phase_remaining = max(0.0, 1.0 - self.utilization) / phase_width
        scaled = configured_count * phase_remaining
        suggested = max(1, min(configured_count, int(math.ceil(scaled - 1e-12))))
        if suggested < configured_count:
            self.reduced_iterations += 1
        return suggested

    def note_candidate_trim(self, *, requested: int, selected: int) -> None:
        requested = int(requested)
        selected = int(selected)
        if selected < requested:
            self.trimmed_candidates += requested - selected

    def fit_fairness_prompts(
        self,
        *,
        task: Any,
        prompts: Sequence[Any],
        eval_strategy: str | None = None,
        block_idx: int | Sequence[int] | None = None,
        system_prompts: str | Sequence[str] | None = None,
    ) -> list[Any]:
        """Return a deterministic prefix whose complete task evaluations fit.

        This is used by full-development NSGA-II.  A prompt is selected only when
        every currently missing prompt×row prediction for that prompt fits.  The
        function never admits a partial fairness unit.
        """
        prompt_list = list(prompts)
        if not prompt_list:
            return []
        positions, _ = task._resolve_positions(
            eval_strategy=eval_strategy, block_idx=block_idx
        )
        available = self.remaining_tokens
        planned_keys = set(getattr(task, "pred_cache", {}).keys())
        selected: list[Any] = []
        for prompt in prompt_list:
            prompt_string = prompt.construct_prompt()
            inputs: list[str] = []
            new_keys: list[tuple[str, str]] = []
            for position in positions:
                key = (prompt_string, str(task.ids[position]))
                if key in planned_keys:
                    continue
                inputs.append(f"{prompt_string}\n{task.xs[position]}")
                new_keys.append(key)
            estimate = (
                self.estimate_call(inputs=inputs, system_prompts=system_prompts)
                if inputs
                else 0
            )
            if estimate > available:
                if self.estimated_next_operation_tokens is None:
                    self.estimated_next_operation_tokens = int(estimate)
                break
            selected.append(prompt)
            available -= estimate
            planned_keys.update(new_keys)
        self.note_candidate_trim(requested=len(prompt_list), selected=len(selected))
        return selected

    def summary(self) -> dict[str, Any]:
        counts = self.token_counts()
        return {
            "controller_version": self.STATE_VERSION,
            "strict_budget_enabled": True,
            "requested_budget_tokens": self.max_tokens,
            "actual_downstream_tokens": counts["total_tokens"],
            "actual_input_tokens": counts["input_tokens"],
            "actual_output_tokens": counts["output_tokens"],
            "remaining_tokens": max(0, self.max_tokens - counts["total_tokens"]),
            "budget_difference_tokens": counts["total_tokens"] - self.max_tokens,
            "budget_utilization": counts["total_tokens"] / self.max_tokens,
            "near_budget_fraction": self.near_budget_fraction,
            "near_budget_mode_entered_at_tokens": self.entered_near_budget_at_tokens,
            "output_token_reserve_per_call": self.output_token_reserve,
            "stopping_reason": self.stopping_reason,
            "stopping_operation": self.stopping_operation,
            "estimated_next_operation_tokens": self.estimated_next_operation_tokens,
            "rejected_operations": self.rejected_operations,
            "reduced_iterations": self.reduced_iterations,
            "trimmed_candidates": self.trimmed_candidates,
            "completed_downstream_calls": self.completed_downstream_calls,
            "maximum_admitted_estimate": self.maximum_admitted_estimate,
            "maximum_actual_call_tokens": self.maximum_actual_call_tokens,
        }

    def state_dict(self) -> dict[str, Any]:
        """Checkpointable controller state; the LLM itself is not serialized."""
        return {
            "version": self.STATE_VERSION,
            "max_tokens": self.max_tokens,
            "max_output_tokens": self.max_output_tokens,
            "near_budget_fraction": self.near_budget_fraction,
            "output_token_reserve": self.output_token_reserve,
            "stop_requested": self.stop_requested,
            "stopping_reason": self.stopping_reason,
            "stopping_operation": self.stopping_operation,
            "estimated_next_operation_tokens": self.estimated_next_operation_tokens,
            "entered_near_budget_at_tokens": self.entered_near_budget_at_tokens,
            "rejected_operations": self.rejected_operations,
            "reduced_iterations": self.reduced_iterations,
            "trimmed_candidates": self.trimmed_candidates,
            "completed_downstream_calls": self.completed_downstream_calls,
            "maximum_admitted_estimate": self.maximum_admitted_estimate,
            "maximum_actual_call_tokens": self.maximum_actual_call_tokens,
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        version = int(payload.get("version", -1))
        if version != self.STATE_VERSION:
            raise ValueError(f"Unsupported budget-controller state version {version}")
        saved_max_tokens = int(payload.get("max_tokens", -1))
        if saved_max_tokens <= 0 or self.max_tokens < saved_max_tokens:
            raise ValueError(
                f"A resumed strict-budget stage cannot lower the token budget: saved={saved_max_tokens}, current={self.max_tokens}"
            )
        immutable = {
            "max_output_tokens": self.max_output_tokens,
            "near_budget_fraction": self.near_budget_fraction,
            "output_token_reserve": self.output_token_reserve,
        }
        for key, current in immutable.items():
            saved = payload.get(key)
            if float(saved) != float(current):
                raise ValueError(
                    f"Budget-controller checkpoint mismatch for {key}: saved={saved!r}, current={current!r}"
                )
        extending_budget = self.max_tokens > saved_max_tokens
        self.stop_requested = (
            False if extending_budget else bool(payload.get("stop_requested", False))
        )
        self.stopping_reason = (
            None if extending_budget else payload.get("stopping_reason")
        )
        self.stopping_operation = (
            None if extending_budget else payload.get("stopping_operation")
        )
        estimated = (
            None if extending_budget else payload.get("estimated_next_operation_tokens")
        )
        self.estimated_next_operation_tokens = (
            None if estimated is None else int(estimated)
        )
        entered = (
            None if extending_budget else payload.get("entered_near_budget_at_tokens")
        )
        self.entered_near_budget_at_tokens = None if entered is None else int(entered)
        for name in (
            "rejected_operations",
            "reduced_iterations",
            "trimmed_candidates",
            "completed_downstream_calls",
            "maximum_admitted_estimate",
            "maximum_actual_call_tokens",
        ):
            setattr(self, name, int(payload.get(name, 0)))

    def write_summary(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(self.summary(), handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(target)
        return target


class BudgetAwarePredictor:
    """Transparent predictor adapter that protects every downstream LLM call."""

    def __init__(self, predictor: Any, controller: TokenBudgetController) -> None:
        self._predictor = predictor
        self.controller = controller
        self.llm = predictor.llm

    def __getattr__(self, name: str) -> Any:
        return getattr(self._predictor, name)

    def predict(
        self,
        prompts: str | Sequence[str],
        xs: Sequence[str],
        system_prompts: str | Sequence[str] | None = None,
    ):
        prompt_list = [prompts] if isinstance(prompts, str) else list(prompts)
        x_list = [str(value) for value in xs]
        if len(prompt_list) != len(x_list):
            raise ValueError(
                f"prompt count {len(prompt_list)} does not match input count {len(x_list)}"
            )
        raw_inputs = [f"{str(prompt)}\n{x}" for prompt, x in zip(prompt_list, x_list)]
        estimate = self.controller.estimate_call(
            inputs=raw_inputs, system_prompts=system_prompts
        )
        operation = f"predictor_batch[{len(raw_inputs)}]"
        self.controller.admit_or_raise(estimated_tokens=estimate, operation=operation)
        before = self.controller.consumed_tokens
        try:
            return self._predictor.predict(
                prompts=prompt_list, xs=x_list, system_prompts=system_prompts
            )
        finally:
            self.controller.observe_completed_call(
                before_tokens=before, estimated_tokens=estimate, operation=operation
            )


class BudgetStopCallback(BaseCallback):
    """Stop the optimizer loop after a pre-call admission rejection."""

    def __init__(self, controller: TokenBudgetController) -> None:
        self.controller = controller

    def on_step_end(self, optimizer: Any) -> bool:
        return not self.controller.stop_requested

    def on_train_end(self, optimizer: Any) -> bool:
        return True
