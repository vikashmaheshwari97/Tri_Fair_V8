"""Atomic, resumable checkpoints for long Tri-Fair budget ladders.

PyTorch is imported lazily. This keeps configuration inspection, CLI ``--help``,
and CPU-only metric tests usable on machines where CUDA/PyTorch is unavailable
or where the local Windows PyTorch installation is broken. On the Linux GPU
cluster, torch RNG state is captured and restored normally.
"""

from __future__ import annotations
import importlib
import logging
import os
import pickle
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import numpy as np
from promptolution.utils.callbacks import BaseCallback
from promptolution.utils.prompt import Prompt
from tri_fair.runtime.budget import BudgetExhausted

logger = logging.getLogger(__name__)
CHECKPOINT_VERSION = 5


def _load_torch(*, required: bool = False):
    """Import torch without making module import depend on a working torch DLL."""
    try:
        return importlib.import_module("torch")
    except (ImportError, OSError) as error:
        if required:
            raise RuntimeError(
                "PyTorch could not be imported. Local analysis/tests can run without it, but model execution requires a working PyTorch installation."
            ) from error
        logger.debug("PyTorch unavailable; torch RNG state will be skipped: %s", error)
        return None


def _prompt_to_dict(prompt: Prompt) -> Dict[str, Any]:
    return {
        "instruction": prompt.instruction,
        "few_shots": list(prompt.few_shots),
        "downstream_template": prompt.downstream_template,
    }


def _prompt_from_dict(payload: Dict[str, Any]) -> Prompt:
    return Prompt(
        instruction=payload["instruction"],
        few_shots=list(payload.get("few_shots", [])),
        downstream_template=payload.get("downstream_template"),
    )


def _token_count(llm: Any) -> Dict[str, int]:
    return {
        "input": int(getattr(llm, "input_token_count", 0)),
        "output": int(getattr(llm, "output_token_count", 0)),
    }


def _set_token_count(llm: Any, payload: Dict[str, int]) -> None:
    llm.input_token_count = int(payload.get("input", 0))
    llm.output_token_count = int(payload.get("output", 0))


def _capture_torch_rng() -> Dict[str, Any]:
    torch = _load_torch(required=False)
    if torch is None:
        return {"available": False, "cpu": None, "cuda": None}
    cuda_available = bool(torch.cuda.is_available())
    return {
        "available": True,
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if cuda_available else None,
    }


def _restore_torch_rng(payload: Optional[Dict[str, Any]]) -> None:
    if not payload or not payload.get("available", False):
        return
    torch = _load_torch(required=False)
    if torch is None:
        logger.warning(
            "Checkpoint contains PyTorch RNG state, but torch is unavailable locally; Python and NumPy RNG state were still restored."
        )
        return
    if payload.get("cpu") is not None:
        torch.set_rng_state(payload["cpu"])
    if torch.cuda.is_available() and payload.get("cuda") is not None:
        torch.cuda.set_rng_state_all(payload["cuda"])


def _task_signature(task: Any) -> str:
    """Bind prediction caches to the exact ordered development data and metric."""
    import hashlib
    import json

    payload = {
        "data": task.df.to_json(orient="split", index=False, date_format="iso"),
        "ids": list(task.ids),
        "blocks": task.blocks,
        "classes": task.classes,
        "metric": task.fairness_metric_name,
        "quality": task.quality_metric_name,
        "fairness_kwargs": task.fairness_kwargs,
        "min_group_count": task.min_group_count,
        "protected_columns": task.protected_columns,
        "metadata_columns": task.metadata_columns,
        "task_description": task.task_description,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def capture_optimizer_state(optimizer: Any) -> Dict[str, Any]:
    task = optimizer.task
    prompt_blocks = {
        prompt.construct_prompt(): list(blocks)
        for prompt, blocks in getattr(task, "prompt_evaluated_blocks", {}).items()
    }
    return {
        "version": CHECKPOINT_VERSION,
        "task_signature": _task_signature(task),
        "optimizer_class": optimizer.__class__.__name__,
        "prompts": [_prompt_to_dict(prompt) for prompt in optimizer.prompts],
        "incumbents": [
            _prompt_to_dict(prompt) for prompt in getattr(optimizer, "incumbents", [])
        ],
        "non_incumbents": [
            _prompt_to_dict(prompt)
            for prompt in getattr(optimizer, "non_incumbents", [])
        ],
        "scores": getattr(optimizer, "scores", []),
        "current_step": int(getattr(optimizer, "current_step", 0)),
        "completed_steps": int(getattr(optimizer, "_completed_steps", 0)),
        "runhistory": getattr(optimizer, "runhistory", None),
        "global_min_bounds": getattr(optimizer, "global_min_bounds", None),
        "global_max_bounds": getattr(optimizer, "global_max_bounds", None),
        "aggregate_records": getattr(optimizer, "aggregate_records", None),
        "optimizer_extra": optimizer.optimizer_state_dict()
        if hasattr(optimizer, "optimizer_state_dict")
        else None,
        "population_metrics": getattr(optimizer, "_population_metrics", None),
        "budget_controller": optimizer.budget_controller.state_dict()
        if getattr(optimizer, "budget_controller", None) is not None
        else None,
        "task": {
            "token_count_cache": getattr(task, "token_count_cache", None),
            "pred_cache": getattr(task, "pred_cache", None),
            "sequence_cache": getattr(task, "sequence_cache", None),
            "eval_cache": getattr(task, "eval_cache", None),
            "last_records": getattr(task, "last_records", None),
            "prompt_evaluated_blocks": prompt_blocks,
            "block_idx": int(getattr(task, "block_idx", 0)),
            "rng_state": getattr(
                getattr(task, "rng", None), "bit_generator", None
            ).state
            if getattr(task, "rng", None) is not None
            else None,
        },
        "tokens": {
            "downstream": _token_count(optimizer.predictor.llm),
            "meta": _token_count(optimizer.meta_llm)
            if hasattr(optimizer, "meta_llm")
            else None,
        },
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": _capture_torch_rng(),
        },
    }


def restore_optimizer_state(optimizer: Any, state: Dict[str, Any]) -> None:
    version = int(state.get("version", -1))
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version {state.get('version')}; expected one of 1, 2, {CHECKPOINT_VERSION}"
        )
    if state.get("optimizer_class") != optimizer.__class__.__name__:
        raise ValueError(
            f"Checkpoint was created by {state.get('optimizer_class')}, not {optimizer.__class__.__name__}"
        )
    if state.get("task_signature") is not None and state[
        "task_signature"
    ] != _task_signature(optimizer.task):
        raise ValueError("checkpoint development data or metric configuration differs")
    optimizer.prompts = [_prompt_from_dict(value) for value in state["prompts"]]
    optimizer.incumbents = [
        _prompt_from_dict(value) for value in state.get("incumbents", [])
    ]
    optimizer.non_incumbents = [
        _prompt_from_dict(value) for value in state.get("non_incumbents", [])
    ]
    optimizer.scores = state.get("scores", [])
    optimizer.current_step = int(state.get("current_step", 0))
    optimizer._completed_steps = int(state.get("completed_steps", 0))
    if state.get("runhistory") is not None:
        optimizer.runhistory = state["runhistory"]
    optimizer.global_min_bounds = state.get("global_min_bounds")
    optimizer.global_max_bounds = state.get("global_max_bounds")
    if state.get("aggregate_records") is not None:
        optimizer.aggregate_records = state["aggregate_records"]
    if state.get("population_metrics") is not None:
        optimizer._population_metrics = state["population_metrics"]
    task_state = state.get("task", {})
    task = optimizer.task
    for name in (
        "pred_cache",
        "sequence_cache",
        "eval_cache",
        "last_records",
        "token_count_cache",
    ):
        if task_state.get(name) is not None:
            setattr(task, name, task_state[name])
    if hasattr(task, "sequence_cache"):
        task.seq_cache = task.sequence_cache
    task.block_idx = int(task_state.get("block_idx", 0))
    if (
        task_state.get("rng_state") is not None
        and getattr(task, "rng", None) is not None
    ):
        task.rng.bit_generator.state = task_state["rng_state"]
    all_prompts = optimizer.incumbents + optimizer.non_incumbents + optimizer.prompts
    by_string = {prompt.construct_prompt(): prompt for prompt in all_prompts}
    task.prompt_evaluated_blocks = {
        by_string[prompt_string]: list(blocks)
        for prompt_string, blocks in task_state.get(
            "prompt_evaluated_blocks", {}
        ).items()
        if prompt_string in by_string
    }
    _set_token_count(optimizer.predictor.llm, state["tokens"]["downstream"])
    if state["tokens"].get("meta") is not None and hasattr(optimizer, "meta_llm"):
        _set_token_count(optimizer.meta_llm, state["tokens"]["meta"])
    controller_state = state.get("budget_controller")
    controller = getattr(optimizer, "budget_controller", None)
    if controller_state is not None:
        if controller is None:
            raise ValueError(
                "Checkpoint contains strict-budget state, but the current run did not create a budget controller"
            )
        controller.load_state_dict(controller_state)
    extra = state.get("optimizer_extra")
    if extra is not None and hasattr(optimizer, "load_optimizer_state_dict"):
        optimizer.load_optimizer_state_dict(extra)
        if hasattr(optimizer, "_bootstrap_cache"):
            optimizer._bootstrap_cache.clear()
    elif getattr(optimizer, "method_version", "").startswith(("7.1", "8.")):
        raise ValueError(
            "This optimizer requires a complete clean-package checkpoint; start a fresh run for the corrected method"
        )
    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])
    if version == 1:
        legacy_payload = {
            "available": state["rng"].get("torch_cpu") is not None,
            "cpu": state["rng"].get("torch_cpu"),
            "cuda": state["rng"].get("torch_cuda"),
        }
        _restore_torch_rng(legacy_payload)
    else:
        _restore_torch_rng(state["rng"].get("torch"))


def save_checkpoint(optimizer: Any, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + f".tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        pickle.dump(
            capture_optimizer_state(optimizer), handle, protocol=pickle.HIGHEST_PROTOCOL
        )
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(target)
    return target


def load_checkpoint(optimizer: Any, path: str | Path) -> None:
    with Path(path).open("rb") as handle:
        state = pickle.load(handle)
    restore_optimizer_state(optimizer, state)
    optimizer._resume_loaded = True
    logger.info(
        "Restored %s from %s at %d downstream tokens",
        optimizer.__class__.__name__,
        path,
        sum(_token_count(optimizer.predictor.llm).values()),
    )


class ResumableOptimizerMixin:
    """BaseOptimizer-compatible loop that skips initialization after restore."""

    _resume_loaded: bool = False
    _completed_steps: int = 0

    def load_checkpoint(self, path: str | Path) -> None:
        load_checkpoint(self, path)

    _adaptive_fields = (
        "initial_common_blocks",
        "last_step_mode",
        "_confirmation_streak",
        "smart_start_prompt_keys",
        "contrastive_modes",
        "_front_signature",
        "_stagnation_steps",
        "_mode_usage",
        "variation_modes",
        "mode_attempts",
        "mode_successes",
        "candidate_modes",
        "_archive_signature",
        "_archive_stagnation_steps",
        "pending_rewards",
        "seen_candidates",
        "operator_history",
        "demo_history",
        "nsga3_survival",
        "_spea2_fitness",
        "random_seed_pool",
    )

    def optimizer_state_dict(self) -> Dict[str, Any]:
        from copy import deepcopy

        configuration = {
            k: deepcopy(getattr(self, k))
            for k in (
                "enable_feedback",
                "enable_demo_search",
                "uncertainty_resamples",
                "uncertainty_alpha",
                "feedback_cases",
                "full_evaluation",
                "effective_parameters",
            )
            if hasattr(self, k)
        }
        return {
            "version": 1,
            "configuration": configuration,
            "method_version": getattr(self, "method_version", None),
            "fields": {
                k: deepcopy(getattr(self, k))
                for k in self._adaptive_fields
                if hasattr(self, k)
            },
        }

    def load_optimizer_state_dict(self, payload: Dict[str, Any]) -> None:
        from copy import deepcopy

        if payload.get("version") != 1 or payload.get("method_version") != getattr(
            self, "method_version", None
        ):
            raise ValueError("checkpoint method version does not match this optimizer")
        for key, value in payload.get("configuration", {}).items():
            if not hasattr(self, key) or getattr(self, key) != value:
                raise ValueError(f"checkpoint optimizer configuration differs: {key}")
        fields = payload.get("fields", {})
        if set(fields) - set(self._adaptive_fields):
            raise ValueError("unknown adaptive checkpoint fields")
        for key, value in fields.items():
            setattr(self, key, deepcopy(value))

    def optimize(self, n_steps: int):
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        if not self._resume_loaded:
            self._pre_optimization_loop()
            self._completed_steps = 0
        controller = getattr(self, "budget_controller", None)
        try:
            for _ in range(self._completed_steps, n_steps):
                original_crossovers = getattr(self, "crossovers_per_iter", None)
                if controller is not None and original_crossovers is not None:
                    self.crossovers_per_iter = controller.suggested_candidate_count(
                        int(original_crossovers)
                    )
                try:
                    self.prompts = self._step()
                except BudgetExhausted as error:
                    logger.info("Strict budget stop before model call: %s", error)
                    break
                finally:
                    if original_crossovers is not None:
                        self.crossovers_per_iter = original_crossovers
                self._completed_steps += 1
                if not self._on_step_end():
                    break
                if controller is not None and controller.stop_requested:
                    break
        except Exception:
            logger.exception("Resumable optimization step failed")
            raise
        finally:
            self._on_train_end()
        return self.prompts


class OptimizerCheckpointCallback(BaseCallback):
    """Write atomic snapshots when downstream-token thresholds are crossed."""

    def __init__(
        self,
        output_dir: str | Path,
        checkpoints: Iterable[int],
        latest_every_steps: int = 10,
    ) -> None:
        self.output_dir = Path(output_dir) / "checkpoints"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints = sorted(
            {int(value) for value in checkpoints if int(value) > 0}
        )
        self.emitted: set[int] = {
            threshold
            for threshold in self.checkpoints
            if (self.output_dir / f"tokens_{threshold}.pkl").exists()
        }
        self.latest_every_steps = max(1, int(latest_every_steps))
        self._steps_seen = 0

    @staticmethod
    def _current_tokens(optimizer: Any) -> int:
        counts = optimizer.predictor.llm.get_token_count()
        return int(counts["total_tokens"])

    def on_step_end(self, optimizer: Any) -> bool:
        self._steps_seen += 1
        current = self._current_tokens(optimizer)
        for threshold in self.checkpoints:
            if threshold <= current and threshold not in self.emitted:
                path = self.output_dir / f"tokens_{threshold}.pkl"
                save_checkpoint(optimizer, path)
                self.emitted.add(threshold)
                logger.info(
                    "Saved checkpoint %s at actual token count %d", path, current
                )
        if self._steps_seen % self.latest_every_steps == 0:
            save_checkpoint(optimizer, self.output_dir / "latest.pkl")
        return True

    def on_train_end(self, optimizer: Any) -> bool:
        save_checkpoint(optimizer, self.output_dir / "latest.pkl")
        return True
