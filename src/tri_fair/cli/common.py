"""Shared helpers for Tri-Fair command-line scripts.

The helpers in this module deliberately avoid optimizer-specific logic.  They
centralize atomic artifact writes, prompt serialization, result conversion,
manifest locking, and validation so the experiment and evaluation entry points
produce one consistent schema.
"""

from __future__ import annotations
from collections.abc import Sequence
import argparse
import contextlib
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, MutableMapping
import numpy as np
import pandas as pd
from promptolution.utils import Prompt
from tri_fair.data.task import FairnessEvalResult

LOGGER = logging.getLogger(__name__)


def utc_now_iso() -> str:
    """Return an RFC-3339-compatible UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def configure_logging(level: str = "INFO") -> None:
    numeric = getattr(logging, str(level).upper(), None)
    if not isinstance(numeric, int):
        raise ValueError(f"Unknown logging level: {level!r}")
    logging.basicConfig(
        level=numeric, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def parse_positive_int_csv(raw: str, *, include: int | None = None) -> list[int]:
    values: set[int] = set()
    for part in str(raw).split(","):
        part = part.strip().replace("_", "")
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError(f"Checkpoint values must be positive, got {value}")
        values.add(value)
    if include is not None:
        if int(include) <= 0:
            raise ValueError("Included checkpoint must be positive")
        values.add(int(include))
    if not values:
        raise ValueError("At least one checkpoint is required")
    return sorted(values)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return str(value)


def json_dumps(value: Any) -> str:
    return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True)


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(dict(payload)), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(target)
    return target


def append_jsonl(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json_dumps(dict(payload)) + "\n"
    with target.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
    return target


def atomic_write_parquet(frame: pd.DataFrame, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + f".tmp-{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    temporary.replace(target)
    return target


def sha256_file(path: str | Path) -> str | None:
    source = Path(path)
    if not source.exists() or not source.is_file():
        return None
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_id(prompt: Prompt | str) -> str:
    text = prompt.construct_prompt() if isinstance(prompt, Prompt) else str(prompt)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompts_to_frame(prompts: Sequence[Prompt]) -> pd.DataFrame:
    rows = []
    for index, prompt in enumerate(prompts):
        full_prompt = prompt.construct_prompt()
        rows.append(
            {
                "prompt": full_prompt,
                "prompt_id": prompt_id(full_prompt),
                "instruction": prompt.instruction,
                "few_shots_json": json_dumps(list(prompt.few_shots)),
                "downstream_template": prompt.downstream_template,
                "prompt_order": index,
            }
        )
    return pd.DataFrame(rows)


def reconstruct_prompts(frame: pd.DataFrame) -> list[Prompt]:
    prompts: list[Prompt] = []
    structured = {"instruction", "few_shots_json", "downstream_template"}.issubset(
        frame.columns
    )
    for _, row in frame.iterrows():
        if structured and pd.notna(row.get("instruction")):
            raw_few_shots = row.get("few_shots_json")
            if raw_few_shots is None or (
                isinstance(raw_few_shots, float) and np.isnan(raw_few_shots)
            ):
                few_shots: list[str] = []
            elif isinstance(raw_few_shots, str):
                parsed = json.loads(raw_few_shots or "[]")
                if not isinstance(parsed, list):
                    raise TypeError("few_shots_json must decode to a list")
                few_shots = [str(value) for value in parsed]
            elif isinstance(raw_few_shots, (list, tuple, np.ndarray)):
                few_shots = [str(value) for value in raw_few_shots]
            else:
                raise TypeError(
                    f"Unsupported few-shot representation: {type(raw_few_shots)}"
                )
            downstream_template = row.get("downstream_template")
            prompts.append(
                Prompt(
                    instruction=str(row["instruction"]),
                    few_shots=few_shots,
                    downstream_template=str(downstream_template)
                    if downstream_template is not None and pd.notna(downstream_template)
                    else None,
                )
            )
        else:
            prompts.append(Prompt(instruction=str(row["prompt"])))
    return prompts


def set_generation_limit(llm: Any, max_tokens: int) -> None:
    max_tokens = int(max_tokens)
    if max_tokens <= 0:
        raise ValueError("Generation token limits must be positive")
    sampling_params = getattr(llm, "sampling_params", None)
    if sampling_params is None:
        raise TypeError(
            "The configured LLM does not expose sampling_params; cannot enforce the requested output-token limit"
        )
    sampling_params.max_tokens = max_tokens


def token_counts(llm: Any) -> dict[str, int]:
    if hasattr(llm, "get_token_count"):
        raw = llm.get_token_count()
        return {
            "input_tokens": int(raw.get("input_tokens", 0)),
            "output_tokens": int(raw.get("output_tokens", 0)),
            "total_tokens": int(raw.get("total_tokens", 0)),
        }
    input_tokens = int(getattr(llm, "input_token_count", 0))
    output_tokens = int(getattr(llm, "output_token_count", 0))
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def result_columns(result: Any, *, prefix: str, model_config: Any) -> dict[str, Any]:
    """Convert a task result to the canonical analysis schema.

    Objective vectors use the optimizer convention: maximize quality, maximize
    negative cost, and maximize negative unfairness.
    """
    quality = np.asarray(result.agg_scores, dtype=float)
    input_tokens = np.asarray(result.agg_input_tokens, dtype=float)
    output_tokens = np.asarray(result.agg_output_tokens, dtype=float)
    cost = (
        float(model_config.input_costs) * input_tokens
        + float(model_config.output_costs) * output_tokens
    )
    columns: dict[str, Any] = {
        f"{prefix}_quality": quality,
        f"{prefix}_cost": cost,
        f"{prefix}_input_tokens": input_tokens,
        f"{prefix}_output_tokens": output_tokens,
    }
    if isinstance(result, FairnessEvalResult):
        fairness = np.asarray(result.fairness_loss, dtype=float)
        ready = np.asarray(result.fairness_ready, dtype=bool)
        columns[f"{prefix}_fairness"] = fairness
        columns[f"{prefix}_fairness_ready"] = ready
        columns[f"{prefix}_fairness_diagnostics_json"] = [
            json_dumps(value) for value in result.fairness_diagnostics
        ]
        columns[f"{prefix}_group_support_json"] = [
            json_dumps(value) for value in result.fairness_support
        ]
        columns[f"{prefix}_objective_vector"] = [
            [float(q), -float(c), -float(f)] for q, c, f in zip(quality, cost, fairness)
        ]
    else:
        n = len(quality)
        columns[f"{prefix}_fairness"] = np.full(n, np.nan, dtype=float)
        columns[f"{prefix}_fairness_ready"] = np.zeros(n, dtype=bool)
        columns[f"{prefix}_fairness_diagnostics_json"] = ["{}"] * n
        columns[f"{prefix}_group_support_json"] = ["{}"] * n
        columns[f"{prefix}_objective_vector"] = [
            [float(q), -float(c)] for q, c in zip(quality, cost)
        ]
    return columns


def add_result_columns(
    frame: pd.DataFrame, result: Any, *, prefix: str, model_config: Any
) -> pd.DataFrame:
    if len(frame) != len(result.agg_scores):
        raise ValueError(
            f"Result length mismatch: frame has {len(frame)} rows but result has {len(result.agg_scores)}"
        )
    output = frame.reset_index(drop=True).copy()
    for name, values in result_columns(
        result, prefix=prefix, model_config=model_config
    ).items():
        output[name] = values
    return output


def stable_latest_per_prompt(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    work = frame.copy()
    if "prompt_id" not in work:
        work["prompt_id"] = work["prompt"].astype(str).map(prompt_id)
    sort_columns = [column for column in ("step", "prompt_order") if column in work]
    if sort_columns:
        work = work.sort_values(sort_columns, kind="stable")
    return (
        work.groupby("prompt_id", as_index=False, sort=False)
        .tail(1)
        .reset_index(drop=True)
    )


@contextlib.contextmanager
def directory_lock(
    lock_path: str | Path,
    *,
    timeout_seconds: float = 3600.0,
    poll_seconds: float = 1.0,
    stale_after_seconds: float = 6 * 3600.0,
) -> Iterator[None]:
    """Portable lock implemented with an atomically-created directory.

    This prevents concurrent SLURM tasks from racing while they create the same
    immutable split manifest.  The lock contains owner metadata and can recover
    from a stale directory left by a terminated job.
    """
    target = Path(lock_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    while True:
        try:
            target.mkdir()
            atomic_write_json(
                target / "owner.json",
                {
                    "pid": os.getpid(),
                    "host": os.environ.get("HOSTNAME", "unknown"),
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "created_at": utc_now_iso(),
                },
            )
            break
        except FileExistsError:
            try:
                age = time.time() - target.stat().st_mtime
            except FileNotFoundError:
                continue
            if age > stale_after_seconds:
                LOGGER.warning(
                    "Removing stale lock directory %s (age %.1fs)", target, age
                )
                for child in target.glob("*"):
                    child.unlink(missing_ok=True)
                try:
                    target.rmdir()
                except OSError:
                    pass
                continue
            if time.monotonic() - started > timeout_seconds:
                raise TimeoutError(f"Timed out waiting for lock {target}")
            time.sleep(poll_seconds)
    try:
        yield
    finally:
        for child in target.glob("*"):
            child.unlink(missing_ok=True)
        try:
            target.rmdir()
        except FileNotFoundError:
            pass


def manifest_lock_path(manifest_dir: str | Path, dataset: str, seed: int) -> Path:
    return Path(manifest_dir) / ".locks" / f"{dataset}_seed{int(seed)}.lock"


def resolve_manifest_path(manifest_dir: str | Path, dataset: str, seed: int) -> Path:
    return Path(manifest_dir) / f"{dataset}_seed{int(seed)}.json"


def merge_parquet_rows(
    new_rows: pd.DataFrame, path: str | Path, *, deduplicate_on: Sequence[str]
) -> pd.DataFrame:
    target = Path(path)
    if target.exists():
        existing = pd.read_parquet(target)
        combined = pd.concat([existing, new_rows], ignore_index=True, sort=False)
    else:
        combined = new_rows.copy()
    keys = [column for column in deduplicate_on if column in combined.columns]
    if keys:
        combined = combined.drop_duplicates(subset=keys, keep="last")
    combined = combined.reset_index(drop=True)
    atomic_write_parquet(combined, target)
    return combined


def compare_immutable_args(
    previous: Mapping[str, Any], current: Mapping[str, Any], keys: Sequence[str]
) -> None:
    mismatches: MutableMapping[str, tuple[Any, Any]] = {}
    for key in keys:
        if key in previous and previous.get(key) != current.get(key):
            mismatches[key] = (previous.get(key), current.get(key))
    if mismatches:
        details = ", ".join(
            (
                f"{key}: previous={before!r}, requested={after!r}"
                for key, (before, after) in mismatches.items()
            )
        )
        raise ValueError(f"Resume configuration mismatch: {details}")
