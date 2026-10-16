from __future__ import annotations
from pathlib import Path
import pandas as pd
from tri_fair.cli.common import stable_latest_per_prompt, prompt_id


def load_prompt_candidates(
    log_path: str | Path, *, step: int, selection: str
) -> tuple[pd.DataFrame, int]:
    path = Path(log_path) / "step_results.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    required = {"step", "prompt"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path} is empty")
    frame = frame.copy()
    frame["step"] = pd.to_numeric(frame["step"], errors="raise").astype(int)
    chosen_step = int(frame["step"].max()) if int(step) == -1 else int(step)
    if chosen_step not in set(frame["step"]):
        available = sorted(frame["step"].unique().tolist())
        raise ValueError(
            f"Step {chosen_step} is absent from {path}; available range is {available[0]}..{available[-1]}"
        )
    current = frame[frame["step"] == chosen_step].copy()
    incumbent_history = pd.DataFrame(columns=frame.columns)
    if "is_incumbent" in frame:
        flags = frame["is_incumbent"].fillna(False).astype(bool)
        incumbent_history = frame[(frame["step"] <= chosen_step) & flags].copy()
    if selection == "current":
        selected = current
    elif selection == "incumbents":
        selected = incumbent_history
    elif selection == "current_and_incumbents":
        selected = pd.concat(
            [current, incumbent_history], ignore_index=True, sort=False
        )
    else:
        raise ValueError(f"Unknown selection policy: {selection}")
    if selected.empty:
        raise ValueError(
            f"Selection policy {selection!r} produced no prompts at step {chosen_step}"
        )
    if "prompt_id" not in selected:
        selected["prompt_id"] = selected["prompt"].astype(str).map(prompt_id)
    selected = stable_latest_per_prompt(selected)
    selected["chosen_step"] = chosen_step
    selected["selection_policy"] = selection
    return (selected.reset_index(drop=True), chosen_step)


def _rename_development_columns(frame: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "quality": "dev_quality",
        "cost": "dev_cost",
        "fairness": "dev_fairness",
        "fairness_ready": "dev_fairness_ready",
        "fairness_diagnostics_json": "dev_fairness_diagnostics_json",
        "group_support_json": "dev_group_support_json",
        "objective_vector": "dev_objective_vector",
    }
    safe_mapping = {
        source: target
        for source, target in mapping.items()
        if source in frame.columns and target not in frame.columns
    }
    return frame.rename(columns=safe_mapping)


def _step_token_table(frame: pd.DataFrame) -> pd.Series:
    if "total_tokens_downstream" in frame:
        tokens = pd.to_numeric(frame["total_tokens_downstream"], errors="coerce")
    else:
        input_source = (
            frame["input_tokens_downstream"]
            if "input_tokens_downstream" in frame
            else pd.Series(0, index=frame.index, dtype=float)
        )
        output_source = (
            frame["output_tokens_downstream"]
            if "output_tokens_downstream" in frame
            else pd.Series(0, index=frame.index, dtype=float)
        )
        input_tokens = pd.to_numeric(input_source, errors="coerce").fillna(0)
        output_tokens = pd.to_numeric(output_source, errors="coerce").fillna(0)
        tokens = input_tokens + output_tokens
    work = frame.assign(_tokens=tokens)
    return work.groupby("step", sort=True)["_tokens"].max().dropna().astype(int)
