"""Leakage-safe, group-complete sampling for Tri-Fair.

All functions return stable manifest IDs rather than positional indices.  This
allows the generated manifests to survive dataframe reordering and makes exact
experiment reproduction straightforward.
"""

from __future__ import annotations
from collections.abc import Sequence
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable
import numpy as np
import pandas as pd

MANIFEST_ID = "_manifest_id"


@dataclass(frozen=True)
class SampledPartition:
    ids: list[str]
    remainder: pd.DataFrame


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataframe is missing required sampling columns: {sorted(missing)}"
        )


def _round_robin_rows(
    df: pd.DataFrame, *, strata_columns: Sequence[str], size: int, seed: int
) -> list[str]:
    """Balanced deterministic sampling without replacement.

    Rows are shuffled inside each stratum and selected round-robin.  This is
    more robust than ``groupby.sample(n=...)`` when some strata are small.
    """
    _require_columns(df, [MANIFEST_ID, *strata_columns])
    if size > len(df):
        raise ValueError(f"Requested {size} rows but only {len(df)} are available")
    rng = np.random.default_rng(seed)
    if not strata_columns:
        ids = df[MANIFEST_ID].astype(str).to_numpy(copy=True)
        rng.shuffle(ids)
        return ids[:size].tolist()
    groups: dict[tuple, list[str]] = {}
    grouped = df.groupby(list(strata_columns), dropna=False, sort=True)
    for key, group in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)
        ids = group[MANIFEST_ID].astype(str).to_numpy(copy=True)
        rng.shuffle(ids)
        groups[key_tuple] = ids.tolist()
    keys = list(groups)
    rng.shuffle(keys)
    selected: list[str] = []
    cursor = 0
    while len(selected) < size:
        progressed = False
        for offset in range(len(keys)):
            key = keys[(cursor + offset) % len(keys)]
            bucket = groups[key]
            if bucket:
                selected.append(bucket.pop())
                progressed = True
                if len(selected) == size:
                    break
        cursor = (cursor + 1) % max(1, len(keys))
        if not progressed:
            break
    if len(selected) != size:
        raise RuntimeError(
            f"Balanced sampler produced {len(selected)} of {size} requested rows"
        )
    return selected


def sample_bbq_partition(df: pd.DataFrame, *, size: int, seed: int) -> SampledPartition:
    _require_columns(df, [MANIFEST_ID, "category", "template_id"])
    if size % 4 != 0:
        raise ValueError(
            "BBQ partitions must be divisible by four to keep template quartets intact"
        )
    rng = np.random.default_rng(seed)
    units = (
        df.groupby("template_id", sort=True)
        .agg(category=("category", "first"), n=(MANIFEST_ID, "size"))
        .reset_index()
    )
    malformed = units[units["n"] != 4]
    if len(malformed):
        raise ValueError(
            f"BBQ template units must contain exactly four polarity/context variants; found malformed units: {malformed.head().to_dict(orient='records')}"
        )
    required_units = size // 4
    buckets: dict[str, list[str]] = {}
    for category, group in units.groupby("category", sort=True):
        values = group["template_id"].astype(str).to_numpy(copy=True)
        rng.shuffle(values)
        buckets[str(category)] = values.tolist()
    categories = sorted(buckets)
    rng.shuffle(categories)
    selected_units: list[str] = []
    while len(selected_units) < required_units:
        progressed = False
        for category in categories:
            if buckets[category]:
                selected_units.append(buckets[category].pop())
                progressed = True
                if len(selected_units) == required_units:
                    break
        if not progressed:
            break
    if len(selected_units) != required_units:
        raise ValueError(
            "Not enough complete BBQ template quartets for the requested partition"
        )
    selected_df = df[df["template_id"].astype(str).isin(selected_units)]
    selected_ids = selected_df[MANIFEST_ID].astype(str).tolist()
    if len(selected_ids) != size:
        raise RuntimeError(f"Expected {size} BBQ rows, selected {len(selected_ids)}")
    remainder = df[~df[MANIFEST_ID].astype(str).isin(selected_ids)].copy()
    return SampledPartition(ids=selected_ids, remainder=remainder)


def sample_bios_partition(
    df: pd.DataFrame, *, size: int, seed: int
) -> SampledPartition:
    ids = _round_robin_rows(
        df, strata_columns=("profession_id", "gender"), size=size, seed=seed
    )
    remainder = df[~df[MANIFEST_ID].astype(str).isin(ids)].copy()
    return SampledPartition(ids=ids, remainder=remainder)


def _civil_group_memberships(
    row: pd.Series, identity_columns: Sequence[str]
) -> list[tuple[str, str]]:
    label = str(row["target"])
    return [
        (identity, label)
        for identity in identity_columns
        if int(row.get(identity, 0) or 0) == 1
    ]


def sample_civil_partition(
    df: pd.DataFrame,
    *,
    size: int,
    seed: int,
    identity_columns: Sequence[str],
    target_per_group: int | None = None,
) -> SampledPartition:
    """Greedily cover overlapping identity × label groups, then fill randomly."""
    _require_columns(df, [MANIFEST_ID, "target", *identity_columns])
    if size > len(df):
        raise ValueError(
            f"Requested {size} Civil Comments rows but only {len(df)} exist"
        )
    rng = np.random.default_rng(seed)
    all_groups = [
        (identity, label)
        for identity in identity_columns
        for label in ("non_toxic", "toxic")
    ]
    quota = target_per_group or max(2, size // len(all_groups))
    remaining_quota = {group: quota for group in all_groups}
    candidates = df.copy()
    candidates["_memberships"] = candidates.apply(
        lambda row: _civil_group_memberships(row, identity_columns), axis=1
    )
    remaining_indices = list(candidates.index)
    rng.shuffle(remaining_indices)
    selected_indices: list[int] = []
    while len(selected_indices) < size and any(
        (value > 0 for value in remaining_quota.values())
    ):
        best_score = 0
        best_indices: list[int] = []
        for idx in remaining_indices:
            memberships = candidates.at[idx, "_memberships"]
            score = sum(
                (1 for group in memberships if remaining_quota.get(group, 0) > 0)
            )
            if score > best_score:
                best_score = score
                best_indices = [idx]
            elif score == best_score and score > 0:
                best_indices.append(idx)
        if best_score == 0:
            break
        chosen = int(rng.choice(best_indices))
        selected_indices.append(chosen)
        remaining_indices.remove(chosen)
        for group in candidates.at[chosen, "_memberships"]:
            if remaining_quota.get(group, 0) > 0:
                remaining_quota[group] -= 1
    if len(selected_indices) < size:
        fill = np.asarray(remaining_indices, dtype=int)
        rng.shuffle(fill)
        selected_indices.extend(fill[: size - len(selected_indices)].tolist())
    selected = candidates.loc[selected_indices, MANIFEST_ID].astype(str).tolist()
    remainder = df[~df[MANIFEST_ID].astype(str).isin(selected)].copy()
    return SampledPartition(ids=selected, remainder=remainder)


def sample_partition(
    df: pd.DataFrame,
    *,
    dataset_alias: str,
    size: int,
    seed: int,
    identity_columns: Sequence[str] = (),
) -> SampledPartition:
    if dataset_alias == "winobias":
        from tri_fair.data.winobias import sample_units
        ids = sample_units(df, size=size, seed=seed)
        return SampledPartition(ids=ids, remainder=df[~df[MANIFEST_ID].isin(ids)].copy())
    if dataset_alias == "bbq":
        return sample_bbq_partition(df, size=size, seed=seed)
    if dataset_alias == "bias_in_bios":
        return sample_bios_partition(df, size=size, seed=seed)
    if dataset_alias == "civil_comments":
        return sample_civil_partition(
            df, size=size, seed=seed, identity_columns=identity_columns
        )
    ids = _round_robin_rows(df, strata_columns=(), size=size, seed=seed)
    return SampledPartition(
        ids=ids, remainder=df[~df[MANIFEST_ID].astype(str).isin(ids)].copy()
    )


def _bbq_blocks(df: pd.DataFrame, *, block_size: int, seed: int) -> list[list[str]]:
    if block_size != 44:
        raise ValueError(
            "The supported BBQ group-complete block is 44 rows: 11 categories × 4 variants"
        )
    _require_columns(df, [MANIFEST_ID, "category", "template_id"])
    rng = np.random.default_rng(seed)
    category_units: dict[str, list[str]] = defaultdict(list)
    units = df.groupby("template_id", sort=True).agg(
        category=("category", "first"), n=(MANIFEST_ID, "size")
    )
    for template_id, row in units.iterrows():
        if int(row["n"]) != 4:
            raise ValueError(
                f"BBQ template {template_id!r} does not contain four examples"
            )
        category_units[str(row["category"])].append(str(template_id))
    for values in category_units.values():
        rng.shuffle(values)
    categories = sorted(category_units)
    if len(categories) != 11:
        raise ValueError(f"Expected 11 BBQ categories, found {len(categories)}")
    n_blocks = len(df) // block_size
    blocks: list[list[str]] = []
    for _ in range(n_blocks):
        selected_units = []
        for category in categories:
            if not category_units[category]:
                raise ValueError(
                    f"Category {category!r} lacks enough templates for balanced blocks"
                )
            selected_units.append(category_units[category].pop())
        block = (
            df[df["template_id"].astype(str).isin(selected_units)][MANIFEST_ID]
            .astype(str)
            .tolist()
        )
        if len(block) != block_size:
            raise RuntimeError(
                f"Constructed BBQ block of size {len(block)}, expected {block_size}"
            )
        rng.shuffle(block)
        blocks.append(block)
    return blocks


def _bios_blocks(df: pd.DataFrame, *, block_size: int, seed: int) -> list[list[str]]:
    _require_columns(df, [MANIFEST_ID, "profession_id", "gender"])
    cells = sorted(df.groupby(["profession_id", "gender"]).groups)
    if block_size % len(cells) != 0:
        raise ValueError(
            f"Bias-in-Bios block_size={block_size} must be divisible by the {len(cells)} profession × gender cells"
        )
    per_cell = block_size // len(cells)
    rng = np.random.default_rng(seed)
    buckets: dict[tuple, list[str]] = {}
    for cell, group in df.groupby(["profession_id", "gender"], sort=True):
        ids = group[MANIFEST_ID].astype(str).to_numpy(copy=True)
        rng.shuffle(ids)
        buckets[cell] = ids.tolist()
    n_blocks = len(df) // block_size
    blocks: list[list[str]] = []
    for _ in range(n_blocks):
        block: list[str] = []
        for cell in cells:
            if len(buckets[cell]) < per_cell:
                raise ValueError(
                    f"Cell {cell} lacks support for a complete Bias-in-Bios block"
                )
            block.extend((buckets[cell].pop() for _ in range(per_cell)))
        rng.shuffle(block)
        blocks.append(block)
    return blocks


def _civil_blocks(
    df: pd.DataFrame, *, block_size: int, seed: int, identity_columns: Sequence[str]
) -> list[list[str]]:
    remaining = df.copy()
    blocks: list[list[str]] = []
    n_blocks = len(df) // block_size
    for block_index in range(n_blocks):
        sampled = sample_civil_partition(
            remaining,
            size=block_size,
            seed=seed + 1009 * block_index,
            identity_columns=identity_columns,
            target_per_group=max(2, block_size // (2 * len(identity_columns))),
        )
        blocks.append(sampled.ids)
        remaining = sampled.remainder
    return blocks


def build_fairness_blocks(
    df: pd.DataFrame,
    *,
    dataset_alias: str,
    block_size: int,
    seed: int,
    identity_columns: Sequence[str] = (),
) -> list[list[str]]:
    if len(df) % block_size != 0:
        raise ValueError(
            f"Development size {len(df)} must be divisible by block size {block_size}"
        )
    if dataset_alias == "winobias":
        from tri_fair.data.winobias import build_blocks
        return build_blocks(df, block_size=block_size, seed=seed)
    if dataset_alias == "bbq":
        blocks = _bbq_blocks(df, block_size=block_size, seed=seed)
    elif dataset_alias == "bias_in_bios":
        blocks = _bios_blocks(df, block_size=block_size, seed=seed)
    elif dataset_alias == "civil_comments":
        blocks = _civil_blocks(
            df, block_size=block_size, seed=seed, identity_columns=identity_columns
        )
    else:
        ids = df[MANIFEST_ID].astype(str).to_numpy(copy=True)
        rng = np.random.default_rng(seed)
        rng.shuffle(ids)
        blocks = [
            ids[i : i + block_size].tolist() for i in range(0, len(ids), block_size)
        ]
    flat = [item for block in blocks for item in block]
    if len(flat) != len(set(flat)):
        raise RuntimeError(
            "A development example appears in more than one fairness block"
        )
    if set(flat) != set(df[MANIFEST_ID].astype(str)):
        raise RuntimeError(
            "Fairness blocks do not exactly cover the development partition"
        )
    return blocks
