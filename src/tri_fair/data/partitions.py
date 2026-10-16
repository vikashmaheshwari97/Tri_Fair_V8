"""Outcome-independent search/validation split with sufficient fairness support."""

from __future__ import annotations
import numpy as np
from tri_fair.data.task import FairnessTask
from tri_fair.fairness.statistics import wilson_width


def _cell_counts(task, frame):
    """Counts only: neither model predictions nor test examples enter this rule."""
    metric = task.fairness_metric_name
    if metric.startswith("bios"):
        return frame.groupby([task.y_column, "gender"], dropna=False).size().tolist()
    if metric.startswith("bbq"):
        return (
            frame.groupby(["category", "context_condition"], dropna=False)
            .size()
            .tolist()
        )
    if metric.startswith("civil"):
        counts = []
        for identity in task.protected_columns:
            for present in [0, 1]:
                for label in task.classes:
                    counts.append(
                        int(
                            (
                                (frame[identity].fillna(0).astype(int) == present)
                                & (frame[task.y_column].astype(str) == str(label))
                            ).sum()
                        )
                    )
        return counts
    raise ValueError(f"v8 partition does not define support cells for {metric}")


def _supports_all_outcomes(task, blocks):
    frame = task.df.iloc[task._positions_for_blocks(blocks)]
    counts = _cell_counts(task, frame)
    width = task.fairness_kwargs.get(
        "max_accuracy_ci_width", task.fairness_kwargs.get("max_rate_ci_width", 1.0)
    )
    confidence = task.fairness_kwargs.get("confidence", 0.95)
    return bool(
        counts
        and min(counts) >= task.min_group_count
        and all(
            (wilson_width(n // 2, n, confidence=confidence) <= width for n in counts)
        )
    )


def split_search_validation(task):
    """Reserve enough complete blocks for the worst-case Wilson width gate.

    The partition uses only manifest IDs, labels and protected metadata. Choose
    the smallest validation prefix that leaves sufficient support in search too.
    Keep a fresh v8 manifest/results namespace; older splits are not comparable.
    """
    rng = np.random.default_rng(task.seed + 81073)
    order = rng.permutation(task.n_blocks).tolist()
    for reserve in range(2, task.n_blocks - 1):
        valid_blocks = sorted(order[:reserve])
        search_blocks = sorted(order[reserve:])
        if _supports_all_outcomes(task, valid_blocks) and _supports_all_outcomes(
            task, search_blocks
        ):
            break
    else:
        raise ValueError(
            "insufficient group support for separate v8 search and validation; use the enlarged v8 profile and prepare fresh manifests"
        )

    def make(blocks, role):
        positions = task._positions_for_blocks(blocks)
        frame = task.df.iloc[positions].copy().reset_index(drop=True)
        ids = [[task.ids[i] for i in task.blocks[b]] for b in blocks]
        result = FairnessTask(
            df=frame,
            task_description=task.task_description,
            classes=task.classes,
            quality_metric=task.quality_metric_name,
            fairness_metric=task.fairness_metric,
            fairness_metric_name=task.fairness_metric_name,
            protected_columns=task.protected_columns,
            metadata_columns=task.metadata_columns,
            min_group_count=task.min_group_count,
            fairness_kwargs=task.fairness_kwargs,
            block_ids=ids,
            eval_strategy=task.eval_strategy if role == "search" else "full",
            seed=task.seed,
            id_column=task.id_column,
            x_column=task.x_column,
            y_column=task.y_column,
        )
        result.manifest_path = getattr(task, "manifest_path", "")
        result.study_partition = {
            "version": 2,
            "role": role,
            "rule": "outcome-independent-worst-case-Wilson",
            "search_original_blocks": search_blocks,
            "validation_original_blocks": valid_blocks,
            "search_ids": [
                task.ids[i] for i in task._positions_for_blocks(search_blocks)
            ],
            "validation_ids": [
                task.ids[i] for i in task._positions_for_blocks(valid_blocks)
            ],
        }
        return result

    return (make(search_blocks, "search"), make(valid_blocks, "validation"))
