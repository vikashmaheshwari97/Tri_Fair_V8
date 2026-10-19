"""Search-only adaptive operator credit with matched cached observations."""

from __future__ import annotations

import math
import numpy as np
from tri_fair.reporting.metrics import hypervolume_3d

ARMS = ("attributed_reflection", "stable_components", "complementary_merge")


def select_arm(trials, rewards, *, exploration, interval, arms=ARMS):
    """Deterministic UCB with cold-start coverage and periodic exploration."""
    for arm in arms:
        if trials[arm] == 0:
            return arm
    total = sum(trials.values())
    if total % interval == 0:
        return min(arms, key=lambda arm: (trials[arm], arms.index(arm)))
    return max(
        arms,
        key=lambda arm: (
            rewards[arm] / trials[arm]
            + exploration * math.sqrt(math.log(total + 1) / trials[arm]),
            -arms.index(arm),
        ),
    )


def cached_front_gain(optimizer, before, after):
    """Re-score both fronts on identical already observed search blocks.

    No full-coverage vs partial-coverage HV comparison and no new inference.
    The statistic is a search heuristic, never a held-out performance claim.
    """
    pool = list(dict.fromkeys([*before, *after]))
    if not before or not after:
        return dict(eligible=False, reason="empty_front", blocks=[])
    mapping = optimizer.task.get_evaluated_blocks(pool)
    blocks = sorted(set.intersection(*(set(mapping[p]) for p in pool)))
    if len(blocks) < optimizer.min_racing_blocks:
        return dict(eligible=False, reason="insufficient_shared_blocks", blocks=blocks)
    vectors = {}
    bound = float(optimizer.task.fairness_kwargs["normalization_cost_upper_bound"])
    for p in pool:
        frame = optimizer.task.cached_prediction_frame(p, blocks)
        vector = optimizer.task.objectives_from_frame(
            frame, optimizer.cost_per_input_token, optimizer.cost_per_output_token
        )
        if not np.isfinite(vector).all():
            return dict(
                eligible=False, reason="shared_fairness_not_ready", blocks=blocks
            )
        vectors[p] = np.asarray([1 - vector[0], -vector[1] / bound, -vector[2]])
    old = hypervolume_3d(np.asarray([vectors[p] for p in before]))
    new = hypervolume_3d(np.asarray([vectors[p] for p in after]))
    return dict(
        eligible=True,
        reason=None,
        blocks=blocks,
        hv_before=old,
        hv_after=new,
        hv_gain=new - old,
    )


def complementary_parents(optimizer):
    pool = list(optimizer.parent_pool())
    if len(pool) < 2:
        return None
    mapping = optimizer.task.get_evaluated_blocks(pool)
    blocks = sorted(set.intersection(*(set(mapping[p]) for p in pool)))
    if not blocks:
        return None
    evidence = {}
    for p in pool:
        frame = optimizer.task.cached_prediction_frame(p, blocks)
        correct = set(
            frame.loc[
                frame._review_target == frame._review_prediction, "_review_id"
            ].astype(str)
        )
        evidence[p] = correct
    mother = max(
        pool,
        key=lambda p: (
            len(evidence[p]),
            -len(p.construct_prompt()),
            p.construct_prompt(),
        ),
    )
    alternatives = [p for p in pool if p != mother]
    father = max(
        alternatives,
        key=lambda p: (
            len(evidence[p] - evidence[mother]),
            len(evidence[mother] ^ evidence[p]),
            -len(p.construct_prompt()),
            p.construct_prompt(),
        ),
    )
    return (
        mother,
        father,
        dict(
            blocks=blocks,
            mother_correct=len(evidence[mother]),
            father_correct=len(evidence[father]),
            father_repairs=len(evidence[father] - evidence[mother]),
            mother_repairs=len(evidence[mother] - evidence[father]),
        ),
    )
