"""Cached search-block sensitivity and few-shot selection; never read holdouts.

Leave-one-block-out ranges are selection heuristics, not confidence intervals.
The original pooled quality, cost and fairness remain the reported objectives.
"""

from __future__ import annotations

import re
import numpy as np
from tri_fair.search.attribution import contract_instruction


def search_stability(optimizer, prompt, blocks):
    blocks = tuple(sorted(set(map(int, blocks))))
    if not blocks:
        raise ValueError("Search stability requires observed blocks")
    observed = set(optimizer.task.get_evaluated_blocks([prompt])[prompt])
    if not set(blocks) <= observed:
        raise ValueError("Stability diagnostics cannot request new predictions")
    key = (prompt.construct_prompt(), blocks)
    cache = optimizer.stability_cache
    if key in cache:
        return dict(cache[key])
    task = optimizer.task
    frame = task.cached_prediction_frame(prompt, blocks)
    cost_in, cost_out = optimizer.cost_per_input_token, optimizer.cost_per_output_token
    pooled = task.objectives_from_frame(frame, cost_in, cost_out)
    folds = []
    if len(blocks) >= 3:
        for omit in blocks:
            subset = task.cached_prediction_frame(
                prompt, [b for b in blocks if b != omit]
            )
            vector = task.objectives_from_frame(subset, cost_in, cost_out)
            if np.isfinite(vector).all():
                folds.append(vector)
    ranges = (
        np.max(np.abs(np.asarray(folds) - pooled), axis=0) if folds else np.zeros(3)
    )
    result = dict(
        blocks=list(blocks),
        usable_folds=len(folds),
        quality=float(pooled[0]),
        cost=float(-pooled[1]),
        unfairness=float(-pooled[2]),
        quality_sensitivity=float(ranges[0]),
        fairness_sensitivity=float(ranges[2]),
        conservative_quality=float(
            pooled[0] - optimizer.stability_strength * ranges[0]
        ),
        conservative_unfairness=float(
            -pooled[2] + optimizer.stability_strength * ranges[2]
        ),
        interpretation="search-block sensitivity, not a confidence interval",
    )
    if len(cache) >= 256:
        cache.clear()
    cache[key] = result
    return dict(result)


def choose_stable_parent(optimizer, mode, attempt):
    pool = list(optimizer.parent_pool())
    if not pool:
        raise RuntimeError("No evaluated parent available")
    maps = optimizer.task.get_evaluated_blocks(pool)
    common = sorted(set.intersection(*(set(maps[p]) for p in pool)))
    if not common:
        return pool[attempt % len(pool)], {"usable_folds": 0, "blocks": []}
    rows = [(p, search_stability(optimizer, p, common)) for p in pool]
    floor = (
        max(r["conservative_quality"] for _, r in rows)
        - optimizer.parent_quality_margin
    )
    eligible = [(p, r) for p, r in rows if r["conservative_quality"] >= floor]
    bound = float(optimizer.task.fairness_kwargs["normalization_cost_upper_bound"])

    def key(item):
        p, r = item
        q, f, c = r["conservative_quality"], r["conservative_unfairness"], r["cost"]
        value = (
            (-q, f, c)
            if mode == "quality"
            else (f, -q, c)
            if mode == "fairness"
            else (c, -q, f)
            if mode == "cost"
            else ((1 - q) ** 2 + (c / bound) ** 2 + f**2, -q, c)
        )
        return (*value, p.construct_prompt())

    ranked = sorted(rows if mode == "quality" else eligible, key=key)
    if mode == "explore":
        return rows[attempt % len(rows)]
    return ranked[0]


def balanced_gold_demonstrations(optimizer, attribution, *, offset=0):
    """Rank short GOLD examples from the frozen few-shot pool only.

    WinoBias uses a complete pro/anti pair. Other tasks prefer a different
    target/protected group for the second example. Labels are never generated.
    """
    pool = optimizer.df_few_shots
    maximum = min(2, int(optimizer.upper_shots))
    if pool is None or pool.empty or maximum < 1:
        return []
    terms = set(
        re.findall(
            r"[a-z]{3,}",
            " ".join(
                e.get("input", "") for e in attribution.get("evidence", [])
            ).lower(),
        )
    )
    target = {str(e["gold"]) for e in attribution.get("confusions", [])}
    rows = []
    for position, (_, row) in enumerate(pool.iterrows()):
        text, label = (
            str(row[optimizer.task.x_column]),
            str(row[optimizer.task.y_column]).casefold(),
        )
        if label not in optimizer.task.classes or len(text) > 700:
            continue
        words = set(re.findall(r"[a-z]{3,}", text.lower()))
        relevance = len(words & terms) / max(1, len(words | terms))
        group = tuple(str(row.get(c, "")) for c in optimizer.task.protected_columns)
        rows.append(
            dict(
                text=text,
                label=label,
                group=group,
                pair=str(row.get("pair_id", "")),
                stereotype=str(row.get("stereotype", "")),
                position=position,
                score=(label not in target, -relevance, len(text), position),
            )
        )
    if not rows:
        return []
    rows.sort(key=lambda r: r["score"])
    chosen = [rows[offset % min(4, len(rows))]]
    if maximum == 2:
        if chosen[0]["pair"]:
            partners = [
                r
                for r in rows
                if r["pair"] == chosen[0]["pair"]
                and r["stereotype"] != chosen[0]["stereotype"]
            ]
        else:
            partners = [r for r in rows if r["position"] != chosen[0]["position"]]
            partners.sort(
                key=lambda r: (
                    r["label"] == chosen[0]["label"],
                    r["group"] == chosen[0]["group"],
                    r["score"],
                )
            )
        if partners:
            chosen.append(partners[0])
    return [
        f"Input:\n{r['text']}\nOutput:\n<final_answer>{r['label']}</final_answer>"
        for r in chosen
    ]


def canonical_instruction(text, classes, *, catalog=False):
    # Avoid repeatedly appending the standard output contract. Preserve the
    # substantive parent rule and any existing explicit label schema.
    return contract_instruction(" ".join(str(text).split()), classes, catalog=catalog)


def paired_search_evidence(task, prompt, blocks, *, limit=2):
    """Observed paired disagreements, not synthetic counterfactual predictions."""
    if not str(task.fairness_metric_name).startswith("winobias"):
        return []
    frame = task.cached_prediction_frame(prompt, blocks)
    if "pair_id" not in frame:
        return []
    pairs = []
    for pair, rows in frame.groupby("pair_id", sort=True):
        if len(rows) != 2 or rows._review_target.nunique() != 1:
            continue
        errors = int((rows._review_target != rows._review_prediction).sum())
        disagreement = rows._review_prediction.nunique() > 1
        if not errors:
            continue
        cases = [
            dict(
                input=str(r[task.x_column])[:240],
                gold=str(r._review_target),
                prediction=str(r._review_prediction),
                condition=str(r.stereotype),
            )
            for _, r in rows.iterrows()
        ]
        pairs.append((not disagreement, -errors, str(pair), cases))
    return [dict(pair_id=p[2], observed_cases=p[3]) for p in sorted(pairs)[:limit]]
