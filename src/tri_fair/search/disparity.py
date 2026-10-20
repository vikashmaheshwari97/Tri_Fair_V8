"""Metric-aligned SEARCH evidence; no predictions or holdout access here."""

from __future__ import annotations

import numpy as np

from tri_fair.search.refinement import head_tail


def disparity_evidence(task, prompt, *, offset=0, limit=4):
    """Show an actual disparity's error and same-label controls on both sides.

    Groups with uniformly poor predictions need not have a disparity. Rank
    components of the configured metric instead of ranking group error rates.
    Examples illustrate a comparison; they are not causal counterfactual pairs.
    """
    frame = task.cached_prediction_frame(prompt, range(task.n_blocks)).copy()
    diag = task.get_last_record(prompt).get("diagnostics", {})
    truth = frame._review_target.astype(str)
    prediction = frame._review_prediction.astype(str)
    correct = truth == prediction
    components = []
    family = task.fairness_metric_name
    # The published loss may be a utility safeguard rather than the raw gap.
    # Target the binding term; reducing an already smaller gap cannot reduce it.
    penalty = float(diag.get("utility_penalty", diag.get("accuracy_floor_penalty", 0)))
    raw_gap = float(
        diag.get(
            "equalized_odds_rms", diag.get("rms_tpr_gap", diag.get("raw_gap_rms", 1))
        )
    )
    recalls = diag.get(
        "class_recall", diag.get("profession_recall", diag.get("recall_by_label", {}))
    )
    if penalty > raw_gap and recalls:
        labels = sorted(recalls, key=lambda label: (recalls[label], label))
        labels = [label for label in labels if ((truth == label) & ~correct).any()][:3]
        if labels:
            label = labels[offset % len(labels)]
            cases = []
            for status in (False, True):
                rows = frame[(truth == label) & (correct == status)]
                if not rows.empty:
                    row = rows.iloc[offset % len(rows)]
                    cases.append(
                        dict(
                            search_id=str(row._review_id),
                            input=head_tail(str(row[task.x_column]), 1000),
                            gold=str(row._review_target),
                            prediction=str(row._review_prediction),
                            correct=status,
                            protected_context={
                                c: str(row[c]) for c in task.protected_columns
                            },
                        )
                    )
            return dict(
                metric=family,
                components=[],
                selected_component=dict(
                    metric="binding_utility_safeguard",
                    label=label,
                    measured_recall=float(recalls[label]),
                    utility_penalty=penalty,
                    raw_gap=raw_gap,
                ),
                cases=cases,
                comparison="repair class recall before optimizing a nonbinding disparity term",
            )
    if "bios" in family:
        for label, gap in diag.get("tpr_gap_by_profession", {}).items():
            if not np.isfinite(gap):
                continue
            mask = truth == label
            # TPR male-female: repair the lower-TPR group without damaging
            # successes of the same profession in either group.
            lower = 1 if gap > 0 else 0
            components.append(
                (
                    abs(gap),
                    dict(
                        metric="profession_conditional_tpr_gap",
                        label=label,
                        attribute="gender",
                        signed_gap=float(gap),
                        repair_group=lower,
                        improve="recall",
                    ),
                    mask,
                    frame.gender.astype(int) == lower,
                )
            )
    elif "civil" in family:
        for identity, rates in diag.get("rates_by_identity", {}).items():
            for kind, label in (("tpr", "toxic"), ("fpr", "non_toxic")):
                gap = float(rates[kind + "_gap"])
                if not np.isfinite(gap):
                    continue
                # TPR: improve low side. FPR: correct false positives on high side.
                side = int(gap < 0) if kind == "tpr" else int(gap > 0)
                components.append(
                    (
                        abs(gap),
                        dict(
                            metric=kind + "_present_absent_gap",
                            label=label,
                            attribute=identity,
                            signed_gap=gap,
                            repair_group=side,
                            improve="recall" if kind == "tpr" else "specificity",
                        ),
                        truth == label,
                        frame[identity].fillna(0).astype(int) == side,
                    )
                )
    elif "wino" in family:
        for kind, gap in diag.get("gap_by_type", {}).items():
            rates = diag.get("accuracy_by_cell", {})
            low = min(("pro", "anti"), key=lambda s: rates.get(kind + "/" + s, 1))
            components.append(
                (
                    float(gap),
                    dict(
                        metric="paired_accuracy_gap",
                        task_type=kind,
                        attribute="stereotype",
                        repair_group=low,
                        improve="coreference_accuracy",
                        absolute_gap=float(gap),
                    ),
                    frame.task_type == kind,
                    frame.stereotype == low,
                )
            )
    else:
        # BBQ's objective includes directional bias and coverage, not a binary
        # identity TPR gap. Preserve measured category diagnostics verbatim.
        categories = []
        for name, values in diag.get("by_category", {}).items():
            for condition in ("ambig", "disambig"):
                value = values.get(condition)
                if value is not None and np.isfinite(value):
                    categories.append(
                        dict(
                            category=name,
                            condition=condition,
                            bias=float(value),
                            accuracy=values.get(condition + "_accuracy"),
                            support=values.get(condition + "_support"),
                        )
                    )
        categories.sort(key=lambda c: (-abs(c["bias"]), c["category"], c["condition"]))
        selected = categories[offset % min(3, len(categories))] if categories else None
        cases = []
        if selected:
            mask = (frame.category == selected["category"]) & (
                frame.context_condition == selected["condition"]
            )
            for status in (False, True):
                rows = frame[mask & (correct == status)]
                if not rows.empty:
                    row = rows.iloc[offset % len(rows)]
                    cases.append(
                        dict(
                            search_id=str(row._review_id),
                            input=head_tail(str(row[task.x_column]), 1000),
                            gold=str(row._review_target),
                            prediction=str(row._review_prediction),
                            correct=status,
                        )
                    )
        return dict(
            metric=family,
            components=categories[:3],
            selected_component=selected,
            coverage_shortfall=diag.get("disambig_coverage_shortfall"),
            cases=cases,
            comparison="directional bias and answer coverage; not identity TPR",
        )
    components.sort(key=lambda x: (-x[0], str(x[1])))
    usable = [c for c in components if c[0] > 0 and ((~correct) & c[2] & c[3]).any()]
    if not usable:
        return dict(metric=family, components=[], cases=[])
    # Rotate across the three largest observed disparities, not tiny gaps.
    _, detail, relevant, repair_side = usable[offset % min(3, len(usable))]
    chosen = []
    masks = [
        relevant & repair_side & ~correct,
        relevant & repair_side & correct,
        relevant & ~repair_side & correct,
        relevant & ~repair_side & ~correct,
    ]
    for mask in masks:
        rows = frame.loc[mask]
        if rows.empty:
            continue
        row = rows.iloc[offset % len(rows)]
        source = str(row[task.x_column])
        chosen.append(
            dict(
                search_id=str(row._review_id),
                input=head_tail(source, 1000),
                input_truncated=len(source) > 1000,
                gold=str(row._review_target),
                prediction=str(row._review_prediction),
                correct=str(row._review_target) == str(row._review_prediction),
                protected_context={c: str(row[c]) for c in task.protected_columns},
            )
        )
    return dict(
        metric=family,
        components=[c[1] for c in usable[:3]],
        selected_component=detail,
        cases=chosen[:limit],
        comparison="observed same-label cases, not causal matched pairs",
    )


def confusion_demonstrations(optimizer, evidence, *, offset=0):
    """Explore the entire frozen few-shot pool, including confusion alternatives.

    No label is generated. WinoBias keeps its indivisible pro/anti pair.
    """
    from tri_fair.search.stability import balanced_gold_demonstrations

    task, pool = optimizer.task, optimizer.df_few_shots
    maximum = int(optimizer.upper_shots)
    if pool is None or pool.empty or maximum == 0:
        return []
    if "wino" in task.fairness_metric_name:
        return balanced_gold_demonstrations(optimizer, evidence, offset=offset)
    targets = []
    aligned = evidence.get("metric_aligned", {})
    component = aligned.get("selected_component", {})
    if component.get("label"):
        targets.append(component["label"])
    for row in evidence.get("confusions", []):
        targets.extend([row["gold"], row["predicted"]])
    targets = list(dict.fromkeys(t for t in targets if t in task.classes))
    if targets:
        shift = (offset // 2) % len(targets)
        targets = targets[shift:] + targets[:shift]
    candidates = []
    for position, (_, row) in enumerate(pool.iterrows()):
        text, label = str(row[task.x_column]), str(row[task.y_column]).casefold()
        if label not in task.classes or len(text) > 1800:
            continue
        candidates.append(
            dict(
                position=position,
                text=text,
                label=label,
                group=tuple(str(row.get(c, "")) for c in task.protected_columns),
            )
        )
    chosen = []
    for slot in range(min(maximum, len(candidates))):
        available = [r for r in candidates if r not in chosen]
        target = targets[slot % len(targets)] if targets else None
        ranked = sorted(
            available,
            key=lambda r: (
                r["label"] != target
                if target
                else any(r["label"] == c["label"] for c in chosen),
                any(r["group"] == c["group"] for c in chosen),
                len(r["text"]),
                r["position"],
            ),
        )
        # Rotate inside the preferred label/group stratum, across ALL rows.
        best = ranked[0]
        stratum = [
            r
            for r in ranked
            if r["label"] == best["label"] and r["group"] == best["group"]
        ]
        chosen.append(stratum[offset % len(stratum)])
    return [
        f"Input:\n{r['text']}\nOutput:\n<final_answer>{r['label']}</final_answer>"
        for r in chosen
    ]


def coverage_select(losses, capacity):
    """Pareto first: preserve balanced and three objective anchors, then marginal HV.

    This is Tri-Fair survival, not a change to comparator survival or reporting.
    Losses retain their measured values; objective bounds only set their units.
    """
    from pymoo.indicators.hv import HV

    losses = np.asarray(losses, dtype=float)
    if (
        losses.ndim != 2
        or losses.shape[1] != 3
        or not np.isfinite(losses).all()
        or capacity < 1
    ):
        raise ValueError("finite n-by-3 losses and positive capacity required")
    remaining, chosen = list(range(len(losses))), []
    metric = HV(ref_point=np.ones(3))

    def hv(indices):
        values = losses[indices]
        values = values[(values <= 1).all(axis=1)]
        return float(metric(values)) if len(values) else 0.0

    while remaining and len(chosen) < capacity:
        front = [
            i
            for i in remaining
            if not any(
                np.all(losses[j] <= losses[i]) and np.any(losses[j] < losses[i])
                for j in remaining
            )
        ]
        if len(front) <= capacity - len(chosen):
            chosen.extend(front)
        else:
            anchors = [min(front, key=lambda i: (float(losses[i] @ losses[i]), i))]
            anchors += [
                min(front, key=lambda i: (losses[i, axis], float(losses[i].sum()), i))
                for axis in range(3)
            ]
            for i in anchors:
                if i not in chosen and len(chosen) < capacity:
                    chosen.append(i)
            while len(chosen) < capacity:
                candidates = [i for i in front if i not in chosen]
                chosen.append(
                    max(candidates, key=lambda i: (hv(chosen + [i]) - hv(chosen), -i))
                )
        remaining = [i for i in remaining if i not in front]
    return chosen
