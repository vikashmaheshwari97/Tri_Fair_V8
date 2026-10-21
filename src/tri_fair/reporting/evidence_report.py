"""Additional views of measured fronts; primary objectives are never replaced."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def visible_demonstration_count(row):
    from promptolution.utils.prompt import Prompt

    demos = json.loads(row.few_shots_json)
    if not demos:
        return 0
    template = row.get("downstream_template")
    if not isinstance(template, str):
        template = None
    # Use the exact stored template, including historical missing insertions.
    prompt = Prompt(
        str(row.instruction), demos, downstream_template=template
    ).construct_prompt()
    return sum(bool(d.strip()) and d.strip() in prompt for d in demos)


def representative_table(indicators):
    rows = []
    for _, r in indicators.iterrows():
        for choice in ("quality_first", "cost_first", "fairness_first", "balanced"):
            rows.append(
                dict(
                    dataset=r.dataset,
                    model=r.model,
                    optimizer=r.optimizer,
                    seed=int(r.seed),
                    selection=choice,
                    quality=100 * r[f"{choice}_test_quality"],
                    cost=r[f"{choice}_test_cost"],
                    unfairness=r[f"{choice}_test_fairness"],
                )
            )
    return pd.DataFrame(rows)


def fairness_table(final, split):
    rows = []
    for _, r in final.iterrows():
        key = f"{split}_fairness_diagnostics_json"
        value = r.get(key, "{}")
        diag = json.loads(value) if isinstance(value, str) else (value or {})
        raw = next(
            (
                diag[k]
                for k in (
                    "equalized_odds_rms",
                    "rms_tpr_gap",
                    "raw_gap_rms",
                    "rms_abs_bias",
                )
                if k in diag
            ),
            None,
        )
        rows.append(
            dict(
                dataset=r.dataset,
                optimizer=r.optimizer,
                seed=int(r.seed),
                prompt_id=r.prompt_id,
                metric=diag.get("metric"),
                reported_unfairness=r[f"{split}_fairness"],
                raw_component=raw,
                utility_penalty=diag.get(
                    "utility_penalty", diag.get("accuracy_floor_penalty")
                ),
                fairness_ready=bool(r[f"{split}_fairness_ready"]),
                interpretation="raw component is diagnostic; reported objective remains unchanged",
            )
        )
    return pd.DataFrame(rows)


def supplementary_figures(final, indicators, out, *, split, style):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    out = Path(out)
    representative_table(indicators).to_csv(
        out / "representative_tradeoffs.csv", index=False
    )
    fairness_table(final, split).to_csv(out / "fairness_components.csv", index=False)
    datasets = [
        d
        for d in ("bbq", "civil_comments", "bias_in_bios", "winobias")
        if d in set(final.dataset)
    ]
    methods = [m for m in style if m in set(final.optimizer)]
    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(4.1 * len(datasets), 3.8),
        squeeze=False,
        layout="constrained",
    )
    for ax, ds in zip(axes.flat, datasets):
        sub = final[final.dataset == ds]
        for method in methods:
            g = sub[sub.optimizer == method]
            ax.scatter(
                g[f"{split}_cost"],
                100 * g[f"{split}_quality"],
                color=style[method][1],
                s=34,
                alpha=0.8,
                edgecolors="white",
                linewidths=0.4,
            )
        ax.set_title(ds.replace("_", " ").title())
        ax.set_xlabel("Avg. Cost [$] per 1M Calls")
        ax.grid(alpha=0.15)
    axes[0, 0].set_ylabel(
        "Validation quality (%)" if split == "validation" else "Test quality (%)"
    )
    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                color=style[m][1],
                marker="o",
                linestyle="none",
                label=style[m][0],
            )
            for m in methods
        ],
        loc="outside upper center",
        ncol=min(5, len(methods)),
        frameon=False,
    )
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"quality_cost_fronts.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Few-shot colors exactly match the MO-CAPO analysis palette. Unlike
    # connecting fronts across seeds, these are individual observed prompts.
    colors = {
        0: "#ffffcc",
        1: "#c7e9b4",
        2: "#7fcdbb",
        3: "#41b6c4",
        4: "#2c7fb8",
        5: "#253494",
    }
    fig, axes = plt.subplots(
        len(methods),
        len(datasets),
        figsize=(3.6 * len(datasets), 2.5 * len(methods) + 0.6),
        squeeze=False,
        sharex="col",
        sharey="col",
        layout="constrained",
    )
    for row, method in enumerate(methods):
        for col, ds in enumerate(datasets):
            ax = axes[row, col]
            g = final[(final.dataset == ds) & (final.optimizer == method)]
            for shots, gg in g.groupby("few_shot_count"):
                ax.scatter(
                    gg[f"{split}_cost"],
                    100 * gg[f"{split}_quality"],
                    color=colors[min(5, int(shots))],
                    edgecolors="#333333",
                    s=45,
                    linewidths=0.6,
                    alpha=0.9,
                )
            if col == 0:
                ax.set_ylabel(style[method][0] + "\nQuality (%)")
            if row == 0:
                ax.set_title(ds.replace("_", " ").title())
            if row == len(methods) - 1:
                ax.set_xlabel("Avg. Cost [$] per 1M Calls")
            ax.grid(alpha=0.15)
    counts = sorted(set(final.few_shot_count.astype(int)))
    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                color=colors[min(5, n)],
                markeredgecolor="#333333",
                label=f"{n} shots",
            )
            for n in counts
        ],
        loc="outside upper center",
        ncol=max(1, len(counts)),
        frameon=False,
    )
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"fewshot_mocapo_palette.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return ["quality_cost_fronts", "fewshot_mocapo_palette"]
