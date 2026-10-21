"""Final audited fronts only: partial-fidelity history is never shown as full data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

STYLE = {
    "Tri-Fair-v9.4": ("Tri-Fair v9.4", "#000000"),
    "Tri-Fair-v9.3": ("Tri-Fair v9.3", "#56B4E9"),
    "Tri-Fair-v9.4-NoEvidence": ("No evidence edits", "#0072B2"),
    "Tri-Fair-v9.4-NoStability": ("No stability guard", "#D55E00"),
    "Tri-Fair-v9.2": ("Tri-Fair v9.2", "#777777"),
    "NSGAII-PO-Fair-Plain": ("NSGA-II", "#E69F00"),
    "NSGAIII-PO-Fair-Plain": ("NSGA-III", "#785EF0"),
    "SPEA2-PO-Fair-Plain": ("SPEA2", "#009E73"),
    "RandomSearch-PO-Fair-Plain": ("Random Search", "#CC79A7"),
    "Tri-Fair-v9.2-NoWeakerDominance": ("No weaker dominance", "#0072B2"),
    "Tri-Fair-v9.2-FullEval": ("Full-evaluation ablation", "#D55E00"),
    "NSGAII-PO-Fair": ("NSGA-II", "#E69F00"),
    "NSGAIII-PO-Fair": ("NSGA-III", "#785EF0"),
    "SPEA2-PO-Fair": ("SPEA2", "#009E73"),
    "RandomSearch-PO-Fair": ("Random Search", "#CC79A7"),
}
DATASETS = {
    "bbq": "BBQ",
    "civil_comments": "Civil Comments",
    "bias_in_bios": "Bias-in-Bios",
    "winobias": "WinoBias",
}


def render(
    batch_root, output_dir, model="qwen-3-30b", split="validation", budget=2000000
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    from tri_fair.reporting.front_report import analyse
    from tri_fair.reporting.research_figures import final_rows

    styles = dict(STYLE)
    paths = sorted(
        (Path(batch_root) / f"heldout_{split}" / model).glob(
            "*/seed*/eval_checkpoints_union.parquet"
        )
    )
    if not paths:
        raise ValueError("No audited held-out files")
    frame = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    if set(frame.evaluation_split) != {split} or set(frame.model) != {model}:
        raise ValueError("Mixed model/split")
    if not frame.selection_policy.eq("audited_final_shortlist").all():
        raise ValueError("Audited-front plots require final audited membership only")
    if not frame.actual_budget_tokens.le(budget).all():
        raise ValueError("Over-budget rows")
    if "Tri-Fair-v9.4" not in set(frame.optimizer) and "Tri-Fair-v9.3" in set(
        frame.optimizer
    ):
        styles["Tri-Fair-v9.3"] = ("Tri-Fair v9.3", "#000000")
    indicators = analyse(frame, split=split, budget=budget)
    final = final_rows(frame).copy()
    if set(indicators.optimizer) - styles.keys():
        raise ValueError("Unknown method style")
    if not final[f"{split}_fairness_ready"].all():
        raise ValueError("Incomplete fairness support")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    indicators.to_csv(out / "measured_final_indicators.csv", index=False)
    inp, output = final[f"{split}_input_tokens"], final[f"{split}_output_tokens"]
    if (inp + output <= 0).any():
        raise ValueError("Missing measured tokens")
    final["output_token_share"] = output / (inp + output)
    final["stored_few_shot_count"] = final.few_shots_json.map(
        lambda x: len(json.loads(x))
    )
    # Older Tri-Fair initializers could store examples in a template with no
    # insertion point. Figures must show demonstrations actually sent to the LLM.
    from tri_fair.reporting.evidence_report import visible_demonstration_count

    final["few_shot_count"] = final.apply(visible_demonstration_count, axis=1)
    final["hidden_demonstration_count"] = (
        final.stored_few_shot_count - final.few_shot_count
    )
    final.to_csv(out / "fewshot_measured_points.csv", index=False)
    methods = [m for m in styles if m in set(indicators.optimizer)]
    datasets = [d for d in DATASETS if d in set(indicators.dataset)]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    made = []

    def save(fig, name):
        for ext in ("png", "pdf", "svg"):
            fig.savefig(out / f"{name}.{ext}", dpi=220, bbox_inches="tight")
        plt.close(fig)
        made.append(name)

    for specs, name in [
        (
            [
                ("hv_test_optimistic_3d", "HVopt ↑", 1),
                ("hv_test_pessimistic_3d", "HVpes ↑", 1),
                ("noisy_r2_3d", "nR2 ↓", 1),
            ],
            "final_front_indicators",
        ),
        (
            [
                ("balanced_test_quality", "Validation quality ↑ (%)", 100),
                ("balanced_test_cost", "Avg. Cost [$] per 1M Calls ↓", 1),
                ("balanced_test_fairness", "Unfairness ↓", 1),
            ],
            "balanced_objectives",
        ),
    ]:
        fig, axes = plt.subplots(
            3,
            len(datasets),
            figsize=(3.8 * len(datasets), 9),
            squeeze=False,
            layout="constrained",
        )
        for col, ds in enumerate(datasets):
            for row, (metric, title, scale) in enumerate(specs):
                ax = axes[row, col]
                for i, method in enumerate(methods):
                    v = (
                        indicators[
                            (indicators.dataset == ds)
                            & (indicators.optimizer == method)
                        ][metric].dropna()
                        * scale
                    )
                    if v.empty:
                        raise ValueError(f"Missing {ds}/{method}")
                    sd = v.std(ddof=1) if len(v) > 1 else None
                    ax.barh(
                        i, v.mean(), color=styles[method][1], height=0.57, alpha=0.88
                    )
                    if sd is not None:
                        ax.errorbar(
                            v.mean(),
                            i,
                            xerr=sd,
                            color="#333333",
                            capsize=3,
                            fmt="none",
                            lw=1,
                        )
                ax.set_yticks(
                    range(len(methods)),
                    [styles[m][0] for m in methods] if col == 0 else [],
                )
                ax.invert_yaxis()
                ax.set_xlabel(title)
                ax.grid(axis="x", alpha=0.15)
                ax.set_axisbelow(True)
                if row == 0:
                    ax.set_title(DATASETS[ds], weight="bold")
        fig.suptitle(
            "Audited held-out fronts"
            if name.startswith("final")
            else "Search-selected balanced prompts",
            fontsize=15,
        )
        save(fig, name)
    fig, axes = plt.subplots(
        len(methods),
        len(datasets),
        figsize=(3.5 * len(datasets), 2.4 * len(methods) + 0.5),
        squeeze=False,
        layout="constrained",
    )
    upper = max(0.01, float(final.output_token_share.max()))
    norm = Normalize(0, upper)
    for row, method in enumerate(methods):
        for col, ds in enumerate(datasets):
            ax = axes[row, col]
            g = final[(final.optimizer == method) & (final.dataset == ds)]
            for count, gg in g.groupby("few_shot_count"):
                ax.scatter(
                    gg[f"{split}_cost"],
                    100 * gg[f"{split}_quality"],
                    c=gg.output_token_share,
                    cmap="viridis",
                    norm=norm,
                    marker={0: "o", 1: "s", 2: "^"}.get(int(count), "D"),
                    s=42,
                    edgecolors="#444444",
                    linewidths=0.4,
                    alpha=0.8,
                )
            if col == 0:
                ax.set_ylabel(styles[method][0] + "\nQuality ↑ (%)")
            if row == 0:
                ax.set_title(DATASETS[ds], weight="bold")
            if row == len(methods) - 1:
                ax.set_xlabel("Avg. Cost [$] per 1M Calls")
            ax.grid(alpha=0.15)
    sm = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
    fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.65, label="Output token share")
    handles = [
        Line2D([], [], marker=s, color="#555555", linestyle="none", label=f"{n} shots")
        for n, s in [(0, "o"), (1, "s"), (2, "^")]
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=3)
    save(fig, "fewshot_cost_quality")
    from tri_fair.reporting.evidence_report import supplementary_figures

    made.extend(
        supplementary_figures(final, indicators, out, split=split, style=styles)
    )
    sources = [
        "measured_final_indicators.csv",
        "fewshot_measured_points.csv",
        "representative_tradeoffs.csv",
        "fairness_components.csv",
    ]
    (out / "figure_provenance.json").write_text(
        json.dumps(
            dict(
                figures=made,
                source_sha256={
                    n: hashlib.sha256((out / n).read_bytes()).hexdigest()
                    for n in sources
                },
                uncertainty="sample SD across seeds; absent for one seed; not confidence intervals",
                trajectory_figures="not generated: only terminal audited coverage is evaluated",
                metric_values_modified=False,
                split=split,
                model=model,
            ),
            indent=2,
        )
        + "\n"
    )
    return made


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", default="qwen-3-30b")
    p.add_argument("--split", default="validation", choices=["validation", "test"])
    p.add_argument("--budget", type=int, default=2000000)
    print(json.dumps(render(**vars(p.parse_args()))))
