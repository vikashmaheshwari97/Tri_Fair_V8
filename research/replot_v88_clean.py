"""CPU-only clean plots from the existing measured CSV exports; never edit experiment outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from clean_style_v88 import extra_figures, step_summary, LINESTYLES, TOKEN_LABEL

DATASETS = {
    "bbq": "BBQ",
    "civil_comments": "Civil Comments",
    "bias_in_bios": "Bias-in-Bios",
    "winobias": "WinoBias",
}
STYLE = {
    "Tri-Fair-v8.8": ("Tri-Fair", "#000000", "o"),
    "NSGAII-PO-Fair": ("NSGA-II-PO-Fair", "#E69F00", "s"),
    "NSGAIII-PO-Fair": ("NSGA-III-PO-Fair", "#785EF0", "^"),
    "SPEA2-PO-Fair": ("SPEA2-PO-Fair", "#009E73", "D"),
    "RandomSearch-PO-Fair": ("Random Search", "#CC79A7", "v"),
}
CSV_NAMES = (
    "measured_final_indicators.csv",
    "measured_historical_trajectories.csv",
    "fewshot_measured_points.csv",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_exports(source, model, split, budget):
    source = Path(source).resolve()
    indicators, history, final = [pd.read_csv(source / n) for n in CSV_NAMES]
    for f in (indicators, history, final):
        if set(f.model) != {model} or not f.actual_budget_tokens.le(budget).all():
            raise ValueError("Unexpected model or over-budget rows in figure input")
    if set(indicators.evaluation_split) != {split} or set(final.evaluation_split) != {
        split
    }:
        raise ValueError("CSV exports are not the requested held-out split")
    if indicators[["dataset", "optimizer", "seed"]].duplicated().any():
        raise ValueError("Duplicate final indicator runs")
    if final[["dataset", "optimizer", "seed", "prompt_id"]].duplicated().any():
        raise ValueError("Duplicate final candidates")
    keys = set(
        indicators[["dataset", "optimizer", "seed"]].itertuples(index=False, name=None)
    )
    if keys != set(
        final[["dataset", "optimizer", "seed"]].itertuples(index=False, name=None)
    ):
        raise ValueError("Final prompt and indicator identities differ")
    if not final[f"{split}_fairness_ready"].eq(True).all():
        raise ValueError("Final prompts have incomplete fairness support")
    if set(final.optimizer) - set(STYLE):
        raise ValueError("Unknown method needs an explicit style")
    if (
        not history.empty
        and history[
            ["dataset", "optimizer", "seed", "chosen_step", "actual_budget_tokens"]
        ]
        .duplicated()
        .any()
    ):
        raise ValueError("Duplicate historical indicator rows")
    return indicators, history, final


def render(
    source,
    output,
    *,
    model="qwen-3-30b",
    split="validation",
    budget=2_000_000,
    formats=("png", "pdf", "svg"),
):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents:
        raise ValueError(
            "Use a separate sibling output directory to preserve original exports"
        )
    fingerprints = {n: sha(source / n) for n in CSV_NAMES}
    indicators, history, final = load_exports(source, model, split, budget)
    datasets = [d for d in DATASETS if d in set(final.dataset)]
    methods = [m for m in STYLE if m in set(final.optimizer)]
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.labelweight": "bold",
            "axes.linewidth": 1.1,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    made = []
    handles = [
        Line2D(
            [], [], color=STYLE[m][1], linestyle=LINESTYLES[m], lw=2, label=STYLE[m][0]
        )
        for m in methods
    ]
    for metric, label, name in (
        ("noisy_r2_3d", "nR2 indicator ↓", "trajectory_nR2"),
        ("hv_test_pessimistic_3d", "Pessimistic hypervolume ↑", "trajectory_HVpes"),
    ):
        fig, axes = plt.subplots(
            1, len(datasets), figsize=(4.15 * len(datasets), 5.8), squeeze=False
        )
        for i, (ax, ds) in enumerate(zip(axes[0], datasets)):
            for m in methods:
                group = history[(history.dataset == ds) & (history.optimizer == m)]
                curves = []
                for _, g in group.groupby("seed"):
                    g = g.sort_values("actual_budget_tokens").drop_duplicates(
                        "actual_budget_tokens", keep="last"
                    )
                    curves.append(
                        (g.actual_budget_tokens.to_numpy() / 1e6, g[metric].to_numpy())
                    )
                expected = indicators[
                    (indicators.dataset == ds) & (indicators.optimizer == m)
                ].seed.nunique()
                if len(curves) != expected:
                    raise ValueError(f"Missing historical seed support for {ds}/{m}")
                step_summary(
                    ax,
                    curves,
                    color=STYLE[m][1],
                    marker=None,
                    trajectory=True,
                    linestyle=LINESTYLES[m],
                )
            ax.set(xlabel=TOKEN_LABEL, ylabel=label, xlim=(0, budget / 1e6))
            ax.set_box_aspect(1)
            ax.grid(True, ls="--", lw=0.6, alpha=0.25)
            ax.text(
                0.5,
                -0.24,
                f"({chr(97 + i)}) {DATASETS[ds]}",
                ha="center",
                va="top",
                transform=ax.transAxes,
                fontsize=13,
            )
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=3,
            frameon=False,
            fontsize=12,
        )
        fig.subplots_adjust(left=0.055, right=0.99, bottom=0.31, top=0.83, wspace=0.35)
        fig.text(
            0.5,
            0.025,
            f"{model} | Held-out {split} | Historical mean ± sample SD on common observed support.\nFinal audited fronts are reported separately; no smoothing or extrapolation.",
            ha="center",
            fontsize=9,
            color="#46505A",
        )
        for ext in formats:
            fig.savefig(output / f"{name}.{ext}", dpi=250, bbox_inches="tight")
        plt.close(fig)
        made.append(name)
    made += extra_figures(
        final,
        history,
        indicators,
        output,
        datasets=datasets,
        methods=methods,
        style=STYLE,
        names=DATASETS,
        split=split,
        model=model,
        budget=budget,
        formats=formats,
    )
    assert fingerprints == {n: sha(source / n) for n in CSV_NAMES}, (
        "Source exports were modified"
    )
    # These are exact input copies for inspection, not recomputed or edited scores.
    for n in CSV_NAMES:
        (output / n).write_bytes((source / n).read_bytes())
    protocol = dict(
        plot_only=True,
        model=model,
        split=split,
        budget=budget,
        source_csv_directory=str(source),
        source_sha256=fingerprints,
        figures=made,
        metric_recalculation=False,
        source_exports_modified=False,
        changes=[
            "No final-audit scatter markers on historical trajectories",
            "Method-specific line styles and lighter unchanged SD bands",
            "Final audited means and sample SD in labelled horizontal intervals",
        ],
        interpretation="Historical incumbents and final audited fronts use distinct selection policies; do not connect them or force monotonic held-out curves",
    )
    (output / "clean_plot_protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n"
    )
    return protocol


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-root", required=True)
    p.add_argument("--output-dir")
    p.add_argument("--model", default="qwen-3-30b")
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--budget", type=int, default=2_000_000)
    p.add_argument("--formats", default="png,pdf,svg")
    a = p.parse_args()
    formats = tuple(a.formats.split(","))
    if not set(formats) <= {"png", "pdf", "svg"}:
        p.error("Unknown export format")
    base = Path(a.batch_root) / f"aggregate_{a.split}" / a.model
    result = render(
        base / "paper_figures",
        Path(a.output_dir) if a.output_dir else base / "paper_figures_clean",
        model=a.model,
        split=a.split,
        budget=a.budget,
        formats=formats,
    )
    print(
        f"Saved {len(result['figures'])} clean figure sets. Input CSV checksums unchanged. No model calls."
    )


if __name__ == "__main__":
    main()
