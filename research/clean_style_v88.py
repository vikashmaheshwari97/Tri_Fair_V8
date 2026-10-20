"""Measured MO-CAPO visual conventions, with explicit support and readable layouts.

Original rendering code. Palette and viridis convention inspected in
mo374z/mo-capo/analysis/analysis_pipeline.py. Validation is labelled validation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

COST_LABEL = "Avg. Cost [$] per 1M Calls"
TOKEN_LABEL = "Token Budget [×10⁶]"
LINESTYLES = {
    "Tri-Fair-v8.8": "-",
    "NSGAII-PO-Fair": "--",
    "NSGAIII-PO-Fair": "-.",
    "SPEA2-PO-Fair": (0, (5, 1.5)),
    "RandomSearch-PO-Fair": ":",
}


def output_share(frame, split):
    """Actual mean output tokens / (mean input + mean output); no imputation."""
    inp = pd.to_numeric(frame[f"{split}_input_tokens"], errors="raise").to_numpy(float)
    out = pd.to_numeric(frame[f"{split}_output_tokens"], errors="raise").to_numpy(float)
    if (
        not np.isfinite(inp).all()
        or not np.isfinite(out).all()
        or (inp < 0).any()
        or (out < 0).any()
        or ((inp + out) <= 0).any()
    ):
        raise ValueError(
            "Few-shot colour needs valid measured input/output token counts"
        )
    return out / (inp + out)


def seed_curves(group, x, y, *, minimum=False):
    curves = []
    for _, g in group.groupby("seed"):
        s = (
            g.groupby(x, sort=True)[y].min()
            if minimum
            else g.groupby(x, sort=True)[y].max()
        )
        s = s.cummin() if minimum else s.cummax()
        curves.append((s.index.to_numpy(float), s.to_numpy(float)))
    return curves


def step_summary(ax, curves, *, color, marker, trajectory=False, linestyle="-"):
    """Plot observed staircases, mean±SD for histories, median+range for fronts."""
    curves = [(x, y) for x, y in curves if len(x)]
    if not curves:
        return
    grid = np.unique(np.concatenate([x for x, _ in curves]))
    stack = []
    for x, y in curves:
        idx = np.searchsorted(x, grid, side="right") - 1
        values = np.full(len(grid), np.nan)
        valid = idx >= 0
        if trajectory:
            valid &= grid <= x[-1]
        values[valid] = y[idx[valid]]
        stack.append(values)
    stack = np.asarray(stack)
    common = np.isfinite(stack).all(axis=0)
    if not common.any():
        return
    stack, grid = stack[:, common], grid[common]
    center = stack.mean(axis=0) if trajectory else np.median(stack, axis=0)
    ax.step(
        grid,
        center,
        where="post",
        color=color,
        linestyle=linestyle,
        lw=2.0,
        ms=4,
    )
    if len(stack) > 1:
        if trajectory:
            spread = stack.std(axis=0, ddof=1)
            lo, hi = center - spread, center + spread
        else:
            lo, hi = stack.min(axis=0), stack.max(axis=0)
        ax.fill_between(grid, lo, hi, step="post", color=color, alpha=0.08, lw=0)


def place_bubbles(ax, x, y, shots, shares):
    """Move only overlapping annotations in display space; leaders retain true data coordinates."""
    ax.scatter(
        x,
        y,
        c=shares,
        norm=Normalize(0, 1),
        cmap="viridis",
        s=24,
        edgecolors="black",
        lw=0.5,
    )
    ax.figure.canvas.draw()
    accepted = []
    # Radius in points; a larger few-shot count gets a larger text circle.
    size = max(11, 8 + 3 * max(len(str(s)) for s in shots))
    for xx, yy, count, value in zip(x, y, shots, shares):
        origin = ax.transData.transform((xx, yy))
        chosen = (0, 0)
        positions = [(0, 0)] + [
            (r * math.cos(a), r * math.sin(a))
            for r in (18, 30, 44, 60, 78, 98)
            for a in np.linspace(0, 2 * math.pi, 20, endpoint=False)
        ]
        scale = ax.figure.dpi / 72
        for dx, dy in positions:
            point = origin + np.array([dx, dy]) * scale
            bbox = ax.get_window_extent()
            if not (
                bbox.x0 + 14 < point[0] < bbox.x1 - 14
                and bbox.y0 + 14 < point[1] < bbox.y1 - 14
            ):
                continue
            if all(np.linalg.norm(point - p) >= (size + 5) * scale for p in accepted):
                chosen = (dx, dy)
                break
        else:
            raise ValueError(
                "Few-shot labels need more canvas space; refusing overlapping export"
            )
        accepted.append(origin + np.array(chosen) * scale)
        ax.annotate(
            str(count),
            (xx, yy),
            xytext=chosen,
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=size - 1,
            fontweight="bold",
            color="black" if value > 0.72 else "white",
            bbox=dict(
                boxstyle="circle,pad=0.24",
                fc=plt.get_cmap("viridis")(value),
                ec="black",
                lw=1.3,
            ),
            arrowprops=None
            if chosen == (0, 0)
            else dict(arrowstyle="-", color="#777777", lw=0.7),
            zorder=5,
            annotation_clip=False,
        )


def extra_figures(
    final,
    trajectory,
    indicators,
    output,
    *,
    datasets,
    methods,
    style,
    names,
    split,
    model,
    budget,
    formats,
):
    output = Path(output)
    made = []
    handles = [
        Line2D(
            [],
            [],
            color=style[m][1],
            linestyle=LINESTYLES.get(m, "-"),
            label=style[m][0],
            lw=2,
        )
        for m in methods
    ]

    def save(fig, name):
        for ext in formats:
            fig.savefig(output / f"{name}.{ext}", dpi=250, bbox_inches="tight")
        plt.close(fig)
        made.append(name)

    for ds in datasets:
        f = final[final.dataset == ds]
        fig, axes = plt.subplots(2, 2, figsize=(12, 10.4))
        quality = (
            "Validation Macro-F1 (%)"
            if ds == "bias_in_bios"
            else "Validation Accuracy (%)"
        )
        for ax in axes.ravel():
            ax.grid(True, lw=0.6, alpha=0.25)
            ax.tick_params(labelsize=11)
        for m in methods:
            g = f[f.optimizer == m].copy()
            _, color, marker = style[m]
            g["quality_percent"] = 100 * g[f"{split}_quality"]
            step_summary(
                axes[0, 0],
                seed_curves(g, f"{split}_cost", "quality_percent"),
                color=color,
                marker=marker,
                linestyle=LINESTYLES.get(m, "-"),
            )
            step_summary(
                axes[0, 1],
                seed_curves(g, f"{split}_fairness", "quality_percent"),
                color=color,
                marker=marker,
                linestyle=LINESTYLES.get(m, "-"),
            )
            if not trajectory.empty:
                hist = trajectory[
                    (trajectory.dataset == ds) & (trajectory.optimizer == m)
                ].copy()
                hist["million_tokens"] = hist.actual_budget_tokens / 1e6
                for ax, metric in (
                    (axes[1, 0], "noisy_r2_3d"),
                    (axes[1, 1], "hv_test_pessimistic_3d"),
                ):
                    curves = []
                    for _, h in hist.groupby("seed"):
                        h = h.sort_values("million_tokens").drop_duplicates(
                            "million_tokens", keep="last"
                        )
                        curves.append(
                            (h.million_tokens.to_numpy(), h[metric].to_numpy())
                        )
                    step_summary(
                        ax,
                        curves,
                        color=color,
                        marker=marker,
                        trajectory=True,
                        linestyle=LINESTYLES.get(m, "-"),
                    )
        axes[0, 0].set(xlabel=COST_LABEL, ylabel=quality)
        axes[0, 1].set(xlabel="Unfairness ceiling ↓", ylabel=quality)
        axes[1, 0].set(xlabel=TOKEN_LABEL, ylabel="nR2 ↓", xlim=(0, budget / 1e6))
        axes[1, 1].set(xlabel=TOKEN_LABEL, ylabel="HVpes ↑", xlim=(0, budget / 1e6))
        for ax in axes.ravel():
            ax.set_title(f"{names[ds]} — {model}", fontsize=14)
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.055),
            ncol=3,
            frameon=False,
            fontsize=11,
        )
        fig.text(
            0.5,
            0.018,
            "Held-out validation. Top: median attainment and seed range. Bottom: historical mean ± sample SD on common support.\nFinal audited fronts are reported separately in final_front_indicators. Two-objective projections omit the third objective.",
            ha="center",
            fontsize=9,
        )
        fig.subplots_adjust(
            left=0.095, right=0.98, bottom=0.20, top=0.95, hspace=0.34, wspace=0.28
        )
        save(fig, f"{ds}_reference_panels")

    # Final audited fronts stay separate from historical trajectories.
    # Labelled horizontal intervals replace scattered method-specific glyphs.
    fig, axes = plt.subplots(
        len(datasets), 3, figsize=(14.2, 3.05 * len(datasets)), squeeze=False
    )
    for row, ds in enumerate(datasets):
        for col, (metric, label) in enumerate(
            (
                ("hv_test_optimistic_3d", "HVopt ↑"),
                ("hv_test_pessimistic_3d", "HVpes ↑"),
                ("noisy_r2_3d", "nR2 ↓"),
            )
        ):
            ax = axes[row, col]
            for i, m in enumerate(methods):
                values = indicators[
                    (indicators.dataset == ds) & (indicators.optimizer == m)
                ][metric].to_numpy(float)
                sd = values.std(ddof=1) if len(values) > 1 else None
                ax.errorbar(
                    values.mean(),
                    i,
                    xerr=sd,
                    color=style[m][1],
                    fmt="o",
                    capsize=3,
                    ms=5,
                    elinewidth=1.4,
                )
            ax.set(
                yticks=range(len(methods)),
                yticklabels=[style[m][0] for m in methods] if col == 0 else [],
                xlabel=label,
                title=names[ds],
            )
            ax.set_ylim(len(methods) - 0.5, -0.5)
            ax.tick_params(axis="y", length=0, labelsize=10)
            ax.grid(axis="x", alpha=0.20)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.margins(x=0.12)
    fig.suptitle(
        f"{model} | Held-out validation | mean ± sample SD across seeds", fontsize=14
    )
    fig.text(
        0.5,
        0.012,
        "Final audited fronts | 3 seeds per method | intervals = sample SD, not confidence intervals",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.965), h_pad=2.0, w_pad=2.4)
    save(fig, "final_front_indicators")

    composition = []
    for m in methods:
        subset = final[final.optimizer == m].copy()
        subset["output_token_share"] = output_share(subset, split)
        subset["few_shot_count"] = subset.few_shots_json.map(
            lambda v: len(json.loads(v or "[]"))
        )
        composition.append(subset)
        cols = min(2, len(datasets))
        rows = math.ceil(len(datasets) / cols)
        max_count = max((len(subset[subset.dataset == d]) for d in datasets), default=1)
        panel_size = max(5.3, 4.4 + 0.10 * max_count)
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(panel_size * cols + 1.1, panel_size * rows),
            squeeze=False,
        )
        for i, ax in enumerate(axes.ravel()):
            if i >= len(datasets):
                ax.set_visible(False)
                continue
            ds = datasets[i]
            g = subset[subset.dataset == ds]
            xx = g[f"{split}_cost"].to_numpy(float)
            yy = g[f"{split}_quality"].to_numpy(float)
            xspan = max(np.ptp(xx), 0.1 * np.mean(xx), 1)
            yspan = max(np.ptp(yy), 0.02)
            ax.set_xlim(xx.min() - 0.20 * xspan, xx.max() + 0.20 * xspan)
            ax.set_ylim(
                max(0, yy.min() - 0.18 * yspan), min(1, yy.max() + 0.18 * yspan)
            )
            ax.set(
                xlabel=COST_LABEL,
                ylabel="Validation Macro-F1"
                if ds == "bias_in_bios"
                else "Validation Accuracy",
                title=names[ds],
            )
            ax.grid(ls="--", alpha=0.25)
        fig.subplots_adjust(
            left=0.12 if cols == 1 else 0.085,
            right=0.84 if cols == 1 else 0.85,
            top=0.81 if rows == 1 else 0.87,
            bottom=0.23 if rows == 1 else 0.16,
            hspace=0.36,
            wspace=0.32,
        )
        fig.suptitle(
            f"{style[m][0]} — {model}\nFew-shot count and output-token share",
            fontsize=13,
            y=0.99,
        )
        for i, ds in enumerate(datasets):
            g = subset[subset.dataset == ds]
            place_bubbles(
                axes.ravel()[i],
                g[f"{split}_cost"].to_numpy(float),
                g[f"{split}_quality"].to_numpy(float),
                g.few_shot_count.to_numpy(),
                g.output_token_share.to_numpy(),
            )
        cax = fig.add_axes(
            [0.91, 0.30 if rows == 1 else 0.23, 0.018, 0.42 if rows == 1 else 0.52]
        )
        bar = fig.colorbar(
            plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap="viridis"), cax=cax
        )
        bar.set_label("Output Token Share", fontsize=12)
        footer = (
            "Colour = output / (input + output) tokens.\nNumbers = few-shot examples. Leaders retain actual points.\nCost = configured token-price estimate; not GPU charges."
            if cols == 1
            else "Colour = output / (input + output) tokens. Numbers = few-shot examples.\nLeaders retain actual points. Cost = configured token-price estimate; not GPU charges."
        )
        fig.text(0.46, 0.025, footer, ha="center", fontsize=9)
        save(fig, f"fewshot_{m}")
    pd.concat(composition, ignore_index=True).drop(
        columns=[c for c in final.columns if "diagnostics_json" in c]
    ).to_csv(output / "fewshot_measured_points.csv", index=False)
    return made
