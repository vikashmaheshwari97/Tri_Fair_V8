"""Plot measured audited candidates; no smoothing or synthetic points."""

import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout="constrained")
    colors = ["#1764a3", "#c46024", "#3d875b", "#8b5aa9"]
    for index, run in enumerate(report["runs"]):
        points = run.get("final_incumbent_candidates", [])
        if not points:
            continue
        q = [100 * x["quality"] for x in points]
        c = [x["cost"] for x in points]
        f = [x["fairness"] for x in points]
        for ax, x, y in [(axes[0], c, q), (axes[1], f, q), (axes[2], c, f)]:
            ax.scatter(
                x,
                y,
                s=68,
                c=colors[index % len(colors)],
                marker=["o", "s", "^", "D"][index % 4],
                label=run["method"],
                edgecolors="white",
                linewidths=0.7,
                zorder=3,
            )
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.18, zorder=0)
        ax.margins(0.18)
    axes[0].set(
        xlabel="Inference cost (lower is better)",
        ylabel="Macro-F1 (%)" if report["dataset"] == "bias_in_bios" else "Quality (%)",
        title="Quality–cost trade-off",
    )
    axes[1].set(
        xlabel="Unfairness (lower is better)",
        ylabel=axes[0].get_ylabel(),
        title="Quality–fairness trade-off",
    )
    axes[2].set(
        xlabel="Inference cost (lower is better)",
        ylabel="Unfairness (lower is better)",
        title="Cost–fairness trade-off",
    )
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(
        f"{report['model']} / {report['dataset']} / seed {report['seed']}: audited search shortlists",
        fontsize=12,
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "pilot_objective_tradeoffs.pdf", bbox_inches="tight")
    fig.savefig(out / "pilot_objective_tradeoffs.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
