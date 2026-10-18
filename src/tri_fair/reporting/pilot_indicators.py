"""Transparent front indicators from fully evaluated pilot shortlists.

Search R2 is an in-sample diagnostic. MO-CAPO noisy R2 requires aligned
held-out outcomes and is deliberately unavailable from a search-only pilot.
"""

from __future__ import annotations
import argparse
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tri_fair.reporting.metrics import hypervolume_3d, noisy_r2
from tri_fair.reporting.objectives import pareto_mask


def shortlist_indicators(run):
    candidates = run.get("final_incumbent_candidates", [])
    if run.get("status") != "measured" or not run.get("comparable") or not candidates:
        return dict(
            status="unavailable", reason="no_comparable_full_coverage_shortlist"
        )
    bound = float(
        run["dataset_configuration"]["fairness"]["fairness_kwargs"][
            "normalization_cost_upper_bound"
        ]
    )
    if bound <= 0:
        raise ValueError("Positive fixed cost normalization bound required")
    raw = np.array(
        [[1 - c["quality"], c["cost"] / bound, c["fairness"]] for c in candidates],
        dtype=float,
    )
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite pilot candidate objective")
    front = raw[pareto_mask(raw)]
    return dict(
        status="measured_search_only",
        search_hv_shortlist_3d=hypervolume_3d(front),
        search_r2_shortlist_3d=noisy_r2(front, front, n_preferences=4096, seed=2026),
        noisy_r2_3d=None,
        noisy_r2_reason="requires aligned held-out outcomes for every candidate",
        candidates=len(candidates),
        nondominated_candidates=len(front),
        scope="audited shortlist; may omit other search-front candidates",
        reference_point=[1.1, 1.1, 1.1],
        normalization=[1.0, bound, 1.0],
        preference_rule="4096 Dirichlet(1,1,1) preferences; seed 2026; lower R2 is better",
    )


def export_report(report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in report["runs"]:
        indicator = shortlist_indicators(run)
        rows.append(
            dict(
                method=run["method"],
                accuracy=run.get("accuracy"),
                macro_f1=run.get("quality")
                if run.get("quality_metric") == "macro_f1"
                else None,
                cost=run.get("cost"),
                unfairness=run.get("unfairness"),
                **indicator,
            )
        )
    payload = dict(
        dataset=report["dataset"],
        model=report["model"],
        seed=report.get("seed"),
        split="search",
        benchmark_result=False,
        significance_established=False,
        rows=rows,
    )
    (output / "pilot_front_indicators.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    table = pd.DataFrame(rows).reindex(
        columns=[
            "method",
            "accuracy",
            "macro_f1",
            "cost",
            "unfairness",
            "search_hv_shortlist_3d",
            "search_r2_shortlist_3d",
        ]
    )
    table.to_csv(output / "pilot_comparison.csv", index=False)
    # Minimal LaTeX with escaped method identifiers; no dependency on Jinja.
    headers = [
        "Method",
        "Accuracy",
        "Macro-F1",
        "Cost",
        "Unfairness",
        "Search HV",
        "Search R2",
    ]
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\hline",
        " & ".join(headers) + r" \\",
        r"\hline",
    ]
    for _, row in table.iterrows():
        vals = [str(row["method"]).replace("_", r"\_")] + [
            "--" if pd.isna(v) else f"{v:.4f}" for v in row.iloc[1:]
        ]
        lines.append(" & ".join(vals) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    (output / "pilot_comparison.tex").write_text("\n".join(lines), encoding="utf-8")
    text = (
        "Single-seed search-split pilot. The table reports the selected balanced prompt and indicators of the audited shortlist. "
        "HV uses loss coordinates (1-quality, cost/C, unfairness) with fixed reference (1.1,1.1,1.1). "
        "R2 uses the same search values for selection and scoring and is NOT held-out noisy R2. "
        "Differences are descriptive; no statistical significance is established. A higher HV alone does not imply better values in every objective."
    )
    body = (
        f'<!doctype html><meta charset="utf-8"><title>Tri-Fair pilot comparison</title><style>body{{font:16px system-ui;max-width:1100px;margin:40px auto;color:#17324d}}table{{border-collapse:collapse;width:100%}}td,th{{padding:12px;border-bottom:1px solid #ccd7df;text-align:right}}th{{background:#e9f1f8}}p{{line-height:1.6}}</style><h1>{html.escape(report["model"])} / {html.escape(report["dataset"])}</h1><p>{text}</p>'
        + table.to_html(index=False, float_format=lambda x: f"{x:.4f}")
    )
    (output / "pilot_comparison.html").write_text(body, encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = export_report(
        json.loads(Path(args.report).read_text(encoding="utf-8")), args.output_dir
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
