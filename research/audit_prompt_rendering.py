"""Read-only audit of stored vs rendered prompts in an extracted results batch.

Usage: python research/audit_prompt_rendering.py --batch-root PATH --output audit.json
No checkpoint unpickling, model loading, dataset download or GPU work.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pandas as pd


def audit(root):
    root = Path(root)
    runs = []
    for path in sorted(root.rglob("run_summary.json")):
        summary = json.loads(path.read_text())
        diagnostics = summary.get("search_diagnostics") or {}
        trace = diagnostics.get("recent_proposals", [])
        counts = Counter((x["mode"], x["status"]) for x in trace)
        runs.append(
            dict(
                dataset=summary["dataset"],
                method=summary["optimizer"],
                seed=summary.get("random_seed", summary.get("seed")),
                source=str(path.relative_to(root)),
                proposal_round=diagnostics.get("proposal_round"),
                retained_trace_length=len(trace),
                mode_status_counts={
                    f"{m}/{s}": n for (m, s), n in sorted(counts.items())
                },
                downstream_tokens=summary["final_downstream_tokens"]["total_tokens"],
                meta_tokens=summary["final_meta_tokens"]["total_tokens"],
                budget_overshoot_tokens=summary.get("budget_overshoot_tokens"),
            )
        )
    renderings = []
    for path in sorted(root.rglob("eval_checkpoints_union.parquet")):
        frame = pd.read_parquet(path)
        required = {
            "optimizer",
            "dataset",
            "prompt_id",
            "prompt",
            "few_shots_json",
            "downstream_template",
        }
        if not required <= set(frame):
            raise ValueError(f"Missing prompt metadata in {path}")
        for (dataset, method), group in frame.groupby(["dataset", "optimizer"]):
            unique = group.drop_duplicates("prompt_id")
            invisible = []
            with_demos = 0
            for row in unique.itertuples():
                demos = json.loads(row.few_shots_json)
                with_demos += bool(demos)
                if demos and any(d.strip() not in row.prompt for d in demos):
                    invisible.append(row.prompt_id)
            renderings.append(
                dict(
                    dataset=dataset,
                    method=method,
                    unique_logged_prompts=len(unique),
                    with_stored_demonstrations=with_demos,
                    prompts_with_invisible_demonstrations=len(invisible),
                    affected_prompt_ids=invisible,
                    source=str(path.relative_to(root)),
                    source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            )
    return dict(
        scope="saved run diagnostics and logged held-out candidates; not every generated candidate",
        benchmark_results_modified=False,
        runs=runs,
        renderings=renderings,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit(args.batch_root), indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
