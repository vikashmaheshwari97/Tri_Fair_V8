"""CPU-only diagnostic of saved search readiness; no model calls or checkpoint loading."""

import argparse
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd


def audit(path):
    path = Path(path)
    rows = []
    if path.suffix.lower() == ".zip":
        archive = zipfile.ZipFile(path)
        sources = [
            (n, lambda n=n: archive.read(n))
            for n in archive.namelist()
            if n.endswith("/step_results.parquet") and "/Tri-Fair-" in n
        ]
    else:
        archive = None
        sources = [
            (str(p), p.read_bytes)
            for p in path.glob("**/Tri-Fair-*/run/step_results.parquet")
        ]
    try:
        for name, read in sources:
            frame = pd.read_parquet(io.BytesIO(read()))
            failed = frame[~frame.fairness_ready.eq(True)]
            reasons = Counter()
            for value in failed.fairness_diagnostics_json:
                reasons.update(json.loads(value).get("readiness_reasons", []))
            levels = (
                frame.assign(
                    block_count=frame.evaluated_on_blocks.map(
                        lambda v: len(json.loads(v))
                    )
                )
                .groupby("block_count")
                .fairness_ready.agg(["count", "sum"])
            )
            rows.append(
                dict(
                    source=name,
                    logged_population_rows=len(frame),
                    ready_rows=int(frame.fairness_ready.eq(True).sum()),
                    support_by_evaluation_level={
                        str(k): {"rows": int(v["count"]), "ready_rows": int(v["sum"])}
                        for k, v in levels.iterrows()
                    },
                    failure_reasons=dict(reasons),
                )
            )
    finally:
        if archive is not None:
            archive.close()
    if not rows:
        raise ValueError("No Tri-Fair search populations found")
    return dict(
        gpu_used=False,
        metrics_changed=False,
        records=rows,
        interpretation="Population rows repeat prompts across steps; counts are diagnostics, not independent experimental replications.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    args = p.parse_args()
    print(json.dumps(audit(args.input), indent=2))
