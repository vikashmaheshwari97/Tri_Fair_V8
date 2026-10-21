"""Final audited plots with explicit plain-baseline identities; measurements unchanged."""

import argparse
import json

import plot_v92 as base

base.STYLE = {
    "Tri-Fair-v9.3": ("Tri-Fair v9.3", "#000000"),
    "Tri-Fair-v9.2": ("Tri-Fair v9.2 control", "#0072B2"),
    "Tri-Fair-v9.3-NoReadinessWarmup": ("No readiness warm-up", "#D55E00"),
    "NSGAII-PO-Fair-Plain": ("NSGA-II (plain)", "#E69F00"),
    "NSGAIII-PO-Fair-Plain": ("NSGA-III (plain)", "#785EF0"),
    "SPEA2-PO-Fair-Plain": ("SPEA2 (plain)", "#009E73"),
    "RandomSearch-PO-Fair-Plain": ("Random Search (plain)", "#CC79A7"),
}
if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", default="qwen-3-30b")
    p.add_argument("--split", default="validation", choices=["validation", "test"])
    p.add_argument("--budget", type=int, default=2000000)
    print(json.dumps(base.render(**vars(p.parse_args()))))
