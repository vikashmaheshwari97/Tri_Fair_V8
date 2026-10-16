"""Generate or execute a model/dataset/method/seed matrix without shell expansion."""

from __future__ import annotations
import argparse
import itertools
import json
import shlex
import subprocess
import sys
from pathlib import Path
from tri_fair.config.registry import METHODS


def commands(args):
    models = args.models.split(",")
    datasets = args.datasets.split(",")
    methods = args.methods.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    if (
        not set(methods) <= set(METHODS)
        or not set(models) <= {"qwen-3-30b", "mistral-3-24b", "gpt-oss-120b"}
        or (not set(datasets) <= {"bbq", "bias_in_bios", "civil_comments"})
    ):
        raise ValueError("unknown model, dataset or method")
    for model, dataset, method, seed in itertools.product(
        models, datasets, methods, seeds
    ):
        out = Path(args.output_dir) / model / dataset / method / f"seed{seed}"
        yield [
            sys.executable,
            "-m",
            "tri_fair.cli.train",
            "--experiment-name",
            "tri_fair_v8",
            "--model",
            model,
            "--dataset",
            dataset,
            "--optimizer",
            method,
            "--random-seed",
            str(seed),
            "--budget-per-run",
            str(args.budget),
            "--budget-checkpoints",
            str(args.budget),
            "--manifest-dir",
            args.manifest_dir,
            "--n-init-prompts",
            str(args.n_init_prompts),
            "--output-dir",
            str(out),
        ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="qwen-3-30b,mistral-3-24b,gpt-oss-120b")
    parser.add_argument("--datasets", default="bbq,civil_comments,bias_in_bios")
    parser.add_argument(
        "--methods",
        default="Tri-Fair-v8,NSGAII-PO-Fair,SPEA2-PO-Fair,NSGAIII-PO-Fair,RandomSearch-PO-Fair,NSGAII-PO-Fair-Enhanced",
    )
    parser.add_argument("--seeds", default="72,73,74")
    parser.add_argument("--budget", type=int, default=5000000)
    parser.add_argument("--n-init-prompts", type=int, default=12)
    parser.add_argument("--manifest-dir", default="data/splits_v8_clean")
    parser.add_argument("--output-dir", default="results/tri_fair_v8_clean")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--write-plan")
    args = parser.parse_args()
    if args.budget <= 0 or args.n_init_prompts < 2:
        parser.error("positive budget and at least two initial prompts are required")
    plan = list(commands(args))
    if args.task_index is not None and (not 0 <= args.task_index < len(plan)):
        parser.error(f"task-index must be in 0..{len(plan) - 1}")
    if args.write_plan:
        Path(args.write_plan).parent.mkdir(parents=True, exist_ok=True)
        Path(args.write_plan).write_text(json.dumps(plan, indent=2), encoding="utf-8")
    selected = (
        list(enumerate(plan))
        if args.task_index is None
        else [(args.task_index, plan[args.task_index])]
    )
    for index, command in selected:
        print(f"{index}: {shlex.join(command)}", flush=True)
        if args.execute:
            subprocess.run(command, cwd=Path(__file__).resolve().parents[3], check=True)


if __name__ == "__main__":
    main()
