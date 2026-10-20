"""Probe first; rerun one matched Bios comparison only after a verified pass.

No optimizer monkeypatching, label voting, cached-answer substitution, automatic
fallback, result deletion, or merging with the old inference profile.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from types import SimpleNamespace

METHODS = (
    "Tri-Fair-v8.8",
    "NSGAII-PO-Fair",
    "NSGAIII-PO-Fair",
    "SPEA2-PO-Fair",
    "RandomSearch-PO-Fair",
)
PACKAGES = (
    "vllm",
    "torch",
    "promptolution",
    "transformers",
    "datasets",
    "numpy",
    "pandas",
    "pymoo",
    "huggingface-hub",
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hash(repo):
    digest = hashlib.sha256()
    for path in sorted(
        [*(repo / "src").rglob("*.py"), *(repo / "scripts").rglob("*.py")]
    ):
        digest.update(str(path.relative_to(repo)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def packages():
    result = {}
    for name in PACKAGES:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def strict_labels(outputs, classes):
    """Unlike upstream MarkerBasedPredictor, never substitute classes[0]."""
    import re

    result = []
    for output in outputs:
        values = re.findall(r"<final_answer>([^<>]*)</final_answer>", str(output))
        if len(values) != 1 or values[0].strip().casefold() not in classes:
            raise ValueError(f"Invalid raw classification output: {str(output)!r}")
        result.append(values[0].strip().casefold())
    return result


def validate_proof(proof_path, repo):
    proof = read(proof_path)
    if (
        proof.get("status") != "passed"
        or proof.get("model") != "qwen-3-30b"
        or proof.get("dataset") != "bias_in_bios"
        or proof.get("seed") != 72
        or proof.get("batch_invariant") is not True
        or proof.get("scenarios") != 8
        or proof.get("prediction_count") != 192
        or proof.get("raw_output_count") != 192
    ):
        raise ValueError("Proof is not a complete passing eight-scenario Bios probe")
    if proof.get("source_sha256") != source_hash(repo):
        raise ValueError("Source changed since the probe; rerun the small probe")
    if proof.get("launcher_sha256") != file_hash(__file__):
        raise ValueError("Stability launcher changed since the probe")
    if proof.get("packages") != packages():
        raise ValueError("Package versions changed since the probe")
    if proof.get("manifest_sha256") != file_hash(proof["manifest_path"]):
        raise ValueError("Frozen split manifest changed since the probe")
    if proof.get("model_config_sha256") != file_hash(proof["model_config_path"]):
        raise ValueError("Local model configuration changed since the probe")
    if proof.get("original_probe_sha256") != file_hash(proof["original_probe_path"]):
        raise ValueError("Original failure evidence changed since the probe")
    details = proof.get("checks", [])
    if len(details) != 8 or any(
        x.get("status") != "passed"
        or x.get("prediction_count") != 24
        or x.get("mismatches")
        or x.get("invalid_label_examples")
        for x in details
    ):
        raise ValueError("Probe details are incomplete or contain disagreements")
    return proof


def probe(args, repo):
    # This environment is established before importing any GPU-dependent code.
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    from promptolution.predictors import MarkerBasedPredictor
    from promptolution.utils.prompt import Prompt
    from tri_fair.cli.common import set_generation_limit
    from tri_fair.config.datasets import ALL_DATASETS
    from tri_fair.config.models import ALL_MODELS
    from tri_fair.config.study import apply_study_profile
    from tri_fair.data.loading import create_dev_tasks
    from tri_fair.data.partitions import split_search_validation
    from tri_fair.optimizers.guarded import GuardedTriFairOptimizer
    from tri_fair.runtime.decoding import (
        configure_downstream_greedy,
        configure_structured_labels,
    )
    from tri_fair.runtime.budget import BudgetAwarePredictor, TokenBudgetController
    from tri_fair.runtime.models import create_llm, _resolve_qwen_snapshot
    from tri_fair.runtime.utils import seed_everything
    from tri_fair.search.rendering import output_contract, rendered_prompt
    from tri_fair.search.stability import balanced_gold_demonstrations

    class RecordingPredictor(MarkerBasedPredictor):
        def _extract_preds(self, outputs):
            raw_outputs.extend(str(x) for x in outputs)
            return strict_labels(outputs, self.classes)

    out = Path(args.output_root).resolve()
    out.mkdir(parents=True, exist_ok=False)
    versions = packages()
    proof = dict(
        status="starting",
        model="qwen-3-30b",
        dataset="bias_in_bios",
        seed=72,
        batch_invariant=True,
        source_sha256=source_hash(repo),
        launcher_sha256=file_hash(__file__),
        packages=versions,
        hostname=platform.node(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        scopes="search only; diagnostic tokens separate from optimization cap",
        diagnostic_downstream_cap=100000,
        guarantee="finite same-process probe, not universal or cross-hardware determinism",
        checks=[],
        scenarios=0,
        prediction_count=0,
    )
    raw_outputs = []
    llm = None
    try:
        if versions["vllm"] != "0.24.0" or versions["promptolution"] != "2.2.3":
            raise RuntimeError(
                "This repair is checked against vllm 0.24.0 and promptolution 2.2.3"
            )
        dataset = apply_study_profile(ALL_DATASETS["bias_in_bios"])
        manifest_dir = Path(args.manifest_dir).expanduser().resolve()
        # Never silently create new splits during a recovery diagnostic.
        if not manifest_dir.is_dir() or not any(manifest_dir.rglob("*.json")):
            raise FileNotFoundError("Existing frozen split manifests are required")
        manifests_before = {
            str(p.resolve()): file_hash(p) for p in manifest_dir.rglob("*.json")
        }
        seed_everything(72)
        dev, fewshots = create_dev_tasks(
            dataset, eval_strategy="full", seed=72, manifest_dir=manifest_dir
        )
        task, _ = split_search_validation(dev)
        manifest = Path(task.manifest_path).resolve()
        if manifests_before.get(str(manifest)) != file_hash(manifest):
            raise RuntimeError("Recovery must reuse an existing unchanged manifest")
        holder = SimpleNamespace(task=task, df_few_shots=fewshots, upper_shots=2)
        instructions = random.Random(72).sample(list(dataset.initial_prompts), 4)
        prompts = [
            rendered_prompt(
                output_contract(text),
                balanced_gold_demonstrations(holder, {}, offset=i // 2)
                if i % 2
                else [],
                template=Prompt(text).downstream_template,
            )
            for i, text in enumerate(instructions)
        ]
        first_hash = hashlib.sha256(prompts[0].construct_prompt().encode()).hexdigest()
        old = read(args.original_probe)
        if first_hash != old.get("instruction_sha256"):
            raise ValueError(
                "Reconstructed prompt differs from the failed original probe"
            )
        expected_ids = [str(x["example_id"]) for x in old.get("predictions", [])]
        import numpy as np

        actual_ids = [
            str(task.ids[i]) for i in np.linspace(0, len(task.xs) - 1, 8, dtype=int)
        ]
        if actual_ids != expected_ids:
            raise ValueError("Frozen search examples differ from the failed probe")
        config = ALL_MODELS["qwen-3-30b"]
        snapshot = _resolve_qwen_snapshot(
            config,
            model_name=config.model.removeprefix("vllm-"),
            revision=config.llm_kwargs["revision"],
        )
        proof.update(
            manifest_path=str(manifest),
            manifest_sha256=file_hash(manifest),
            model_config_path=str(snapshot / "config.json"),
            model_config_sha256=file_hash(snapshot / "config.json"),
            original_probe_path=str(Path(args.original_probe).resolve()),
            original_probe_sha256=file_hash(args.original_probe),
            first_instruction_sha256=first_hash,
            example_ids=actual_ids,
        )
        llm = create_llm(config, seed=72)
        configure_downstream_greedy(llm, seed=72)
        set_generation_limit(llm, 16)
        configure_structured_labels(llm, task.classes)
        predictor = BudgetAwarePredictor(
            RecordingPredictor(llm, task.classes),
            TokenBudgetController(
                llm=llm,
                max_tokens=100000,
                max_output_tokens=16,
                event_log_path=out / "probe_budget.jsonl",
            ),
        )
        for index, prompt in enumerate(prompts):
            for repeat in range(2):
                checker = SimpleNamespace(
                    repeat_check_examples=8,
                    task=task,
                    prompts=[prompt],
                    predictor=predictor,
                    artifact_directory=out / f"prompt{index}_repeat{repeat}",
                )
                # Run the SAME mandatory gate used by the optimizer. No suppression.
                GuardedTriFairOptimizer._repeatability_check(checker)
                record = checker.repeatability_result
                if (
                    record.get("examples") != 8
                    or len(record.get("predictions", [])) != 8
                ):
                    raise RuntimeError("Probe returned a truncated prediction batch")
                proof["checks"].append(record)
                proof["scenarios"] += 1
                proof["prediction_count"] += record["prediction_count"]
        proof["status"] = "passed"
    except Exception as error:
        proof.update(status="failed", error_type=type(error).__name__, error=str(error))
    finally:
        proof["actual_downstream_tokens"] = llm.get_token_count() if llm else None
        proof["raw_output_count"] = len(raw_outputs)
        write(out / "raw_outputs.json", raw_outputs)
        write(out / "stability_proof.json", proof)
    print(json.dumps(proof, indent=2), flush=True)
    return 0 if proof["status"] == "passed" else 1


def train(args, repo):
    proof = validate_proof(args.proof, repo)
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    from tri_fair.cli.pilot_v88 import main as pilot

    root = Path(args.output_root).resolve()
    if root.exists():
        raise FileExistsError(
            "Use a fresh output root; saved results are never overwritten"
        )
    root.mkdir(parents=True)
    write(
        root / "stable_profile.json",
        dict(
            proof_path=str(Path(args.proof).resolve()),
            proof_sha256=file_hash(args.proof),
            batch_invariant=True,
            methods=METHODS,
            reuse_old_baseline_runs=False,
            dataset="bias_in_bios",
            seed=72,
            budget=2000000,
        ),
    )
    rc = pilot(
        [
            "--execute",
            "--model",
            "qwen-3-30b",
            "--dataset",
            "bias_in_bios",
            "--seed",
            "72",
            "--budget",
            "2000000",
            "--batch-invariant",
            "--reclaim-audit-reserve",
            "--methods",
            ",".join(METHODS),
            "--manifest-dir",
            str(Path(proof["manifest_path"]).parent),
            "--output-root",
            str(root),
        ]
    )
    reports = list(root.glob("qwen-3-30b/bias_in_bios/seed72/*/pilot_report.json"))
    if rc or len(reports) != 1 or not read(reports[0]).get("all_controls_comparable"):
        raise RuntimeError("Matched Bios pilot incomplete; validation was not started")
    write(
        root / "ready_for_validation.json",
        dict(report=str(reports[0].resolve()), proof_sha256=file_hash(args.proof)),
    )
    return 0


def evaluate(args, repo):
    validate_proof(args.proof, repo)
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    root = Path(args.output_root).resolve()
    ready = read(root / "ready_for_validation.json")
    if ready["proof_sha256"] != file_hash(args.proof):
        raise ValueError("Training and validation proofs differ")
    proof = read(args.proof)
    out = root / "heldout_validation/qwen-3-30b/bias_in_bios/seed72"
    out.mkdir(parents=True, exist_ok=True)
    target = out / "eval_checkpoints_union.parquet"
    command = [
        sys.executable,
        "-m",
        "tri_fair.cli.evaluate",
        "--study-root",
        str(Path(ready["report"]).parent),
        "--model",
        "qwen-3-30b",
        "--dataset",
        "bias_in_bios",
        "--random-seed",
        "72",
        "--optimizers",
        ",".join(METHODS),
        "--manifest-dir",
        str(Path(proof["manifest_path"]).parent),
        "--maximum-actual-tokens",
        "2000000",
        "--max-output-tokens",
        "16",
        "--structured-labels",
        "--batch-invariant",
        "--audited-final",
        "--include-history",
        "--evaluation-split",
        "validation",
        "--output-file",
        str(target),
    ]
    subprocess.run(command, check=True)
    for module, folder in [
        ("front_report", "indicators"),
        ("research_figures", "figures"),
    ]:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tri_fair.reporting." + module,
                "--evaluations",
                str(target),
                "--split",
                "validation",
                "--budget",
                "2000000",
                "--output-dir",
                str(out / folder),
            ],
            check=True,
        )
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["probe", "validate", "train", "evaluate"])
    parser.add_argument("--original-probe")
    parser.add_argument("--proof")
    parser.add_argument("--output-root")
    parser.add_argument("--manifest-dir", default="data/splits_v8_clean")
    args = parser.parse_args(argv)
    if args.action == "probe" and (not args.original_probe or not args.output_root):
        parser.error("probe requires --original-probe and a fresh --output-root")
    if args.action != "probe" and not args.proof:
        parser.error("a passing --proof is required")
    if args.action in {"train", "evaluate"} and not args.output_root:
        parser.error("--output-root is required")
    repo = Path(__file__).resolve().parents[1]
    if args.action == "validate":
        validate_proof(args.proof, repo)
        print("Stable probe verified; only the five matched Bios runs are eligible.")
        return 0
    return {"probe": probe, "train": train, "evaluate": evaluate}[args.action](
        args, repo
    )


if __name__ == "__main__":
    raise SystemExit(main())
