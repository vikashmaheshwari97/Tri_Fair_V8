"""Pre-create and validate immutable fairness split manifests.

Running this once before submitting a large SLURM array downloads each dataset,
creates seed-specific dev/few-shot/test partitions, and validates group-complete
blocks.  ``experiment.py`` also takes a per-manifest lock, so this step is a
recommended preflight rather than a correctness requirement.
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
from tri_fair.config.datasets import ALL_DATASETS
from tri_fair.config.setup import SETUP
from tri_fair.data.loading import create_dev_tasks, create_test_task
from tri_fair.cli.common import (
    atomic_write_json,
    configure_logging,
    directory_lock,
    manifest_lock_path,
    parse_positive_int_csv,
    resolve_manifest_path,
    sha256_file,
    utc_now_iso,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_DATASETS = "bbq,bias_in_bios,civil_comments"
DEFAULT_SEEDS = "42,43,44"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create and validate fixed Tri-Fair split manifests."
    )
    parser.add_argument("--study-profile", choices=["v8"], default="v8")
    parser.add_argument("--datasets", default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--manifest-dir", default="data/splits_v8_clean")
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--lock-timeout", type=float, default=3600.0)
    parser.add_argument("--report", default="data/splits_v8_clean/manifest_report.json")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _parse_datasets(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one dataset is required")
    unknown = sorted(set(values) - set(ALL_DATASETS))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    non_fair = [
        value for value in values if ALL_DATASETS[value].task_type != "Fairness"
    ]
    if non_fair:
        raise ValueError(
            f"Manifest preparation is only needed for fairness datasets: {non_fair}"
        )
    return values


def run(args: argparse.Namespace) -> Path:
    datasets = _parse_datasets(args.datasets)
    seeds = parse_positive_int_csv(args.seeds)
    entries: list[dict] = []
    for dataset in datasets:
        config = ALL_DATASETS[dataset]
        if args.study_profile == "v8":
            from tri_fair.config.study import apply_study_profile

            config = apply_study_profile(config)
        fairness = config.fairness
        assert fairness is not None
        for seed in seeds:
            LOGGER.info("Preparing manifest for %s seed=%d", dataset, seed)
            lock = manifest_lock_path(args.manifest_dir, dataset, seed)
            with directory_lock(lock, timeout_seconds=args.lock_timeout):
                dev_task, fewshots = create_dev_tasks(
                    dataset_config=config,
                    eval_strategy="full",
                    n_subsamples=0,
                    dev_size=SETUP.dev_size,
                    fs_size=SETUP.fs_size,
                    seed=seed,
                    manifest_dir=args.manifest_dir,
                    regenerate_manifest=args.regenerate,
                )
                test_task = create_test_task(
                    dataset_config=config,
                    eval_strategy="full",
                    n_subsamples=0,
                    test_size=SETUP.test_size,
                    seed=seed,
                    manifest_dir=args.manifest_dir,
                    regenerate_manifest=False,
                )
            manifest_path = resolve_manifest_path(args.manifest_dir, dataset, seed)
            dev_size = len(getattr(dev_task, "df", []))
            test_size = len(getattr(test_task, "df", []))
            block_sizes = [len(block) for block in getattr(dev_task, "blocks", [])]
            if dev_size != fairness.dev_size:
                raise ValueError(
                    f"{dataset}/seed{seed}: dev size {dev_size} != configured {fairness.dev_size}"
                )
            if len(fewshots) != fairness.fs_size:
                raise ValueError(
                    f"{dataset}/seed{seed}: few-shot size {len(fewshots)} != configured {fairness.fs_size}"
                )
            if fairness.test_size is not None and test_size != fairness.test_size:
                raise ValueError(
                    f"{dataset}/seed{seed}: test size {test_size} != configured {fairness.test_size}"
                )
            if not block_sizes or any(
                (size != fairness.block_size for size in block_sizes)
            ):
                raise ValueError(
                    f"{dataset}/seed{seed}: invalid block sizes {block_sizes}; expected every block to have {fairness.block_size} rows"
                )
            partition_report = {}
            if args.study_profile == "v8":
                from tri_fair.data.partitions import split_search_validation

                search_task, validation_task = split_search_validation(dev_task)
                partition_report = {
                    "search_size": len(search_task.df),
                    "validation_size": len(validation_task.df),
                    "partition": search_task.study_partition,
                }
            entries.append(
                {
                    **partition_report,
                    "dataset": dataset,
                    "seed": seed,
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": sha256_file(manifest_path),
                    "dev_size": dev_size,
                    "few_shot_size": len(fewshots),
                    "test_size": test_size,
                    "n_blocks": len(block_sizes),
                    "block_sizes": block_sizes,
                    "fairness_metric": fairness.metric_name,
                }
            )
    report_path = Path(args.report)
    atomic_write_json(
        report_path,
        {
            "created_at": utc_now_iso(),
            "manifest_dir": str(Path(args.manifest_dir).resolve()),
            "regenerated": bool(args.regenerate),
            "entries": entries,
        },
    )
    LOGGER.info("Validated %d manifests; report=%s", len(entries), report_path)
    return report_path


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    run(args)


if __name__ == "__main__":
    main()
