"""Dataset loading, immutable split manifests, and task construction.

Fairness datasets use stable manifest IDs and dataset-specific group-complete
blocks.  The first run creates ``data/splits/<dataset>_seed<seed>.json``; every
later run validates and reuses it, ensuring all optimizers receive identical
examples and blocks.
"""

from __future__ import annotations
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
import numpy as np
import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from promptolution.tasks import ClassificationTask, RewardTask
from promptolution.tasks.base_task import BaseTask, EvalStrategy
from tri_fair.config.schema import DatasetConfig
from tri_fair.config.datasets import BIOS_PROFESSIONS, CIVIL_IDENTITIES
from tri_fair.config.setup import SETUP
from tri_fair.fairness.bbq import (
    attach_official_target_locations,
    prepare_bbq_dataframe,
)
from tri_fair.fairness.block_sampler import (
    MANIFEST_ID,
    build_fairness_blocks,
    sample_partition,
)
from tri_fair.data.task import FairnessTask, resolve_fairness_metric

logger = logging.getLogger(__name__)
MANIFEST_VERSION = 3


@dataclass
class SplitManifest:
    version: int
    dataset_alias: str
    dataset_name: str
    revision: Optional[str]
    seed: int
    partitions: Dict[str, list[str]]
    blocks: list[list[str]]
    metadata: Dict[str, object]

    @classmethod
    def from_json(cls, path: Path) -> "SplitManifest":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(**payload)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(asdict(self), handle, indent=2, sort_keys=True)
        temporary.replace(path)


def _validate_manifest(
    config: DatasetConfig, manifest: SplitManifest, *, seed: int
) -> None:
    """Fail closed when an existing manifest no longer matches configuration."""
    fairness = config.fairness
    if fairness is None:
        raise ValueError("Manifest validation is only valid for fairness datasets")
    expected_header = {
        "version": MANIFEST_VERSION,
        "dataset_alias": config.alias,
        "dataset_name": config.name,
        "revision": config.revision,
        "seed": int(seed),
    }
    actual_header = {
        "version": manifest.version,
        "dataset_alias": manifest.dataset_alias,
        "dataset_name": manifest.dataset_name,
        "revision": manifest.revision,
        "seed": manifest.seed,
    }
    if actual_header != expected_header:
        raise ValueError(
            f"Manifest header does not match the requested experiment. Expected {expected_header}, found {actual_header}."
        )
    required_partitions = {"dev", "few_shot", "test"}
    if set(manifest.partitions) != required_partitions:
        raise ValueError(
            f"Manifest partitions must be {sorted(required_partitions)}, found {sorted(manifest.partitions)}"
        )
    expected_sizes = {
        "dev": fairness.dev_size,
        "few_shot": fairness.fs_size,
        "test": fairness.test_size,
    }
    for name, expected_size in expected_sizes.items():
        ids = [str(value) for value in manifest.partitions[name]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Manifest partition {name!r} contains duplicate row IDs")
        if expected_size is not None and len(ids) != expected_size:
            raise ValueError(
                f"Manifest partition {name!r} has {len(ids)} rows, expected {expected_size}"
            )
    source_by_partition = {
        "dev": fairness.dev_source_split,
        "few_shot": fairness.fewshot_source_split,
        "test": fairness.test_source_split,
    }
    for left in required_partitions:
        for right in required_partitions:
            if left >= right or source_by_partition[left] != source_by_partition[right]:
                continue
            overlap = set(manifest.partitions[left]) & set(manifest.partitions[right])
            if overlap:
                raise ValueError(
                    f"Manifest leakage: {left} and {right} share {len(overlap)} IDs"
                )
    if len(manifest.blocks) != fairness.dev_size // fairness.block_size:
        raise ValueError(
            f"Manifest has {len(manifest.blocks)} blocks, expected {fairness.dev_size // fairness.block_size}"
        )
    if any((len(block) != fairness.block_size for block in manifest.blocks)):
        raise ValueError("Manifest contains a block with an unexpected size")
    flat_blocks = [str(value) for block in manifest.blocks for value in block]
    if len(flat_blocks) != len(set(flat_blocks)):
        raise ValueError("Manifest development blocks overlap")
    if set(flat_blocks) != set(map(str, manifest.partitions["dev"])):
        raise ValueError(
            "Manifest blocks do not exactly cover the development partition"
        )
    expected_metadata = {
        "dev_size": fairness.dev_size,
        "few_shot_size": fairness.fs_size,
        "test_size": fairness.test_size,
        "block_size": fairness.block_size,
        "n_blocks": fairness.dev_size // fairness.block_size,
        "metric_name": fairness.metric_name,
    }
    for key, expected in expected_metadata.items():
        if manifest.metadata.get(key) != expected:
            raise ValueError(
                f"Manifest metadata {key!r}={manifest.metadata.get(key)!r}; expected {expected!r}"
            )


def _process_df(df: pd.DataFrame, dataset_config: DatasetConfig) -> pd.DataFrame:
    """Original MO-CAPO dataframe standardisation."""
    df = df.copy()
    if callable(dataset_config.input):
        df.loc[:, "input"] = dataset_config.input(df)
    else:
        df.loc[:, "input"] = df[dataset_config.input]
    if callable(dataset_config.target):
        df.loc[:, "target"] = dataset_config.target(df)
    elif dataset_config.target is not None:
        df.loc[:, "target"] = df[dataset_config.target]
    return df


def _make_task(
    *,
    dataset_config: DatasetConfig,
    df: pd.DataFrame,
    eval_strategy: EvalStrategy,
    n_subsamples: int,
    seed: int,
    block_ids: Optional[list[list[str]]] = None,
) -> BaseTask:
    if dataset_config.task_type == "Classification":
        task: BaseTask = ClassificationTask(
            df=df,
            task_description=dataset_config.task_description,
            x_column="input",
            y_column="target",
            n_subsamples=n_subsamples,
            eval_strategy=eval_strategy,
            metric=dataset_config.metric_func,
            seed=seed,
        )
    elif dataset_config.task_type == "Reward":
        task = RewardTask(
            df=df,
            reward_function=dataset_config.metric_func,
            task_description=dataset_config.task_description,
            reward_columns=dataset_config.reward_columns,
            x_column="input",
            y_column="target",
            n_subsamples=n_subsamples,
            eval_strategy=eval_strategy,
            seed=seed,
        )
    elif dataset_config.task_type == "Fairness":
        fairness = dataset_config.fairness
        if fairness is None:
            raise ValueError(
                f"Dataset {dataset_config.alias!r} lacks fairness configuration"
            )
        task = FairnessTask(
            df=df,
            task_description=dataset_config.task_description or "",
            classes=list(fairness.class_names),
            quality_metric=fairness.quality_metric,
            fairness_metric=resolve_fairness_metric(fairness.metric_name),
            fairness_metric_name=fairness.metric_name,
            protected_columns=list(fairness.protected_columns),
            metadata_columns=list(fairness.metadata_columns),
            min_group_count=fairness.min_group_count,
            fairness_kwargs=fairness.fairness_kwargs,
            block_ids=block_ids,
            eval_strategy=eval_strategy,
            seed=seed,
            id_column=MANIFEST_ID,
            x_column="input",
            y_column="target",
        )
    else:
        raise ValueError(f"Unsupported task type: {dataset_config.task_type}")
    if dataset_config.alias in ["gsm8k", "mbpp"]:
        task.classes = None
    return task


def _make_manifest_id(split: str, values: Iterable[object]) -> str:
    components = [str(value).replace("|", "_") for value in values]
    return f"{split}|" + "|".join(components)


def _load_bbq(config: DatasetConfig) -> Dict[str, pd.DataFrame]:
    """Load pinned BBQ JSONL files without executing a dataset script."""
    if not config.subsets:
        raise ValueError("BBQ configuration must declare at least one subset")
    frames: list[pd.DataFrame] = []
    for subset in config.subsets:
        filename = f"data/{subset}.jsonl"
        local_path = hf_hub_download(
            repo_id=config.name,
            filename=filename,
            repo_type="dataset",
            revision=config.revision,
        )
        try:
            raw = pd.read_json(local_path, lines=True)
        except ValueError as error:
            raise ValueError(
                f"Failed to parse BBQ subset {subset!r} from {filename!r} at revision {config.revision!r}"
            ) from error
        if raw.empty:
            raise ValueError(
                f"BBQ subset {subset!r} is empty at revision {config.revision!r}"
            )
        if "category" not in raw.columns:
            raw["category"] = subset
        else:
            observed = set(raw["category"].dropna().astype(str).unique())
            if observed and observed != {subset}:
                raise ValueError(
                    f"BBQ file {filename!r} contains categories {sorted(observed)}; expected only {subset!r}"
                )
            raw["category"] = raw["category"].fillna(subset)
        raw = raw.reset_index(drop=True)
        raw["_source_index"] = np.arange(len(raw), dtype=int)
        raw["_source_split"] = "test"
        raw = prepare_bbq_dataframe(raw)
        raw[MANIFEST_ID] = raw.apply(
            lambda row: _make_manifest_id(
                "test", (row["category"], int(row["example_id"]))
            ),
            axis=1,
        )
        frames.append(raw)
    combined = pd.concat(frames, ignore_index=True)
    fairness = config.fairness
    assert fairness is not None
    combined = attach_official_target_locations(
        combined,
        cache_path=fairness.fairness_kwargs.get(
            "official_metadata_cache", "data/external/bbq/additional_metadata.csv"
        ),
        metadata_url=fairness.fairness_kwargs.get(
            "official_metadata_url",
            "https://raw.githubusercontent.com/nyu-mll/BBQ/bea11bd97d79217245b5871acd247b9d6eb24598/analysis_scripts/additional_metadata.csv",
        ),
        require_official=bool(
            fairness.fairness_kwargs.get("require_official_metadata", False)
        ),
    )
    if combined[MANIFEST_ID].duplicated().any():
        duplicates = (
            combined.loc[combined[MANIFEST_ID].duplicated(), MANIFEST_ID]
            .head()
            .tolist()
        )
        raise ValueError(f"Duplicate BBQ manifest IDs: {duplicates}")
    return {"test": combined}


def _load_bias_in_bios(config: DatasetConfig) -> Dict[str, pd.DataFrame]:
    fairness = config.fairness
    assert fairness is not None
    sources: Dict[str, pd.DataFrame] = {}
    for split in sorted(
        {
            fairness.dev_source_split,
            fairness.fewshot_source_split,
            fairness.test_source_split,
        }
    ):
        frame = load_dataset(
            config.name, split=split, revision=config.revision
        ).to_pandas()
        required = {"hard_text", "profession", "gender"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"Bias-in-Bios split {split!r} is missing {sorted(missing)}"
            )
        frame = frame.reset_index(drop=True)
        frame["_source_index"] = np.arange(len(frame), dtype=int)
        frame["_source_split"] = split
        frame["profession_id"] = frame["profession"].astype(int)
        if not frame["profession_id"].between(0, len(BIOS_PROFESSIONS) - 1).all():
            raise ValueError("Bias-in-Bios contains an unknown profession ID")
        frame["input"] = frame["hard_text"].astype(str)
        frame["target"] = frame["profession_id"].map(dict(enumerate(BIOS_PROFESSIONS)))
        frame["gender"] = frame["gender"].astype(int)
        frame[MANIFEST_ID] = frame["_source_index"].map(
            lambda index: _make_manifest_id(split, (index,))
        )
        sources[split] = frame
    return sources


def _find_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str:
    normalised = {str(column).casefold(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate.casefold() in normalised:
            return normalised[candidate.casefold()]
    raise ValueError(f"None of the expected columns {list(candidates)} exists")


def _load_civil_comments(config: DatasetConfig) -> Dict[str, pd.DataFrame]:
    path = hf_hub_download(
        repo_id=config.name,
        filename="all_data_with_identities.csv",
        repo_type="dataset",
        revision=config.revision,
    )
    raw = pd.read_csv(path, low_memory=False)
    text_column = _find_column(raw, ("comment_text", "text"))
    toxicity_column = _find_column(raw, ("toxicity", "label"))
    split_column = _find_column(raw, ("split",))
    source_id_candidates = [
        column
        for column in raw.columns
        if str(column).casefold() in {"id", "comment_id"}
    ]
    if source_id_candidates:
        source_ids = raw[source_id_candidates[0]].astype(str)
    else:
        source_ids = pd.Series(np.arange(len(raw), dtype=int), index=raw.index).astype(
            str
        )
    identity_map = {}
    for identity in CIVIL_IDENTITIES:
        candidates = (identity, "LGBTQ") if identity == "lgbtq" else (identity,)
        identity_map[identity] = _find_column(raw, candidates)
    sources: Dict[str, pd.DataFrame] = {}
    for split in ("train", "val", "test"):
        frame = (
            raw[raw[split_column].astype(str) == split].copy().reset_index(drop=True)
        )
        original_indices = source_ids.loc[
            raw[split_column].astype(str) == split
        ].reset_index(drop=True)
        frame["_source_index"] = original_indices.astype(str)
        frame["_source_split"] = split
        frame["input"] = frame[text_column].fillna("").astype(str)
        toxicity = pd.to_numeric(frame[toxicity_column], errors="coerce").fillna(0.0)
        frame["label_index"] = (toxicity >= 0.5).astype(int)
        frame["target"] = frame["label_index"].map({0: "non_toxic", 1: "toxic"})
        for target_name, source_name in identity_map.items():
            values = pd.to_numeric(frame[source_name], errors="coerce").fillna(0.0)
            frame[target_name] = (values >= 0.5).astype(int)
        frame[MANIFEST_ID] = frame["_source_index"].map(
            lambda index: _make_manifest_id(split, (index,))
        )
        sources[split] = frame
    return sources


def load_fairness_sources(config: DatasetConfig) -> Dict[str, pd.DataFrame]:
    if config.task_type != "Fairness":
        raise ValueError("load_fairness_sources is only valid for fairness datasets")
    if config.loader == "bbq_multi_config":
        return _load_bbq(config)
    if config.loader == "civil_comments_raw_csv":
        return _load_civil_comments(config)
    if config.alias == "bias_in_bios":
        return _load_bias_in_bios(config)
    raise ValueError(f"No fairness source loader registered for {config.alias!r}")


def _subset_by_ids(frame: pd.DataFrame, ids: Iterable[str]) -> pd.DataFrame:
    id_list = [str(value) for value in ids]
    position = {value: index for index, value in enumerate(id_list)}
    subset = frame[frame[MANIFEST_ID].astype(str).isin(position)].copy()
    subset["_manifest_order"] = subset[MANIFEST_ID].astype(str).map(position)
    subset = (
        subset.sort_values("_manifest_order")
        .drop(columns="_manifest_order")
        .reset_index(drop=True)
    )
    missing = set(id_list) - set(subset[MANIFEST_ID].astype(str))
    if missing:
        raise ValueError(
            f"Manifest references {len(missing)} missing IDs, e.g. {sorted(missing)[:3]}"
        )
    return subset


def _create_manifest(
    config: DatasetConfig, sources: Dict[str, pd.DataFrame], *, seed: int
) -> SplitManifest:
    fairness = config.fairness
    assert fairness is not None
    if config.alias == "bbq":
        shared = sources[fairness.dev_source_split]
        dev_sample = sample_partition(
            shared, dataset_alias=config.alias, size=fairness.dev_size, seed=seed + 11
        )
        fs_sample = sample_partition(
            dev_sample.remainder,
            dataset_alias=config.alias,
            size=fairness.fs_size,
            seed=seed + 23,
        )
        test_size = fairness.test_size or len(fs_sample.remainder)
        test_sample = sample_partition(
            fs_sample.remainder,
            dataset_alias=config.alias,
            size=test_size,
            seed=seed + 37,
        )
    else:
        dev_source = sources[fairness.dev_source_split]
        fs_source = sources[fairness.fewshot_source_split]
        test_source = sources[fairness.test_source_split]
        dev_sample = sample_partition(
            dev_source,
            dataset_alias=config.alias,
            size=fairness.dev_size,
            seed=seed + 11,
            identity_columns=fairness.protected_columns,
        )
        fs_sample = sample_partition(
            fs_source,
            dataset_alias=config.alias,
            size=fairness.fs_size,
            seed=seed + 23,
            identity_columns=fairness.protected_columns,
        )
        test_size = fairness.test_size or len(test_source)
        test_sample = sample_partition(
            test_source,
            dataset_alias=config.alias,
            size=test_size,
            seed=seed + 37,
            identity_columns=fairness.protected_columns,
        )
    dev_frame = _subset_by_ids(sources[fairness.dev_source_split], dev_sample.ids)
    blocks = build_fairness_blocks(
        dev_frame,
        dataset_alias=config.alias,
        block_size=fairness.block_size,
        seed=seed + 101,
        identity_columns=fairness.protected_columns,
    )
    return SplitManifest(
        version=MANIFEST_VERSION,
        dataset_alias=config.alias,
        dataset_name=config.name,
        revision=config.revision,
        seed=seed,
        partitions={
            "dev": dev_sample.ids,
            "few_shot": fs_sample.ids,
            "test": test_sample.ids,
        },
        blocks=blocks,
        metadata={
            "dev_size": len(dev_sample.ids),
            "few_shot_size": len(fs_sample.ids),
            "test_size": len(test_sample.ids),
            "block_size": fairness.block_size,
            "n_blocks": len(blocks),
            "metric_name": fairness.metric_name,
        },
    )


def load_or_create_manifest(
    config: DatasetConfig,
    sources: Dict[str, pd.DataFrame],
    *,
    seed: int,
    manifest_dir: str | Path = "data/splits",
    regenerate: bool = False,
) -> tuple[SplitManifest, Path]:
    path = Path(manifest_dir) / f"{config.alias}_seed{seed}.json"
    if path.exists() and (not regenerate):
        manifest = SplitManifest.from_json(path)
        _validate_manifest(config, manifest, seed=seed)
        return (manifest, path)
    manifest = _create_manifest(config, sources, seed=seed)
    _validate_manifest(config, manifest, seed=seed)
    manifest.save(path)
    logger.info("Created immutable split manifest at %s", path)
    return (manifest, path)


def _fairness_frames(
    dataset_config: DatasetConfig,
    *,
    seed: int,
    manifest_dir: str | Path,
    regenerate_manifest: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitManifest, Path]:
    fairness = dataset_config.fairness
    assert fairness is not None
    sources = load_fairness_sources(dataset_config)
    manifest, manifest_path = load_or_create_manifest(
        dataset_config,
        sources,
        seed=seed,
        manifest_dir=manifest_dir,
        regenerate=regenerate_manifest,
    )
    dev_df = _subset_by_ids(
        sources[fairness.dev_source_split], manifest.partitions["dev"]
    )
    fs_df = _subset_by_ids(
        sources[fairness.fewshot_source_split], manifest.partitions["few_shot"]
    )
    test_df = _subset_by_ids(
        sources[fairness.test_source_split], manifest.partitions["test"]
    )
    return (dev_df, fs_df, test_df, manifest, manifest_path)


def create_dev_tasks(
    dataset_config: DatasetConfig,
    eval_strategy: EvalStrategy,
    n_subsamples: int = 30,
    dev_size: int = 300,
    fs_size: int = 200,
    seed: int = 42,
    manifest_dir: str | Path = "data/splits",
    regenerate_manifest: bool = False,
) -> Tuple[BaseTask, pd.DataFrame]:
    """Create development task and few-shot pool.

    For fairness datasets, dataset-specific sizes from ``FairnessConfig`` take
    precedence over the legacy ``dev_size`` and ``fs_size`` arguments.
    """
    if dataset_config.task_type == "Fairness":
        dev_df, fs_df, _, manifest, manifest_path = _fairness_frames(
            dataset_config,
            seed=seed,
            manifest_dir=manifest_dir,
            regenerate_manifest=regenerate_manifest,
        )
        task = _make_task(
            dataset_config=dataset_config,
            df=dev_df,
            eval_strategy=eval_strategy,
            n_subsamples=dataset_config.fairness.block_size,
            seed=seed,
            block_ids=manifest.blocks,
        )
        task.manifest_path = str(manifest_path)
        return (task, fs_df)
    train_df = load_dataset(
        dataset_config.name,
        name=dataset_config.names.train,
        split=dataset_config.splits.train,
        revision=dataset_config.revision,
    ).to_pandas()
    if dataset_config.alias == "mbpp":
        test_df = train_df.sample(SETUP.test_size, random_state=42)
        train_df = train_df.drop(test_df.index)
    if len(train_df) < dev_size + fs_size:
        raise ValueError(
            "Not enough data in training split to create dev and few-shot splits"
        )
    train_sample = train_df.sample(dev_size + fs_size, random_state=seed, replace=False)
    dev_df = _process_df(train_sample.iloc[:dev_size], dataset_config)
    fs_df = _process_df(train_sample.iloc[dev_size:], dataset_config)
    dev_task = _make_task(
        dataset_config=dataset_config,
        df=dev_df,
        eval_strategy=eval_strategy,
        n_subsamples=n_subsamples,
        seed=seed,
    )
    return (dev_task, fs_df)


def create_test_task(
    dataset_config: DatasetConfig,
    eval_strategy: EvalStrategy,
    n_subsamples: int = 30,
    test_size: int = 500,
    seed: int = 42,
    manifest_dir: str | Path = "data/splits",
    regenerate_manifest: bool = False,
) -> BaseTask:
    if dataset_config.task_type == "Fairness":
        _, _, test_df, _, manifest_path = _fairness_frames(
            dataset_config,
            seed=seed,
            manifest_dir=manifest_dir,
            regenerate_manifest=regenerate_manifest,
        )
        task = _make_task(
            dataset_config=dataset_config,
            df=test_df,
            eval_strategy="full",
            n_subsamples=max(1, len(test_df)),
            seed=seed,
            block_ids=None,
        )
        task.manifest_path = str(manifest_path)
        return task
    test_df = load_dataset(
        dataset_config.name,
        name=dataset_config.names.test,
        split=dataset_config.splits.test,
        revision=dataset_config.revision,
    ).to_pandas()
    if dataset_config.alias == "mbpp":
        test_df = test_df.sample(SETUP.test_size, random_state=42)
    if len(test_df) >= test_size:
        test_df = test_df.sample(test_size, random_state=seed, replace=False)
    else:
        logger.warning(
            "Not enough data in test split for %s; using all %d samples",
            dataset_config.alias,
            len(test_df),
        )
    test_df = _process_df(test_df, dataset_config)
    return _make_task(
        dataset_config=dataset_config,
        df=test_df,
        eval_strategy=eval_strategy,
        n_subsamples=n_subsamples,
        seed=seed,
    )
