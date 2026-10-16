"""Typed configuration objects for MO-CAPO and Tri-Fair.

The original MO-CAPO configuration schema is retained, while fairness-aware
experiments add immutable split, protected-attribute, and group-complete block
metadata.  Dataset-specific pilot sizes live in :class:`FairnessConfig`; the
legacy global setup remains available for the original benchmark datasets.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple
from promptolution.tasks.base_task import EvalStrategy

DatasetType = Literal["bbq", "bias_in_bios", "civil_comments"]
TaskType = Literal["Classification", "Reward", "Fairness"]
OptimizerType = str
ModelType = Literal["gpt-oss-120b", "qwen-3-30b", "mistral-3-24b"]
FairnessMetricName = Literal[
    "bbq_bias",
    "bios_tpr_gap",
    "civil_worst_group",
    "bbq_bias_statistical",
    "bios_tpr_gap_statistical",
    "civil_equalized_odds",
]
QualityMetricName = Literal["accuracy", "macro_f1"]
DatasetLoaderName = Literal["datasets", "bbq_multi_config", "civil_comments_raw_csv"]


@dataclass(frozen=True)
class SplitConfig:
    """Optional dataset configuration name and split names."""

    train: Optional[str]
    test: Optional[str]


@dataclass(frozen=True)
class FairnessConfig:
    """Dataset-specific fairness and immutable-manifest settings."""

    metric_name: FairnessMetricName
    quality_metric: QualityMetricName
    protected_columns: Tuple[str, ...]
    metadata_columns: Tuple[str, ...]
    class_names: Tuple[str, ...]
    dev_size: int
    fs_size: int
    test_size: Optional[int]
    block_size: int
    min_group_count: int
    dev_source_split: str
    fewshot_source_split: str
    test_source_split: str
    id_columns: Tuple[str, ...] = ()
    group_unit_column: Optional[str] = None
    stratify_columns: Tuple[str, ...] = ()
    fairness_kwargs: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.dev_size <= 0 or self.fs_size <= 0:
            raise ValueError("dev_size and fs_size must be positive")
        if self.test_size is not None and self.test_size <= 0:
            raise ValueError("test_size must be positive or None")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.dev_size % self.block_size != 0:
            raise ValueError(
                f"dev_size={self.dev_size} must be divisible by block_size={self.block_size}"
            )
        if self.min_group_count <= 0:
            raise ValueError("min_group_count must be positive")
        if not self.class_names:
            raise ValueError("class_names cannot be empty")
        if any((label != label.lower() for label in self.class_names)):
            raise ValueError("class_names must be lower-case for MarkerBasedPredictor")
        required_metadata = set(self.protected_columns) - set(self.metadata_columns)
        if required_metadata:
            raise ValueError(
                f"Every protected column must also be retained as metadata; missing {sorted(required_metadata)}"
            )


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    optimizer: OptimizerType
    optimizer_params: Dict[str, Any]
    eval_strategy: EvalStrategy
    n_subsamples: int = 30


@dataclass(frozen=True)
class ModelConfig:
    model: str
    alias: ModelType
    max_model_len: int
    batch_size: int
    input_costs: float
    output_costs: float
    model_storage_path: Path | str
    llm_kwargs: Dict[str, Any]

    def validate(self) -> None:
        if self.max_model_len <= 0 or self.batch_size <= 0:
            raise ValueError(f"Invalid model capacity for {self.alias}")
        if self.input_costs < 0 or self.output_costs < 0:
            raise ValueError(f"Token costs cannot be negative for {self.alias}")


@dataclass
class DatasetConfig:
    """Configuration for conventional and fairness-aware datasets."""

    name: str
    alias: DatasetType
    revision: Optional[str]
    input: str | Callable
    target: str | Callable | None
    task_type: TaskType
    metric_func: Optional[Callable]
    names: SplitConfig = field(
        default_factory=lambda: SplitConfig(train=None, test=None)
    )
    splits: SplitConfig = field(
        default_factory=lambda: SplitConfig(train="train", test="test")
    )
    initial_prompts: List[str] = field(default_factory=list)
    task_description: Optional[str] = None
    reward_columns: List[str] = field(default_factory=list)
    loader: DatasetLoaderName = "datasets"
    subsets: Tuple[str, ...] = ()
    fairness: Optional[FairnessConfig] = None

    def validate(self) -> None:
        if not self.alias or not self.name:
            raise ValueError("Dataset name and alias cannot be empty")
        if not self.initial_prompts:
            raise ValueError(f"Dataset {self.alias!r} has no initial prompts")
        if self.task_type == "Fairness":
            if self.fairness is None:
                raise ValueError(
                    f"Fairness dataset {self.alias!r} needs FairnessConfig"
                )
            self.fairness.validate()
            if len(self.initial_prompts) < 6:
                raise ValueError(
                    f"Fairness dataset {self.alias!r} requires at least six initial prompts"
                )
        elif self.fairness is not None:
            raise ValueError(
                f"Non-fairness dataset {self.alias!r} cannot define FairnessConfig"
            )


@dataclass(frozen=True)
class SetupConfig:
    dev_size: int
    fs_size: int
    test_size: int
    n_steps: int

    def validate(self) -> None:
        if min(self.dev_size, self.fs_size, self.test_size, self.n_steps) <= 0:
            raise ValueError("All setup sizes and n_steps must be positive")
