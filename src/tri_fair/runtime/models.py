"""Construct configured vLLM-backed Promptolution language models.

Configuration inspection remains CPU-safe. Real model execution requires Linux,
CUDA, PyTorch, vLLM, and a complete local model snapshot when Rocket is offline.
"""

from __future__ import annotations
import importlib
import os
from pathlib import Path
from typing import Any, Iterable
from promptolution.llms import VLLM
from tri_fair.config.schema import ModelConfig

GPT_OSS_ALIAS = "gpt-oss-120b"
QWEN_ALIAS = "qwen-3-30b"
MISTRAL_ALIAS = "mistral-3-24b"
GPT_OSS_REQUIRED_SHARDS = 15
MISTRAL_REQUIRED_FILES = ("config.json", "params.json", "tekken.json")
GPT_OSS_REASONING_EFFORTS = frozenset({"low", "medium", "high"})


class GPTOSSVLLM(VLLM):
    """Promptolution vLLM wrapper with GPT-OSS reasoning-effort rendering."""

    def __init__(
        self, *args: Any, reasoning_effort: str = "low", **kwargs: Any
    ) -> None:
        normalized = str(reasoning_effort).strip().casefold()
        if normalized not in GPT_OSS_REASONING_EFFORTS:
            raise ValueError(
                f"GPT-OSS reasoning_effort must be one of {sorted(GPT_OSS_REASONING_EFFORTS)}, got {reasoning_effort!r}"
            )
        self.reasoning_effort = normalized
        super().__init__(*args, **kwargs)

    def _get_response(self, prompts: list[str], system_prompts: list[str]) -> list[str]:
        rendered_prompts = [
            str(
                self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    reasoning_effort=self.reasoning_effort,
                )
            )
            for prompt, system_prompt in zip(prompts, system_prompts)
        ]
        responses: list[str] = []
        for index in range(0, len(rendered_prompts), self.batch_size):
            batch = rendered_prompts[index : index + self.batch_size]
            outputs = self.llm.generate(batch, self.sampling_params)
            responses.extend((output.outputs[0].text for output in outputs))
        return responses


def _load_torch():
    try:
        return importlib.import_module("torch")
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "PyTorch could not be imported. Tri-Fair model execution requires a working PyTorch/CUDA installation on the Linux GPU cluster."
        ) from error


def get_available_gpu_memory_gb(device_index: int = 0) -> float:
    torch = _load_torch()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required to create the configured vLLM model")
    properties = torch.cuda.get_device_properties(device_index)
    return float(properties.total_memory) / 1024**3


def _resolved_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _snapshot_is_complete(
    snapshot: Path, *, required_shards: int | None = None
) -> tuple[bool, str]:
    if not snapshot.is_dir():
        return (False, "not a directory")
    required = ("config.json",)
    missing = [name for name in required if not (snapshot / name).is_file()]
    if missing:
        return (False, f"missing {missing}")
    tokenizer_present = any(
        (
            (snapshot / name).is_file()
            for name in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model")
        )
    )
    if not tokenizer_present:
        return (False, "missing tokenizer files")
    index_present = (snapshot / "model.safetensors.index.json").is_file()
    shards = sorted(snapshot.glob("model-*.safetensors"))
    single_weights = sorted(snapshot.glob("*.safetensors"))
    if not index_present and (not shards) and (not single_weights):
        return (False, "missing safetensor weights")
    if required_shards is not None and len(shards) != int(required_shards):
        return (False, f"expected {required_shards} model shards, found {len(shards)}")
    return (True, "ok")


def _unique_paths(values: Iterable[str | Path | None]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for value in values:
        if value is None or not str(value).strip():
            continue
        path = _resolved_path(str(value))
        key = str(path)
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def _cache_roots(model_config: ModelConfig) -> list[Path]:
    return _unique_paths(
        (
            os.environ.get("HF_HUB_CACHE"),
            os.environ.get("HUGGINGFACE_HUB_CACHE"),
            model_config.model_storage_path,
            Path.home() / ".cache/huggingface/hub",
        )
    )


def _resolve_qwen_snapshot(
    model_config: ModelConfig, *, model_name: str, revision: str | None
) -> Path:
    explicit = os.environ.get("QWEN_LOCAL_SNAPSHOT", "").strip()
    repo_cache_name = "models--" + model_name.replace("/", "--")
    candidates: list[Path] = []
    if explicit:
        candidates.append(_resolved_path(explicit))
    for root in _cache_roots(model_config):
        candidates.append(root)
        if revision:
            candidates.extend(
                (
                    root / repo_cache_name / "snapshots" / revision,
                    root / "snapshots" / revision,
                    root / revision,
                )
            )
    checked: list[str] = []
    for candidate in _unique_paths(candidates):
        complete, reason = _snapshot_is_complete(candidate)
        checked.append(f"{candidate} ({reason})")
        if complete:
            return candidate
    details = "\n  - ".join(checked) if checked else "no candidate paths"
    raise FileNotFoundError(
        f"Qwen-3-30B local snapshot could not be resolved while Rocket is offline. Set QWEN_LOCAL_SNAPSHOT to the complete snapshot directory. Checked:\n  - {details}"
    )


def _resolve_gpt_oss_snapshot() -> Path:
    raw = os.environ.get("GPT_OSS_LOCAL_SNAPSHOT", "").strip()
    if not raw:
        raise RuntimeError(
            "GPT_OSS_LOCAL_SNAPSHOT must point to the verified local GPT-OSS-120B snapshot directory"
        )
    snapshot = _resolved_path(raw)
    complete, reason = _snapshot_is_complete(
        snapshot, required_shards=GPT_OSS_REQUIRED_SHARDS
    )
    if not complete:
        raise RuntimeError(f"Invalid GPT-OSS local snapshot {snapshot}: {reason}")
    return snapshot


def _mistral_snapshot_is_complete(snapshot: Path) -> tuple[bool, str]:
    """Validate the official Mistral-format local snapshot used by vLLM."""
    if not snapshot.is_dir():
        return (False, "not a directory")
    missing = [
        name for name in MISTRAL_REQUIRED_FILES
        if not (snapshot / name).is_file() or (snapshot / name).stat().st_size == 0
    ]
    if missing:
        return (False, f"missing {missing}")
    weights = sorted(snapshot.glob("consolidated*.safetensors"))
    if not weights:
        return (False, "missing consolidated*.safetensors weights")
    empty = [path.name for path in weights if path.stat().st_size <= 0]
    if empty:
        return (False, f"empty weight files {empty}")
    for path in weights:
        with path.open("rb") as handle:
            if handle.read(100).startswith(b"version https://git-lfs.github.com/spec/"):
                return (False, f"Git LFS pointer instead of downloaded weights: {path.name}")
    return (True, "ok")


def _resolve_mistral_snapshot(model_config: ModelConfig) -> Path:
    """Resolve an explicit directory or the pinned official-format cache snapshot."""
    explicit = os.environ.get("MISTRAL_LOCAL_SNAPSHOT", "").strip()
    candidates = []
    if explicit:
        if "/absolute/path/to/" in explicit.replace("\\", "/"):
            raise FileNotFoundError(
                "MISTRAL_LOCAL_SNAPSHOT contains an example placeholder. "
                "Run 'unset MISTRAL_LOCAL_SNAPSHOT' to use pinned cache discovery, "
                "or set it to the real complete Mistral snapshot directory."
            )
        # An explicit override is intentional: do not silently select another model.
        candidates = _unique_paths((explicit,))
    else:
        revision = model_config.llm_kwargs.get("revision")
        repository = "models--mistralai--Mistral-Small-3.2-24B-Instruct-2506"
        roots = _unique_paths((
            *_cache_roots(model_config),
            os.environ.get("MISTRAL_HF_CACHE"),
            os.environ.get("TF_HF_HUB_CACHE"),
            Path(os.environ["TF_HF_HOME"]) / "hub" if os.environ.get("TF_HF_HOME") else None,
            Path.home() / "projects/models/Mistral-3-24b",
            Path.home() / "models/Mistral-3-24b",
        ))
        # A deliberately staged flat model directory remains supported.
        candidates.append(model_config.model_storage_path)
        for root in roots:
            if root.name == "Mistral-3-24b":
                candidates.append(root)
            if revision:
                candidates.extend((
                    root / repository / "snapshots" / str(revision),
                    root / "snapshots" / str(revision),
                    root / str(revision),
                ))
        candidates = _unique_paths(candidates)
    checked: list[str] = []
    for candidate in candidates:
        complete, reason = _mistral_snapshot_is_complete(candidate)
        checked.append(f"{candidate} ({reason})")
        if complete:
            return candidate
    details = "\n  - ".join(checked) if checked else "no candidate paths"
    raise FileNotFoundError(
        f"Mistral-Small-3.2-24B local snapshot could not be resolved. Set MISTRAL_LOCAL_SNAPSHOT to the complete official-format model directory containing config.json, params.json, tekken.json, and consolidated*.safetensors. Checked:\n  - {details}"
    )


def create_llm(model_config: ModelConfig, seed: int) -> VLLM:
    """Create one vLLM wrapper without mutating the shared ModelConfig."""
    if not model_config.model.startswith("vllm-"):
        raise ValueError(
            f"Unsupported model type {model_config.model!r}; only vllm-* models are supported"
        )
    model_name = model_config.model.removeprefix("vllm-")
    available_memory_gb = get_available_gpu_memory_gb()
    optimal_batch_size = max(
        1, int(round(model_config.batch_size * min(1.0, available_memory_gb / 80.0)))
    )
    llm_kwargs: dict[str, Any] = dict(model_config.llm_kwargs)
    gpu_utilization = float(llm_kwargs.pop("gpu_memory_utilization", 0.9))
    tensor_parallel_size = int(llm_kwargs.pop("tensor_parallel_size", 1))
    dtype = str(llm_kwargs.pop("dtype", "auto"))
    trust_remote_code = bool(llm_kwargs.pop("trust_remote_code", False))
    wrapper_type: type[VLLM] = VLLM
    wrapper_kwargs: dict[str, Any] = {}
    if model_config.alias == QWEN_ALIAS:
        revision = llm_kwargs.get("revision")
        model_name = str(
            _resolve_qwen_snapshot(
                model_config,
                model_name=model_name,
                revision=str(revision) if revision is not None else None,
            )
        )
        llm_kwargs.pop("revision", None)
    elif model_config.alias == GPT_OSS_ALIAS:
        model_name = str(_resolve_gpt_oss_snapshot())
        llm_kwargs.pop("revision", None)
        wrapper_type = GPTOSSVLLM
        wrapper_kwargs["reasoning_effort"] = "low"
    elif model_config.alias == MISTRAL_ALIAS:
        from tri_fair.runtime.backend_compatibility import require_mistral_backend

        require_mistral_backend()
        model_name = str(_resolve_mistral_snapshot(model_config))
        llm_kwargs.pop("revision", None)
    return wrapper_type(
        model_name,
        batch_size=optimal_batch_size,
        max_model_len=model_config.max_model_len,
        model_storage_path=str(model_config.model_storage_path),
        seed=int(seed),
        gpu_memory_utilization=gpu_utilization,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        llm_kwargs=llm_kwargs,
        **wrapper_kwargs,
    )
