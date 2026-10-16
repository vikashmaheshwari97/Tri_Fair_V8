from pathlib import Path

from tri_fair.config.schema import ModelConfig
from tri_fair.runtime import models as llm_creation


class FakeWrapper:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def make_config(*, model: str, alias: str, revision: str) -> ModelConfig:
    return ModelConfig(
        model=model,
        alias=alias,
        max_model_len=3000,
        batch_size=14,
        input_costs=0.0,
        output_costs=0.0,
        model_storage_path=Path("../models/test"),
        llm_kwargs={"revision": revision},
    )


def stage_gptoss_snapshot(root: Path) -> Path:
    root.mkdir(parents=True)
    for name in ("config.json", "tokenizer.json", "model.safetensors.index.json"):
        (root / name).write_text("{}", encoding="utf-8")
    for index in range(1, 16):
        (root / f"model-{index:05d}-of-00015.safetensors").write_bytes(b"x")
    return root


def test_gptoss_uses_verified_local_snapshot(monkeypatch, tmp_path):
    snapshot = stage_gptoss_snapshot(tmp_path / "snapshot")
    monkeypatch.setenv("GPT_OSS_LOCAL_SNAPSHOT", str(snapshot))
    monkeypatch.setattr(llm_creation, "get_available_gpu_memory_gb", lambda: 140.0)
    monkeypatch.setattr(llm_creation, "GPTOSSVLLM", FakeWrapper)

    wrapper = llm_creation.create_llm(
        make_config(
            model="vllm-openai/gpt-oss-120b",
            alias="gpt-oss-120b",
            revision="pinned-revision",
        ),
        seed=42,
    )

    assert wrapper.args == (str(snapshot.resolve()),)
    assert wrapper.kwargs["reasoning_effort"] == "low"
    assert "revision" not in wrapper.kwargs["llm_kwargs"]
    assert wrapper.kwargs["batch_size"] == 14


def test_qwen_uses_verified_local_snapshot(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_creation, "get_available_gpu_memory_gb", lambda: 140.0)
    monkeypatch.setattr(llm_creation, "VLLM", FakeWrapper)

    snapshot = tmp_path / "qwen-snapshot"
    monkeypatch.setattr(
        llm_creation, "_resolve_qwen_snapshot", lambda *a, **kw: snapshot
    )
    wrapper = llm_creation.create_llm(
        make_config(
            model="vllm-Qwen/Qwen3-30B-A3B-Instruct-2507",
            alias="qwen-3-30b",
            revision="qwen-revision",
        ),
        seed=43,
    )

    assert wrapper.args == (str(snapshot),)
    assert "revision" not in wrapper.kwargs["llm_kwargs"]
    assert "reasoning_effort" not in wrapper.kwargs


def test_other_models_keep_hub_routing(monkeypatch):
    monkeypatch.setattr(llm_creation, "get_available_gpu_memory_gb", lambda: 140.0)
    monkeypatch.setattr(llm_creation, "VLLM", FakeWrapper)
    wrapper = llm_creation.create_llm(
        make_config(model="vllm-example/model", alias="other", revision="pinned"),
        seed=43,
    )
    assert wrapper.args == ("example/model",)
    assert wrapper.kwargs["llm_kwargs"]["revision"] == "pinned"
