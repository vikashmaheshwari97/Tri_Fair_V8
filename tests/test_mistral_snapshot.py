"""Find pinned Mistral snapshots and reject placeholder or incomplete paths."""

from pathlib import Path
from dataclasses import replace
import pytest
from tri_fair.runtime import models
from tests.test_gptoss_runtime import make_config, FakeWrapper

REVISION = "95a6d26c4bfb886c58daf9d3f7332c857cb27b43"
REPOSITORY = "models--mistralai--Mistral-Small-3.2-24B-Instruct-2506"


def config(tmp_path):
    value = make_config(
        model="vllm-mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        alias="mistral-3-24b", revision=REVISION,
    )
    return replace(value, model_storage_path=tmp_path / "missing-flat-directory")


def stage(path):
    path.mkdir(parents=True)
    for name in models.MISTRAL_REQUIRED_FILES:
        (path / name).write_text("{}", encoding="utf-8")
    (path / "consolidated.safetensors").write_bytes(b"fake weights")
    return path


def test_placeholder_reports_unset_instruction(monkeypatch, tmp_path):
    monkeypatch.setenv("MISTRAL_LOCAL_SNAPSHOT", "/absolute/path/to/the/pinned/mistral/snapshot")
    with pytest.raises(FileNotFoundError, match="unset MISTRAL_LOCAL_SNAPSHOT"):
        models._resolve_mistral_snapshot(config(tmp_path))


def test_explicit_missing_path_does_not_fall_back(monkeypatch, tmp_path):
    good = stage(tmp_path / "good")
    monkeypatch.setenv("MISTRAL_LOCAL_SNAPSHOT", str(tmp_path / "wrong"))
    cfg = replace(config(tmp_path), model_storage_path=good)
    with pytest.raises(FileNotFoundError, match="could not be resolved"):
        models._resolve_mistral_snapshot(cfg)


def test_hub_discovery_uses_pinned_revision(monkeypatch, tmp_path):
    monkeypatch.delenv("MISTRAL_LOCAL_SNAPSHOT", raising=False)
    monkeypatch.setenv("MISTRAL_HF_CACHE", str(tmp_path))
    good = stage(tmp_path / REPOSITORY / "snapshots" / REVISION)
    stage(tmp_path / REPOSITORY / "snapshots" / "other-revision")
    assert models._resolve_mistral_snapshot(config(tmp_path)) == good.resolve()


@pytest.mark.parametrize("defect", ["missing", "empty", "pointer"])
def test_incomplete_snapshots_are_rejected(tmp_path, defect):
    snapshot = stage(tmp_path / "snapshot")
    if defect == "missing":
        (snapshot / "tekken.json").unlink()
    elif defect == "empty":
        (snapshot / "params.json").write_bytes(b"")
    else:
        (snapshot / "consolidated.safetensors").write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
    assert not models._mistral_snapshot_is_complete(snapshot)[0]


def test_create_llm_uses_verified_mistral_directory(monkeypatch, tmp_path):
    snapshot = stage(tmp_path / "snapshot")
    monkeypatch.setenv("MISTRAL_LOCAL_SNAPSHOT", str(snapshot))
    monkeypatch.setattr(models, "get_available_gpu_memory_gb", lambda: 80)
    monkeypatch.setattr(models, "VLLM", FakeWrapper)
    wrapper = models.create_llm(config(tmp_path), seed=72)
    assert Path(wrapper.args[0]) == snapshot.resolve()
    assert "revision" not in wrapper.kwargs["llm_kwargs"]
