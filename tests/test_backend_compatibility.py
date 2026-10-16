"""Reproduce the missing Pixtral symbol without needing GPU dependencies."""

import json
from types import SimpleNamespace
import pytest
from tri_fair.runtime import backend_compatibility as backend

OLD_PROVIDER = """
class PixtralRotaryEmbedding: pass
def apply_rotary_pos_emb(): pass
def position_ids_in_meshgrid(): pass
"""
NEW_PROVIDER = OLD_PROVIDER.replace("PixtralRotaryEmbedding", "PixtralVisionRotaryEmbedding")
CONSUMER = "from transformers.models.pixtral.modeling_pixtral import PixtralRotaryEmbedding, apply_rotary_pos_emb, position_ids_in_meshgrid"


@pytest.mark.parametrize("source,missing", [
    (OLD_PROVIDER, []), (NEW_PROVIDER, ["PixtralRotaryEmbedding"]),
])
def test_pixtral_import_contract(source, missing):
    assert backend.check_pixtral_sources(CONSUMER, source) == missing


def test_all_imported_symbols_are_checked():
    source = CONSUMER + ", another_helper"
    assert backend.check_pixtral_sources(source, OLD_PROVIDER) == ["another_helper"]


@pytest.mark.parametrize("version,provider,ok", [
    ("5.16.1", OLD_PROVIDER, True),
    ("5.20.0", NEW_PROVIDER, False),
    ("5.16.1", NEW_PROVIDER, False),
])
def test_installed_backend_check(monkeypatch, tmp_path, version, provider, ok):
    files = {}
    for name, source in [("vllm", CONSUMER), ("transformers", provider)]:
        path = tmp_path / f"{name}.py"
        path.write_text(source, encoding="utf-8")
        files[name] = path
    monkeypatch.setattr(backend.metadata, "version", lambda name: "0.24.0" if name == "vllm" else version)
    monkeypatch.setattr(backend.metadata, "distribution", lambda name: SimpleNamespace(locate_file=lambda relative: files[name]))
    report = backend.inspect_backend()
    assert report["ok"] is ok and report["gpu_used"] is False
    if not ok:
        with pytest.raises(RuntimeError, match="Mistral backend preflight failed"):
            backend.require_mistral_backend()
    assert json.loads(json.dumps(report))["ok"] is ok


def test_missing_gpu_packages_fail_clearly(monkeypatch):
    def missing(name):
        raise backend.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(backend.metadata, "version", missing)
    assert not backend.inspect_backend()["ok"]
