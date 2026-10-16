"""Guard the clean package boundaries and its preflight entry points."""

import ast
from pathlib import Path
from types import SimpleNamespace
import pytest
from tri_fair.config.registry import METHODS
from tri_fair.optimizers.trifair import TriFairOptimizer
from tests.fakes import task_and_pool, FakeLLM, FakePredictor


def test_no_historical_optimizer_or_script_imports():
    _check_package_boundaries()


@pytest.mark.parametrize("default_encoding", ["cp1252", "ascii"])
def test_architecture_scan_with_non_utf8_default(monkeypatch, default_encoding):
    """Exercise the complete source scan with Windows-style text defaults."""
    original_read_text = Path.read_text

    def read_with_platform_default(path, encoding=None, errors=None):
        return original_read_text(
            path, encoding=encoding or default_encoding, errors=errors
        )

    monkeypatch.setattr(Path, "read_text", read_with_platform_default)
    _check_package_boundaries()


def _check_package_boundaries():
    import tri_fair

    root = Path(tri_fair.__file__).parent
    for file in root.rglob("*.py"):
        tree = ast.parse(file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules = (
                [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else []
            )
            assert not any(
                m == "src" or m.startswith(("src.", "scripts.", "analysis."))
                for m in modules
            ), file
    assert not any(
        "V7" in cls.__name__ or "V6" in cls.__name__ for cls in TriFairOptimizer.__mro__
    )
    assert len(METHODS) == 10


def test_installed_package_doctor():
    from tri_fair.cli.doctor import inspect_environment

    report = inspect_environment()
    assert report["ok"] and report["packages"]["tri-fair-research"] == "8.1.0"


@pytest.mark.parametrize("valid", [True, False])
def test_gpu_smoke_uses_search_only_and_reports_parsing(monkeypatch, valid):
    import tri_fair.cli.smoke as smoke

    task, pool = task_and_pool()
    monkeypatch.setattr(smoke, "create_dev_tasks", lambda **kw: (task, pool))
    monkeypatch.setattr(smoke, "create_llm", lambda **kw: FakeLLM())

    def predictor(llm, classes):
        p = FakePredictor()
        p.llm = llm
        if not valid:
            p.predict = lambda prompts, xs: (["invalid"] * len(xs), xs)
        return p

    monkeypatch.setattr(smoke, "MarkerBasedPredictor", predictor)
    result = smoke.run(
        SimpleNamespace(
            dataset="bias_in_bios",
            model="qwen-3-30b",
            random_seed=42,
            manifest_dir="unused",
        )
    )
    assert result["search_examples"] == 2 and result["parsing_ok"] is valid
    assert result["benchmark_result"] is False
