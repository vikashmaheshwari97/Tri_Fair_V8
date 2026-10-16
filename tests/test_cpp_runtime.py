"""Regression checks for the FlashInfer GLIBCXX_3.4.29 loader failure."""

from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
from tri_fair.cli import cpp_runtime as runtime

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux dynamic loader checks")


def library(tmp_path, name, symbol=b"GLIBCXX_3.4.29"):
    path = tmp_path / name / "libstdc++.so.6"
    path.parent.mkdir()
    path.write_bytes(b"ELF\0" + symbol + b"\0")
    return path


def test_old_gcc_runtime_is_skipped(tmp_path):
    old = library(tmp_path, "gcc9", b"GLIBCXX_3.4.28")
    new = library(tmp_path, "gcc11")
    probed = []

    def probe(path):
        probed.append(path)
        return True, ""

    assert runtime.resolve_runtime([old, new], probe=probe) == new.resolve()
    assert probed == [new.resolve()]


def test_unloadable_candidate_falls_back(tmp_path):
    bad = library(tmp_path, "incompatible")
    good = library(tmp_path, "system")
    assert runtime.resolve_runtime(
        [bad, good], probe=lambda p: (p == good.resolve(), "GLIBC unavailable")
    ) == good.resolve()


def test_failure_is_actionable(tmp_path):
    with pytest.raises(RuntimeError, match="TF_LIBSTDCXX_PATH"):
        runtime.resolve_runtime([tmp_path / "missing"])


def test_explicit_override_does_not_silently_fall_back(monkeypatch, tmp_path):
    override = tmp_path / "missing.so"
    monkeypatch.setenv("TF_LIBSTDCXX_PATH", str(override))
    assert runtime.candidate_paths() == [override]
    with pytest.raises(RuntimeError):
        runtime.resolve_runtime()


def test_fresh_process_receives_new_runtime(monkeypatch):
    seen = {}

    def run(command, **kwargs):
        seen.update(command=command, **kwargs)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(runtime.subprocess, "run", run)
    monkeypatch.setenv("LD_PRELOAD", "/old/libstdc++.so.6:/opt/libother.so")
    ok, detail = runtime.probe_library(Path("/new/libstdc++.so.6"))
    assert ok and not detail
    assert seen["env"]["LD_PRELOAD"] == "/new/libstdc++.so.6:/opt/libother.so"
    assert seen["command"][-1] == "/new/libstdc++.so.6"


def test_preload_retains_unrelated_libraries():
    assert runtime.preload_value("/new/libstdc++.so.6", "/opt/a.so /old/libstdc++.so.6:/opt/b.so") == "/new/libstdc++.so.6:/opt/a.so:/opt/b.so"
