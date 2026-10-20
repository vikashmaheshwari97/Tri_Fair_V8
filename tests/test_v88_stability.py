"""CPU regressions for the failure path; these do not validate GPU kernels."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tri_fair.runtime.inference_profile import (
    bootstrap_batch_invariance,
    require_matching_profile,
)

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "v88_stability", ROOT / "recovery/v88_bios_stability.py"
)
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


@pytest.fixture(autouse=True)
def isolated_profile(monkeypatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")


def test_cli_bootstrap_sets_flag_before_first_inference_import(tmp_path):
    # Run the actual train/evaluate modules; stop at their first Promptolution
    # import and assert the environment already agrees with the CLI.
    script = """
import importlib.abc, os, runpy, sys
class Stop(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "promptolution":
            assert os.environ.get("VLLM_BATCH_INVARIANT") == "1"
            print("PROFILE_READY_BEFORE_INFERENCE_IMPORT")
            raise SystemExit(0)
sys.meta_path.insert(0, Stop())
module = sys.argv[1]
sys.argv = [module, "--batch-invariant"]
runpy.run_module(module, run_name="__main__")
"""
    env = dict(os.environ)
    env.pop("VLLM_BATCH_INVARIANT", None)
    for module in ("tri_fair.cli.train", "tri_fair.cli.evaluate"):
        result = subprocess.run(
            [sys.executable, "-c", script, module],
            env=env,
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0, result.stderr
        assert "PROFILE_READY_BEFORE_INFERENCE_IMPORT" in result.stdout


def test_profile_cannot_be_enabled_late_or_misreported(monkeypatch):
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace())
    with pytest.raises(RuntimeError, match="after vLLM import"):
        bootstrap_batch_invariance(["--batch-invariant"])
    with pytest.raises(RuntimeError, match="profile mismatch"):
        require_matching_profile(True)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    require_matching_profile(True)
    with pytest.raises(RuntimeError, match="profile mismatch"):
        require_matching_profile(False)


@pytest.mark.parametrize(
    "bad",
    [
        "nurse",
        "",
        "<final_answer>invalid</final_answer>",
        "<final_answer>nurse</final_answer><final_answer>teacher</final_answer>",
    ],
)
def test_raw_parser_rejects_instead_of_substituting_first_class(bad):
    with pytest.raises(ValueError, match="Invalid raw"):
        repair.strict_labels([bad], ["nurse", "teacher"])


def test_all_five_pilot_commands_forward_same_profile(tmp_path):
    from tri_fair.cli.pilot import train_command

    args = SimpleNamespace(
        model="qwen-3-30b",
        dataset="bias_in_bios",
        seed=72,
        budget=2000000,
        manifest_dir=str(tmp_path),
        batch_invariant=True,
    )
    for method in repair.METHODS:
        cmd = train_command(args, method, tmp_path)
        assert cmd.count("--batch-invariant") == 1
        assert cmd[cmd.index("--budget-per-run") + 1] == "2000000"


def fake_probe(monkeypatch, tmp_path, behavior="stable"):
    from tests.fakes import FakeLLM, task_and_pool
    from tri_fair.data import loading, partitions
    from tri_fair.runtime import decoding, models
    from tri_fair.config.datasets import ALL_DATASETS
    from tri_fair.search.rendering import rendered_prompt, output_contract
    from promptolution.utils.prompt import Prompt
    import hashlib
    import random

    task, pool = task_and_pool()
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    manifest = manifests / "bias_in_bios_seed72.json"
    manifest.write_text("{}")
    task.manifest_path = str(manifest)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    monkeypatch.setattr(loading, "create_dev_tasks", lambda *a, **k: (task, pool))
    monkeypatch.setattr(partitions, "split_search_validation", lambda t: (t, t))
    monkeypatch.setattr(models, "_resolve_qwen_snapshot", lambda *a, **k: snapshot)
    monkeypatch.setattr(decoding, "configure_structured_labels", lambda *a: None)
    versions = {x: "fake" for x in repair.PACKAGES}
    versions.update(vllm="0.24.0", promptolution="2.2.3")
    monkeypatch.setattr(repair, "packages", lambda: versions)

    class Engine(FakeLLM):
        def get_response(self, prompts, system_prompts=None):
            outputs = []
            for text in prompts:
                label = (
                    "teacher" if behavior == "batch" and len(prompts) > 1 else "nurse"
                )
                raw = (
                    "invalid"
                    if behavior == "invalid"
                    else f"<final_answer>{label}</final_answer>"
                )
                self.input_token_count += len(self.tokenizer.encode(text))
                self.output_token_count += len(self.tokenizer.encode(raw))
                outputs.append(raw)
            return outputs

    monkeypatch.setattr(models, "create_llm", lambda *a, **k: Engine())
    text = random.Random(72).sample(
        list(ALL_DATASETS["bias_in_bios"].initial_prompts), 4
    )[0]
    prompt = rendered_prompt(
        output_contract(text), [], template=Prompt(text).downstream_template
    )
    original = tmp_path / "original.json"
    original.write_text(
        json.dumps(
            dict(
                instruction_sha256=hashlib.sha256(
                    prompt.construct_prompt().encode()
                ).hexdigest(),
                predictions=[
                    dict(example_id=str(task.ids[i]))
                    for i in np.linspace(0, len(task.xs) - 1, 8, dtype=int)
                ],
            )
        )
    )
    args = SimpleNamespace(
        output_root=str(tmp_path / "probe"),
        manifest_dir=str(manifests),
        original_probe=str(original),
    )
    return args


def test_real_probe_gate_completes_all_rounds_on_stable_fake(monkeypatch, tmp_path):
    args = fake_probe(monkeypatch, tmp_path)
    assert repair.probe(args, ROOT) == 0
    proof_path = Path(args.output_root) / "stability_proof.json"
    proof = repair.validate_proof(proof_path, ROOT)
    assert proof["prediction_count"] == proof["raw_output_count"] == 192
    assert 0 < proof["actual_downstream_tokens"]["total_tokens"] < 100000
    Path(proof["manifest_path"]).write_text('{"changed":true}')
    with pytest.raises(ValueError, match="manifest changed"):
        repair.validate_proof(proof_path, ROOT)


@pytest.mark.parametrize("behavior", ["batch", "invalid"])
def test_actual_failing_probe_is_saved_and_never_accepted(
    monkeypatch, tmp_path, behavior
):
    args = fake_probe(monkeypatch, tmp_path, behavior)
    assert repair.probe(args, ROOT) == 1
    proof = Path(args.output_root) / "stability_proof.json"
    saved = repair.read(proof)
    assert saved["status"] == "failed" and saved["error"]
    with pytest.raises(ValueError, match="not a complete passing"):
        repair.validate_proof(proof, ROOT)
    if behavior == "batch":
        gate = repair.read(
            Path(args.output_root) / "prompt0_repeat0/repeatability_probe.json"
        )
        assert gate["mismatches"] and gate["status"] == "failed"


def test_probe_proof_rejects_different_source_or_incomplete_evidence(
    monkeypatch, tmp_path
):
    args = fake_probe(monkeypatch, tmp_path)
    assert repair.probe(args, ROOT) == 0
    path = Path(args.output_root) / "stability_proof.json"
    saved = repair.read(path)
    monkeypatch.setattr(repair, "source_hash", lambda repo: "changed")
    with pytest.raises(ValueError, match="Source changed"):
        repair.validate_proof(path, ROOT)
    saved["checks"].pop()
    repair.write(path, saved)
    monkeypatch.setattr(repair, "source_hash", lambda repo: saved["source_sha256"])
    with pytest.raises(ValueError, match="details are incomplete"):
        repair.validate_proof(path, ROOT)
