"""CPU-only plot preservation and finite-support checks."""

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def plots(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "research"))
    spec = importlib.util.spec_from_file_location(
        "v88_clean", ROOT / "research/replot_v88_clean.py"
    )
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_history_has_no_scatter_or_method_glyphs_and_no_extrapolation(plots):
    fig, ax = plt.subplots()
    plots.step_summary(
        ax,
        [(np.array([0.2, 0.7, 1.4]), np.array([0.15, 0.14, 0.13]))],
        color="black",
        marker=None,
        trajectory=True,
    )
    assert len(ax.lines) == 1
    assert ax.lines[0].get_marker() in (None, "None", "")
    np.testing.assert_allclose(ax.lines[0].get_xdata(), [0.2, 0.7, 1.4])
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [0.15, 0.14, 0.13])
    assert len(ax.collections) == 0
    plt.close(fig)


def test_three_seed_mean_uses_only_common_observed_support(plots):
    fig, ax = plt.subplots()
    curves = [
        (np.array([0.1, 0.5, 1.0]), np.array([3.0, 2.0, 1.0])),
        (np.array([0.2, 0.6, 1.2]), np.array([4.0, 3.0, 2.0])),
        (np.array([0.3, 0.7, 1.4]), np.array([5.0, 4.0, 3.0])),
    ]
    plots.step_summary(ax, curves, color="black", marker=None, trajectory=True)
    assert ax.lines[0].get_xdata().min() == 0.3 and ax.lines[0].get_xdata().max() == 1.0
    np.testing.assert_allclose(
        ax.lines[0].get_ydata(), [4.0, 11 / 3, 10 / 3, 3.0, 8 / 3]
    )
    assert len(ax.collections) == 1  # One SD band; no final-front scatter collection.
    plt.close(fig)


def fixture_csvs(tmp_path, plots):
    source = tmp_path / "source"
    source.mkdir()
    common = dict(
        dataset=["bbq"],
        optimizer=["Tri-Fair-v8.8"],
        seed=[72],
        model=["qwen-3-30b"],
        actual_budget_tokens=[1_900_000],
    )
    a = pd.DataFrame(dict(common, evaluation_split=["validation"]))
    h = pd.DataFrame(dict(common, chosen_step=[1]))
    f = pd.DataFrame(
        dict(
            common,
            evaluation_split=["validation"],
            prompt_id=["p"],
            validation_fairness_ready=[True],
        )
    )
    for name, frame in zip(plots.CSV_NAMES, (a, h, f)):
        frame.to_csv(source / name, index=False)
    return source, a, h, f


def test_loading_exports_preserves_source_bytes(tmp_path, plots):
    source, _, _, _ = fixture_csvs(tmp_path, plots)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    plots.load_exports(source, "qwen-3-30b", "validation", 2_000_000)
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}


@pytest.mark.parametrize(
    "failure",
    ["wrong_split", "wrong_model", "over_budget", "duplicate_final", "unready"],
)
def test_incompatible_csvs_fail_instead_of_silent_render(tmp_path, plots, failure):
    source, a, h, f = fixture_csvs(tmp_path, plots)
    if failure == "wrong_split":
        a["evaluation_split"] = "test"
    elif failure == "wrong_model":
        a["model"] = "mistral-3-24b"
    elif failure == "over_budget":
        a["actual_budget_tokens"] = 5_000_000
    elif failure == "duplicate_final":
        f = pd.concat([f, f], ignore_index=True)
    else:
        f["validation_fairness_ready"] = False
    for name, frame in zip(plots.CSV_NAMES, (a, h, f)):
        frame.to_csv(source / name, index=False)
    with pytest.raises(ValueError):
        plots.load_exports(source, "qwen-3-30b", "validation", 2_000_000)


def test_output_cannot_replace_original_exports(tmp_path, plots):
    source, _, _, _ = fixture_csvs(tmp_path, plots)
    with pytest.raises(ValueError, match="separate sibling"):
        plots.render(source, source)


def test_original_mocapo_method_colours_are_preserved(plots):
    assert [plots.STYLE[m][1] for m in plots.STYLE] == [
        "#000000",
        "#E69F00",
        "#785EF0",
        "#009E73",
        "#CC79A7",
    ]
