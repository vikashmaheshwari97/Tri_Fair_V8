# Tri_Fair_V8 — clean research project

This is the clean, installable **8.1** refactor of the v8 experimental implementation. It supports three fairness datasets and the current optimizer/comparison methods. There are no historical optimizer modules, archived result files, old split manifests, old experiment entry points or version-specific figure scripts.

The public method name remains `Tri-Fair-v8`. Package version 8.1 identifies the architecture and checkpoint change; it is not a claim of new measured performance. The old and refactored optimizers were compared using deterministic synthetic models; see `validation/behavior_parity.json`. Actual GPU/model experiments remain necessary.

Start with [the local → GitHub → Rocket guide](docs/ROLLOUT.md). For the component design, see [architecture](docs/ARCHITECTURE.md). For experimental interpretation, see [protocol](docs/PROTOCOL.md).

## Windows / PyCharm

Extract into a **new** `Tri_Fair_V8_Clean` directory. Do not overlay the previous 268-file bundle: extracting over it leaves obsolete files behind. Open the new folder as a PyCharm project.

Use **Python 3.12**, not Python 3.13/3.14. Python 3.11 is also allowed by package metadata and the supplied Rocket templates; local validation used 3.12.14. Promptolution 2.2.3 requires Python below 3.13.

From the project root in PowerShell:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install -r requirements-cpu.txt
.\.venv312\Scripts\python.exe -m tri_fair.cli.doctor
.\.venv312\Scripts\python.exe -m pytest -q
```

Set PyCharm's interpreter to `.venv312\Scripts\python.exe`. The displayed `(.venv)` terminal prompt does not prove which interpreter a command uses; explicit paths above avoid that ambiguity. `setup_windows.ps1` automates these checks if your PowerShell policy allows local scripts. It does not delete existing environments.

Installation uses `pyproject.toml` and an editable `tri_fair` package. The code does not depend on a sibling `Tri_Fair` checkout or the old `src` package. On Linux, use `python3.12 -m venv .venv`, activate it, then the same pip/doctor/pytest commands with `python`.

## Supported methods

| CLI name | Search / selection |
|---|---|
| `Tri-Fair-v8` | Readiness-gated progressive racing; pooled objectives; adaptive modes; reflection/demo operators; paired rejection guard; bounded reference/HV archive |
| `NSGAII-PO-Fair` | Full search evaluation; objective-aware proposals; NSGA-II survival |
| `SPEA2-PO-Fair` | Full search evaluation; strength/raw-fitness/density selection |
| `NSGAIII-PO-Fair` | Full search evaluation; pymoo reference-direction survival |
| `RandomSearch-PO-Fair` | Independent LLM proposals from frozen initial seeds; random demonstrations; bounded NSGA-II archive |
| `NSGAII-PO-Fair-Enhanced` | Full search evaluation; additional reflection/demo operators; fixed mode mixture; NSGA-II survival |

Additional component ablations are `Tri-Fair-v8-NoFeedback`, `Tri-Fair-v8-NoDemo`, `Tri-Fair-v8-NoUncertainty`, and `Tri-Fair-v8-FullEval`. They disable the additional reflection step, additional demo-edit step, paired bootstrap guard, and racing, respectively. Inherited diagnostic variation/few-shot operations/readiness guards remain where applicable. SPEA2/NSGA-III are discrete prompt-search adaptations; random search is not uniform sampling over language. No GEPA or DSPy reproduction is claimed.

Models: `qwen-3-30b`, `mistral-3-24b`, optionally `gpt-oss-120b`. Datasets: `bbq`, `civil_comments`, `bias_in_bios`. Model and dataset revisions are pinned in `config/models.py` and `config/datasets.py`. Weights and raw datasets are external downloads; no experiments or performance numbers are fabricated or bundled.

## CLI entry points

| Command | Purpose |
|---|---|
| `python -m tri_fair.cli.doctor` | CPU interpreter/package check |
| `python -m tri_fair.cli.doctor --gpu` | GPU/backend import and CUDA availability check; no weights loaded |
| `python -m tri_fair.cli.prepare` | Prepare immutable manifests and check partition support |
| `python -m tri_fair.cli.smoke` | Two real search-split predictions; backend/parsing check |
| `python -m tri_fair.cli.train` | One optimizer run |
| `python -m tri_fair.cli.matrix` | Print or execute the experiment matrix |
| `python -m tri_fair.cli.evaluate` | Shared unique-prompt validation/test evaluation |
| `python -m tri_fair.reporting.summarize` | Freeze validation selections; summarize per seed; export figures |

The default matrix is 2 models × 3 datasets × 6 methods × 3 seeds = **108 runs**. `python -m tri_fair.cli.matrix --task-index 0` prints one command without executing it. Add `--write-plan results/plan.json` to save the full plan. Add `--execute` only when you intend to run a configured GPU workload. CPU tests use fake models and make no model-service calls.

## Files and reproducibility

- `src/tri_fair/optimizers`: current optimizer, shared Pareto engine and comparison methods.
- `src/tri_fair/search`: objectives, initialization, racing, archive, variation, feedback and resampling.
- `src/tri_fair/data`: loaders, manifests, fairness task and partition rules.
- `src/tri_fair/fairness`: metric definitions, support sampling and interval calculations.
- `src/tri_fair/runtime`: budget admission, checkpoints, model creation and decoding.
- `src/tri_fair/config`: one dataset/model/study registry.
- `src/tri_fair/cli` and `reporting`: canonical runs and result analysis.
- `jobs`: current training/evaluation/smoke SLURM templates and submission helpers.
- `configs/sensitivity`: one-factor sensitivity settings.
- `tests`: one CPU test suite.

Checkpoints use schema 5 and current class/state names. Start a new run namespace: old v7/v8-bundle pickle checkpoints cannot be resumed into this refactor. Prediction caches, RNG state, token counts, adaptive mode credit and survival state are checkpointed. Exact GPU replay remains backend-dependent. Only load trusted pickle checkpoints.

`RELEASE_MANIFEST.json` records provenance and the file inventory. `RELEASE_SHA256SUMS.txt` checks all release files except itself. Verify with `python scripts/verify_release.py`. The ZIP does not include `.git`; initialize Git before publishing. Build caches, virtual environments, raw data, model weights and generated results are ignored by Git.

See [NOTICE.md](NOTICE.md) for code provenance. This project preserves the v8 search design while consolidating its implementation; cleanup alone does not establish better accuracy, fairness or cost.
