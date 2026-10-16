# Local → GitHub → Rocket

The attached Windows log already shows **69 passed** for the previous bundle under Python 3.12. The earlier Python 3.14 dependency problem is resolved. The clean package has a different installation layout, so run its tests once before publishing.

## 1. Create an empty GitHub repository

Create `Tri_Fair_V8` under `vikashmaheshwari97` at https://github.com/new. Choose private initially if you have not decided to release the research code. Leave README, `.gitignore`, and license initialization unchecked so GitHub creates an empty repository. This package already supplies its own README and ignore rules; choose an appropriate license separately.

The connected GitHub tool available during preparation had no repository-creation operation. This release therefore does not claim that the remote repository has been created. Creating the empty repository is the required manual GitHub/PyCharm step. The code package is ready to initialize and push.

## 2. Open and test the new local project

Extract into `C:\Users\maheshwari\PycharmProjects\Tri_Fair_V8_Clean`, open that folder in PyCharm, and run:

```powershell
cd C:\Users\maheshwari\PycharmProjects\Tri_Fair_V8_Clean
py -3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install -r requirements-cpu.txt
.\.venv312\Scripts\python.exe -m tri_fair.cli.doctor
.\.venv312\Scripts\python.exe -m pytest -q
```

Select `.venv312\Scripts\python.exe` in PyCharm. Keep the old project in its separate directory. The clean folder may have a different local name from the GitHub repository.

## 3. Publish this clean source

From the new project folder, after the empty GitHub repository exists and tests pass:

```powershell
git init -b main
git add .
git commit -m "Add clean Tri-Fair v8 research implementation"
git remote add origin https://github.com/vikashmaheshwari97/Tri_Fair_V8.git
git push -u origin main
```

These commands assume a new folder without Git metadata, as supplied in the ZIP. Use your normal Git Credential Manager/PyCharm sign-in. Do not add an access token to source files or paste one into the remote URL. If you created the repository with an initial commit, use the standard clone workflow instead of force-pushing over it. GitHub CI checks Python 3.11 and 3.12; those hosted jobs were not executed during local preparation.

## 4. Clone on Rocket and install a separate environment

Use your chosen project directory on Rocket. Keep existing experiment checkouts/results separate:

```bash
git clone https://github.com/vikashmaheshwari97/Tri_Fair_V8.git
cd Tri_Fair_V8
module --ignore_cache load python/3.11.14
python -m venv "$HOME/venvs/tri-fair-v8"
source "$HOME/venvs/tri-fair-v8/bin/activate"
python -m pip install -r requirements-hpc.txt
python -m tri_fair.cli.doctor
python -m pytest -q
```

Private repository cloning requires your normal authorized GitHub authentication. If your Rocket module setup differs, adapt the module commands. Do not copy the Windows virtual environment to Rocket. GPU validation runs on an allocated GPU node, not the login node. Review the SLURM GPU/memory/partition/account settings for your allocation before submission.

## 5. Stage the same pinned weights and prepare fresh manifests

Reuse your existing verified **model snapshots** and shared raw dataset caches through their absolute paths; no need to download identical weights again:

```bash
export QWEN_LOCAL_SNAPSHOT=/absolute/path/to/the/pinned/qwen/snapshot
export MISTRAL_LOCAL_SNAPSHOT=/absolute/path/to/the/pinned/mistral/snapshot
# Optional:
export GPT_OSS_LOCAL_SNAPSHOT=/absolute/path/to/the/pinned/gpt-oss/snapshot

python -m tri_fair.cli.prepare --datasets bbq,civil_comments,bias_in_bios \
  --seeds 72,73,74 --manifest-dir data/splits_v8_clean \
  --report data/splits_v8_clean/manifest_report.json
```

Replace the path placeholders. Preserve a shared Hugging Face dataset cache separately from model caches. Preparation may need a node with internet access when files are not already cached. Do not reuse old v7 manifests. The current support-aware split uses the larger Bios development profile.

## 6. Submit one GPU smoke job first

```bash
mkdir -p logs
sbatch jobs/smoke.sbatch
```

This loads Qwen by default and makes **two search-split predictions**, checking backend/parsing rather than benchmarking quality. To test Mistral:

```bash
MODEL=mistral-3-24b sbatch jobs/smoke.sbatch
```

Check both stdout and stderr and require successful completion. If the model fails to load, fix paths, backend compatibility or GPU resources before any training array. A CUDA availability check alone does not prove model compatibility.

## 7. Run a small optimizer check, then the matrix

After the backend smoke passes, submit a one-step optimizer debug job:

```bash
sbatch jobs/optimizer_smoke.sbatch
```

This uses two initial instructions, no additional warm starts, one challenger, at most one demonstration and a 1M-token ceiling. It deliberately permits an under-budget exit and writes to a separate `results/optimizer_smoke` directory. It checks optimizer/backend wiring; it is **not** a matched-budget benchmark. Inspect its successful completion and checkpoint before full experiments. The normal training submission enforces budget utilization.

Review the default full matrix without submitting:

```bash
bash jobs/submit_train.sh
```

Default: 2 models × 3 datasets × 6 methods × 3 seeds = 108 runs, with limited concurrency. Change `TF_MODELS`, `TF_DATASETS`, `TF_OPTIMIZERS`, `TF_SEEDS`, `BUDGET`, `TF_RESULTS_NAMESPACE`, `TF_VENV`, `NODELIST`, `PARTITION`, and resource settings deliberately. To submit only one full run first:

```bash
TF_MODELS=qwen-3-30b TF_DATASETS=bbq TF_OPTIMIZERS=Tri-Fair-v8 TF_SEEDS=72 \
  DRY_RUN=0 bash jobs/submit_train.sh
```

After its logs and actual budget are satisfactory, submit the complete matched matrix:

```bash
DRY_RUN=0 bash jobs/submit_train.sh
```

The main job retains the previously used default node through the submission helper; clear it using `NODELIST=''` or set an appropriate node. The default result namespace is `results/tri_fair_v8_clean_5m`. Existing v7 results/checkpoints remain separate.

## 8. Evaluate validation, freeze settings, then evaluate test

```bash
bash jobs/submit_evaluate.sh
DRY_RUN=0 DEPENDENCY=afterok:TRAINING_JOB_ID bash jobs/submit_evaluate.sh
```

The helper defaults to validation. Replace `TRAINING_JOB_ID`, and use the same model/dataset/seed/method lists as training. `V8_RESULTS_ROOT` defaults to the clean training namespace.

For manual evaluation of one model/dataset/seed:

```bash
python -m tri_fair.cli.evaluate --study-root results/tri_fair_v8_clean_5m/qwen-3-30b \
  --model qwen-3-30b --dataset bbq --random-seed 72 --manifest-dir data/splits_v8_clean \
  --optimizers Tri-Fair-v8,NSGAII-PO-Fair,SPEA2-PO-Fair,NSGAIII-PO-Fair,RandomSearch-PO-Fair,NSGAII-PO-Fair-Enhanced
```

Summarize explicit evaluation parquet files using `python -m tri_fair.reporting.summarize --inputs ... --split validation --output-dir results/validation_report --plot pdf`. This writes `validation_selections.json`. Freeze settings before final test. Evaluate test explicitly using `--evaluation-split test`, and report it with `--split test --selection-file results/validation_report/validation_selections.json`.

For manual evaluations, the default output is inside the model study root at `shared_union_validation/<model>/<dataset>/seed<seed>/eval_checkpoints_union.parquet` (or `shared_union_test` for final test). The evaluator requires at least two comparison methods and validates decoding/profile/cache identity.

## Do we need to submit jobs again?

**New v8 experiments require new jobs after these checks.** Existing v7 results cannot establish the performance of v8 under the new partition/profile. Cleanup is checked to preserve the v8 implementation's behavior on synthetic fixtures; it does not establish real-model equivalence or new performance. Keep previous results and do not relabel them as clean v8 results. There is no reason to resubmit every historical experiment or immediately launch the entire 108-run matrix.
