# Tri-Fair V8 Clean: Firefly GPU and GPT-OSS-120B hotfix

Applies to the clean package version 8.1.0. Includes the prior UTF-8 fix.

## What failed

Job 2491099 received a Tesla V100-PCIE-16GB (compute capability 7.0).
The installed PyTorch 2.11.0+cu130 build lists supported architectures starting
at 7.5. Its CUDA kernels failed before model loading completed. Generic GPU
requests allowed this allocation; CUDA availability alone did not detect it.
The dataset manifests and successful CPU tests do not need to be repeated to
repair this hardware allocation issue.

## Changes

- All four batch scripts request one `h200-141g` GPU per task, 32 CPU cores,
  and whole cores. This selects Firefly H200 nodes by GPU type, without
  pinning the job to a single host. Existing explicit Slurm command options
  override script defaults; avoid generic GPU overrides.
- Both submission wrappers request H200 GPUs. The training wrapper no longer
  defaults to pegasus2. Optional `NODELIST` still constrains placement.
- Training, evaluation and Python CLI matrices include `gpt-oss-120b` alongside
  `qwen-3-30b` and `mistral-3-24b`: 162 training tasks and 27 evaluation tasks
  with the six default methods, three datasets and seeds 72,73,74.
- Each job prints the allocation and GPU details, then checks GPU capability,
  total device memory and actual CUDA kernel execution before loading weights.
  This model fleet requires capability >=8.0 and >=70 GiB per visible device
  in this preflight. Passing is a prerequisite, not proof of model compatibility
  or available KV-cache capacity.
- GPT-OSS snapshot resolution also checks the shared Hugging Face hub cache.
  Existing GPT-OSS model registration, pinned revision and wrapper are reused.
- No dependency version or research optimizer behavior is changed.

## Apply locally in PyCharm PowerShell

Save the ZIP in Downloads. If you have edited any replaced files, back them up
first. Extract at the root containing pyproject.toml, src and jobs:

```powershell
cd C:\Users\maheshwari\PycharmProjects\Tri_Fair_V8_Clean
Expand-Archive -LiteralPath "$HOME\Downloads\Tri_Fair_V8_Clean_Firefly_GPTOSS_Hotfix.zip" -DestinationPath . -Force
.\.venv312\Scripts\python.exe -m pytest -q
git diff --stat
```

Once checked, commit and push the patch to your existing Tri_Fair_V8 remote.
The ZIP does not include model weights, datasets, virtual environments or Git
metadata. The code version remains 8.1.0 for the existing editable installation.

## Apply on Rocket after pushing from your local checkout

```bash
cd ~/Tri_Fair_V8
git pull --ff-only
source "$HOME/venvs/tri-fair-v8/bin/activate"
mkdir -p logs
sinfo -N -p gpu -o '%N %G'
for model in qwen-3-30b mistral-3-24b gpt-oss-120b; do
  sbatch --export=ALL,MODEL="$model",DATASET=bbq,SEED=72 jobs/smoke.sbatch
done
```

Run model inference only through a Slurm allocation, not on a login node.
These submissions each request one H200. The exact Firefly node is assigned
by Slurm and printed in the job log. Requesting H200 can increase queue time.

Alternatively transfer this ZIP to Rocket and extract it directly into
~/Tri_Fair_V8 instead of doing the Git push/pull step. Apply the same patch
locally afterwards so both checkouts agree.

If the GPT-OSS snapshot is elsewhere, set this before sbatch (substitute the
actual directory containing config.json and all referenced weight shards):

```bash
export GPT_OSS_LOCAL_SNAPSHOT=/absolute/path/to/complete/snapshot
```

The existing pinned revision is b5c939de8f754692c1647ca79fbf85e8c1e70f8a.
Compute jobs default to offline mode; download missing weights separately on
an internet-enabled login node using the existing project model-cache setup.
This patch does not silently download weights or change revisions.

Wait for each model smoke job to finish and verify its Slurm state/exit code
and parsing_ok result. Then submit an optimizer smoke test for each model:

```bash
for model in qwen-3-30b mistral-3-24b gpt-oss-120b; do
  sbatch --export=ALL,MODEL="$model",DATASET=bbq,SEED=72 jobs/optimizer_smoke.sbatch
done
```

After successful smoke tests, preview the full training matrix:

```bash
unset TF_MODELS
unset NODELIST
DRY_RUN=1 bash jobs/submit_train.sh
```

Expected preview: models include all three; array `0-161%2`; GPU type
`h200-141g:1`. Existing TF_MODELS explicitly overrides defaults, hence unset it
when adopting the three-model default. Old NODELIST=pegasus2 would conflict
with the H200 request, hence unset it too.

To submit the full study once all smoke tests pass:

```bash
DRY_RUN=0 bash jobs/submit_train.sh
```

Do not resubmit old successful research experiments merely to apply a scheduler
fix. The failed smoke job needs a new submission. Keep existing split manifests.
The evaluation wrapper remains a dry run by default; submit evaluation only
after the corresponding training runs are ready.

## Validation and limits

CPU regression tests exercise V100 rejection, A100 40 GB rejection, A100 80 GB
and H200 acceptance, no visible GPUs, and CUDA kernel failures. Shell syntax
and submission dry runs are checked. Actual H200 inference and GPT-OSS model
loading must be validated on Rocket; no GPU is available in the patch-building
environment. Earlier release validation reports remain historical.

Sources checked 16 October 2026:

- https://docs.hpc.ut.ee/public/cluster/Running_jobs/gpu_computing/
- https://github.com/vllm-project/recipes/blob/main/OpenAI/GPT-OSS.md
