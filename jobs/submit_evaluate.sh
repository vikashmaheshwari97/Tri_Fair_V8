#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source jobs/lib/common.sh
declare -a models datasets seeds
tf_split_csv "${TF_MODELS:-qwen-3-30b,mistral-3-24b,gpt-oss-120b}" models
tf_split_csv "${TF_DATASETS:-bbq,civil_comments,bias_in_bios}" datasets
tf_split_csv "${TF_SEEDS:-72,73,74}" seeds
total=$((${#models[@]} * ${#datasets[@]} * ${#seeds[@]}))
mkdir -p logs
command=(sbatch "--gpus-per-task=${GPU_TYPE:-h200-141g}:1" "--cpus-per-task=${CPUS_PER_TASK:-32}" --parsable "--array=0-$((total-1))%${MAX_CONCURRENT:-2}")
[[ -z "${PARTITION:-}" ]] || command+=("--partition=$PARTITION")
[[ -z "${NODELIST:-}" ]] || command+=("--nodelist=$NODELIST")
[[ -z "${DEPENDENCY:-}" ]] || command+=("--dependency=$DEPENDENCY")
command+=(jobs/evaluate.sbatch)
if tf_is_true "${DRY_RUN:-1}"; then
  printf 'DRY RUN:';printf ' %q' "${command[@]}";printf '\n'
else
  "${command[@]}"
fi
