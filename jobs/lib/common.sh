#!/usr/bin/env bash
# Shared helpers for Tri-Fair SLURM jobs.
# This file is sourced; do not submit it directly.

set -o pipefail

tf_log() {
  printf '[%s] %s\n' "$(date -Is)" "$*"
}

tf_warn() {
  printf '[%s] WARNING: %s\n' "$(date -Is)" "$*" >&2
}

tf_die() {
  printf '[%s] ERROR: %s\n' "$(date -Is)" "$*" >&2
  exit 1
}

tf_is_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|y|Y|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

tf_trim() {
  local value="${1:-}"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

tf_split_csv() {
  # Usage: tf_split_csv "a,b,c" output_array_name
  local raw="${1:-}"
  local output_name="$2"
  local -n output_ref="$output_name"
  local -a parts=()
  local part trimmed

  IFS=',' read -r -a parts <<< "$raw"
  output_ref=()
  for part in "${parts[@]}"; do
    trimmed="$(tf_trim "$part")"
    [[ -n "$trimmed" ]] && output_ref+=("$trimmed")
  done
  ((${#output_ref[@]} > 0)) || tf_die "CSV list is empty: '$raw'"
}

tf_contains() {
  local needle="$1"
  shift
  local value
  for value in "$@"; do
    [[ "$value" == "$needle" ]] && return 0
  done
  return 1
}

tf_validate_positive_int() {
  local name="$1"
  local value="$2"
  [[ "$value" =~ ^[0-9]+$ ]] || tf_die "$name must be an integer, got '$value'"
  ((value > 0)) || tf_die "$name must be positive, got '$value'"
}

tf_stage_checkpoints() {
  # Checkpoints are downstream/evaluation-model token counts.
  # 2M is supported as an optional intermediate stage, although the current
  # main plan is 1M -> 5M -> 7.5M.
  local budget="$1"
  case "$budget" in
    250000)  printf '250000' ;;
    500000)  printf '250000,500000' ;;
    750000)  printf '250000,500000,750000' ;;
    1000000) printf '250000,500000,750000,1000000' ;;
    2000000) printf '1250000,1500000,1750000,2000000' ;;
    5000000) printf '2000000,3000000,4000000,5000000' ;;
    7500000) printf '6000000,7000000,7500000' ;;
    *)       printf '%s' "$budget" ;;
  esac
}

tf_python_command() {
  local output_name="$1"
  local -n output_ref="$output_name"

  if [[ -n "${TF_PYTHON:-}" ]]; then
    [[ -x "$TF_PYTHON" ]] \
      || tf_die "TF_PYTHON is not executable: $TF_PYTHON"
    output_ref=("$TF_PYTHON")
  elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    output_ref=("$VIRTUAL_ENV/bin/python")
  elif command -v uv >/dev/null 2>&1; then
    output_ref=(uv run python)
  elif [[ -x ".venv/bin/python" ]]; then
    output_ref=(.venv/bin/python)
  elif command -v python3 >/dev/null 2>&1; then
    output_ref=(python3)
  elif command -v python >/dev/null 2>&1; then
    output_ref=(python)
  else
    tf_die "No usable Python interpreter found"
  fi
}

tf_repo_root() {
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    printf '%s' "$SLURM_SUBMIT_DIR"
  else
    local source_dir
    source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    printf '%s' "$source_dir"
  fi
}

tf_run_output_dir() {
  local root="$1" model="$2" dataset="$3" optimizer="$4" seed="$5"
  local namespace="${TF_RESULTS_NAMESPACE:-tri_fair}"
  printf '%s/results/%s/%s/%s/%s/seed%s' \
    "$root" "$namespace" "$model" "$dataset" "$optimizer" "$seed"
}

tf_read_logging_dir() {
  local output_dir="$1"
  local pointer="$output_dir/logging_dir.txt"
  [[ -s "$pointer" ]] || return 1
  local logging_dir
  logging_dir="$(<"$pointer")"
  logging_dir="$(tf_trim "$logging_dir")"
  [[ -n "$logging_dir" ]] || return 1
  if [[ "$logging_dir" != /* ]]; then
    logging_dir="$(cd "$(dirname "$pointer")" && realpath -m "$logging_dir")"
  fi
  printf '%s' "$logging_dir"
}

tf_write_status_json() {
  local path="$1"
  shift
  mkdir -p "$(dirname "$path")"
  local tmp="${path}.tmp.$$"
  {
    printf '{\n'
    local first=1 key value
    while (($# >= 2)); do
      key="$1"; value="$2"; shift 2
      ((first)) || printf ',\n'
      first=0
      # Values are intentionally serialized as strings to keep this helper
      # dependency-free and safe for job-status metadata.
      printf '  "%s": "%s"' \
        "${key//\"/\\\"}" \
        "${value//\"/\\\"}"
    done
    printf '\n}\n'
  } > "$tmp"
  mv "$tmp" "$path"
}
