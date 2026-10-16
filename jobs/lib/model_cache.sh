#!/usr/bin/env bash
# Resolve complete local model snapshots for Rocket's offline compute nodes.
#
# Important: model snapshots and Hugging Face dataset files may live in
# different cache roots.  Qwen/GPT-OSS are passed to vLLM by absolute local
# snapshot path, while hf_hub_download must continue to use the shared dataset
# cache (normally ~/.cache/huggingface/hub).

tf_configure_hf_snapshot() {
  local model="$1"
  local qwen_revision="0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
  local gptoss_revision="b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
  local candidate=""

  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

  # Keep dataset downloads in the normal shared Hugging Face cache.  Do not
  # point HF_HUB_CACHE at a model-specific directory: doing so hides cached
  # datasets such as heegyu/bbq from hf_hub_download.
  local shared_hf_home="${TF_HF_HOME:-$HOME/.cache/huggingface}"
  local shared_hf_hub_cache="${TF_HF_HUB_CACHE:-$shared_hf_home/hub}"
  export HF_HOME="$shared_hf_home"
  export HF_HUB_CACHE="$shared_hf_hub_cache"
  export HUGGINGFACE_HUB_CACHE="$shared_hf_hub_cache"
  export HF_DATASETS_CACHE="${TF_HF_DATASETS_CACHE:-$shared_hf_home/datasets}"

  case "$model" in
    qwen-3-30b)
      local qwen_cache="${QWEN_HF_CACHE:-$HOME/projects/models/Qwen3-30B}"
      local -a qwen_candidates=(
        "${QWEN_LOCAL_SNAPSHOT:-}"
        "$qwen_cache/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/$qwen_revision"
        "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/$qwen_revision"
      )
      for candidate in "${qwen_candidates[@]}"; do
        [[ -n "$candidate" ]] || continue
        if [[ -f "$candidate/config.json" ]] \
           && [[ -f "$candidate/tokenizer.json" || -f "$candidate/tokenizer_config.json" ]] \
           && { [[ -f "$candidate/model.safetensors.index.json" ]] \
                || compgen -G "$candidate/*.safetensors" >/dev/null; }; then
          export QWEN_LOCAL_SNAPSHOT="$candidate"
          tf_log "Using Qwen local snapshot: $QWEN_LOCAL_SNAPSHOT"
          tf_log "Using Hugging Face hub cache: $HF_HUB_CACHE"
          tf_log "Using Hugging Face datasets cache: $HF_DATASETS_CACHE"
          return 0
        fi
      done
      tf_die "No complete Qwen snapshot found. Set QWEN_LOCAL_SNAPSHOT explicitly."
      ;;

    gpt-oss-120b)
      local -a gptoss_candidates=(
        "${GPT_OSS_LOCAL_SNAPSHOT:-}"
        "$HOME/models/gpt-oss-120b/models--openai--gpt-oss-120b/snapshots/$gptoss_revision"
        "$HOME/projects/models/gpt-oss-120b/models--openai--gpt-oss-120b/snapshots/$gptoss_revision"
        "$HF_HUB_CACHE/models--openai--gpt-oss-120b/snapshots/$gptoss_revision"
      )
      for candidate in "${gptoss_candidates[@]}"; do
        [[ -n "$candidate" ]] || continue
        if [[ -f "$candidate/config.json" ]] \
           && [[ -f "$candidate/tokenizer.json" || -f "$candidate/tokenizer_config.json" ]] \
           && [[ -f "$candidate/model.safetensors.index.json" ]]; then
          export GPT_OSS_LOCAL_SNAPSHOT="$candidate"
          tf_log "Using GPT-OSS local snapshot: $GPT_OSS_LOCAL_SNAPSHOT"
          tf_log "Using Hugging Face hub cache: $HF_HUB_CACHE"
          tf_log "Using Hugging Face datasets cache: $HF_DATASETS_CACHE"
          return 0
        fi
      done
      tf_die "No complete GPT-OSS snapshot found. Set GPT_OSS_LOCAL_SNAPSHOT explicitly."
      ;;

    *)
      tf_die "No model-cache rule for '$model'"
      ;;
  esac
}

#!/usr/bin/env bash
# The v8 matrix supports all three registered local model families.

tf_configure_model_cache() {
  local model="$1"
  if [[ "$model" != mistral-3-24b ]]; then
    tf_configure_hf_snapshot "$model"
    return
  fi
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  export HF_HOME="${TF_HF_HOME:-$HOME/.cache/huggingface}"
  export HF_HUB_CACHE="${TF_HF_HUB_CACHE:-$HF_HOME/hub}"
  export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
  export HF_DATASETS_CACHE="${TF_HF_DATASETS_CACHE:-$HF_HOME/datasets}"
  local snapshot="${MISTRAL_LOCAL_SNAPSHOT:-$PWD/../models/Mistral-3-24b}"
  local file
  for file in config.json params.json tekken.json; do
    [[ -s "$snapshot/$file" ]] || tf_die "Mistral snapshot missing $file: $snapshot"
  done
  compgen -G "$snapshot/consolidated*.safetensors" >/dev/null \
    || tf_die "Mistral snapshot missing consolidated weights: $snapshot"
  export MISTRAL_LOCAL_SNAPSHOT="$snapshot"
  tf_log "Using Mistral local snapshot: $MISTRAL_LOCAL_SNAPSHOT"
}
