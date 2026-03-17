#!/usr/bin/env bash

set -euo pipefail

cd /workspace

config_path="${SEMANTIC_MEMORY_CONFIG:-/workspace/sample_data/demo_local_federated_memory_config.json}"
model_id="${SEMANTIC_MODEL_ID:-sentence-transformers/all-MiniLM-L6-v2}"
layer_index="${SEMANTIC_LAYER_INDEX:--1}"
pooling="${SEMANTIC_POOLING:-mean}"
max_length="${SEMANTIC_MAX_LENGTH:-128}"
warmup_mode="${SEMANTIC_PROVIDER_WARMUP:-off}"

cmd=(
  python
  scripts/semantic_memory_mcp_server.py
  --federated-config "$config_path"
  --semantic-model-id "$model_id"
  --semantic-layer-index "$layer_index"
  --semantic-pooling "$pooling"
  --semantic-max-length "$max_length"
  --semantic-provider-warmup "$warmup_mode"
)

if [[ -n "${SEMANTIC_DEVICE:-}" ]]; then
  cmd+=(--semantic-device "$SEMANTIC_DEVICE")
fi

if [[ -n "${DEFAULT_ACTIVE_STORES:-}" ]]; then
  IFS=',' read -r -a active_store_ids <<< "$DEFAULT_ACTIVE_STORES"
  for store_id in "${active_store_ids[@]}"; do
    trimmed="$(printf '%s' "$store_id" | xargs)"
    if [[ -n "$trimmed" ]]; then
      cmd+=(--default-active-store "$trimmed")
    fi
  done
fi

exec "${cmd[@]}"