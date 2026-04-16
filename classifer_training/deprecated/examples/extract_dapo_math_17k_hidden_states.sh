#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/classifer_training/artifacts/datasets}"
HIDDEN_ROOT="${HIDDEN_ROOT:-$REPO_ROOT/classifer_training/artifacts/hidden}"
INDEX_ROOT="${INDEX_ROOT:-$REPO_ROOT/classifer_training/artifacts/index}"

python -m classifer_training.extract_hidden_states \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --input_path "$DATASET_ROOT/dapo_math_17k" \
  --dataset_name dapo_math_17k \
  --components hidden \
  --hidden_root "$HIDDEN_ROOT" \
  --index_root "$INDEX_ROOT"
