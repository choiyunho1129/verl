#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_NAME="${MODEL_NAME:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"

HIDDEN_ROOT="${HIDDEN_ROOT:-$REPO_ROOT/classifer_training/artifacts/hidden}"
INDEX_ROOT="${INDEX_ROOT:-$REPO_ROOT/classifer_training/artifacts/index}"
LABELS_ROOT="${LABELS_ROOT:-$REPO_ROOT/classifer_training/artifacts/labels}"
MANIFEST_PATH="${MANIFEST_PATH:-$REPO_ROOT/classifer_training/artifacts/manifests/deepscaler_deepseek_1p5b.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/classifer_training/artifacts/models/deepscaler_deepseek_1p5b_regression}"

python -m classifer_training.make_manifest \
  --model_name "$MODEL_NAME" \
  --datasets deepscaler \
  --hidden_root "$HIDDEN_ROOT" \
  --index_root "$INDEX_ROOT" \
  --labels_root "$LABELS_ROOT" \
  --index_filename index.jsonl \
  --output_path "$MANIFEST_PATH"

python -m classifer_training.train \
  --manifest "$MANIFEST_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --task_type regression \
  --target_field sampling_accuracy \
  --model ridge \
  --components hidden \
  --layers all \
  --component_pooling concat \
  --extra_features \
    label.aggregated_features.output_length_mean \
    label.aggregated_features.output_text_entropy_mean
