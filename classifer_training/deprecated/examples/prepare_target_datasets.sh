#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/classifer_training/artifacts/datasets}"
DAPO_TRAIN_EXAMPLES="${DAPO_TRAIN_EXAMPLES:-5000}"
DAPO_VALIDATION_EXAMPLES="${DAPO_VALIDATION_EXAMPLES:-500}"
DAPO_TEST_EXAMPLES="${DAPO_TEST_EXAMPLES:-500}"

python -m classifer_training.prepare_datasets \
  --dataset_name deepscaler \
  --source auto \
  --output_root "$OUTPUT_ROOT"

python -m classifer_training.prepare_datasets \
  --dataset_name dapo_math_17k \
  --source auto \
  --hf_dataset_id open-r1/DAPO-Math-17k-Processed \
  --hf_splits train \
  --train_examples "$DAPO_TRAIN_EXAMPLES" \
  --validation_examples "$DAPO_VALIDATION_EXAMPLES" \
  --test_examples "$DAPO_TEST_EXAMPLES" \
  --output_root "$OUTPUT_ROOT"
