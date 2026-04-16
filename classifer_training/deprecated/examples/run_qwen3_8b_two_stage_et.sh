#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MANIFEST_PATH="${MANIFEST_PATH:-${REPO_ROOT}/classifer_training/artifacts/manifests/dapo_math_17k_qwen3_8b.json}"
LABELS_PATH="${LABELS_PATH:-${REPO_ROOT}/classifer_training/artifacts/labels/dapo_math_17k/qwen3_8b/sampling_labels_enriched_mathverify_fulltext.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/classifer_training/artifacts/models/dapo_math_17k_qwen3_8b_two_stage_et}"
N_ESTIMATORS="${N_ESTIMATORS:-2000}"
MIN_SAMPLES_LEAF="${MIN_SAMPLES_LEAF:-5}"
MAX_FEATURES="${MAX_FEATURES:-0.5}"

MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp}"

python -m classifer_training.train_prompt_et \
  --manifest "${MANIFEST_PATH}" \
  --labels_path "${LABELS_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --mode two_stage \
  --n_estimators "${N_ESTIMATORS}" \
  --min_samples_leaf "${MIN_SAMPLES_LEAF}" \
  --max_features "${MAX_FEATURES}"
