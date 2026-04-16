#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_NAME="${MODEL_NAME:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
MODEL_SLUG="${MODEL_SLUG:-deepseek-ai_DeepSeek-R1-Distill-Qwen-1_5B}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/classifer_training/artifacts/datasets/deepscaler}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/classifer_training/artifacts/runs/deepscaler/${MODEL_SLUG}}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-1.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
BATCH_SIZE="${BATCH_SIZE:-64}"
BACKEND="${BACKEND:-vllm}"
if [ "$#" -gt 0 ]; then
  SEEDS=("$@")
else
  SEEDS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
fi

mkdir -p "${RUN_ROOT}"

for SEED in "${SEEDS[@]}"; do
  python -m classifer_training.sample \
    --model_name_or_path "${MODEL_NAME}" \
    --input_path "${DATASET_DIR}" \
    --dataset_name deepscaler \
    --output_dir "${RUN_ROOT}/temp${TEMPERATURE}_seed${SEED}" \
    --backend "${BACKEND}" \
    --grader math_verify \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --batch_size "${BATCH_SIZE}" \
    --seed "${SEED}" \
    --trust_remote_code \
    --overwrite
done
