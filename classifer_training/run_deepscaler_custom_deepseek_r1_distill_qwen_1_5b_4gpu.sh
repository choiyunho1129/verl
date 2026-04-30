#!/usr/bin/env bash
# Custom DeepScaleR pipeline for DeepSeek-R1-Distill-Qwen-1.5B.
# Defaults: train 4096 x2, validation 1024 x16, layer 19, prompt/think-end last10.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

usage() {
  cat <<'EOF'
Usage:
  bash classifer_training/run_deepscaler_custom_deepseek_r1_distill_qwen_1_5b_4gpu.sh \
    --gpu-ids 0 \
    --python "$(which python)"

Defaults:
  - model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  - dataset: local data/deepscaler train/valid parquet
  - train prompts: 4096, train rollouts/prompt: 2
  - validation prompts: 1024, validation rollouts/prompt: 16
  - sampling: temperature=1, top_p=1, top_k=-1
  - hidden extraction: layer 19, prompt last-10 mean, think-end last-10 hidden

All options supported by run_deepscaler_custom_qwen3_4b_base_4gpu.sh are forwarded.
EOF
}

for arg in "$@"; do
  case "$arg" in
    -h|--help)
      usage
      exit 0
      ;;
  esac
done

MODEL_NAME="${MODEL_NAME:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
MODEL_SLUG="${MODEL_SLUG:-deepseek-ai_DeepSeek-R1-Distill-Qwen-1_5B}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3}"
GENERATION_PARALLELISM="${GENERATION_PARALLELISM:-tp}"
TRAIN_PROMPTS="${TRAIN_PROMPTS:-4096}"
VALIDATION_PROMPTS="${VALIDATION_PROMPTS:-1024}"
TRAIN_NUM_SAMPLES="${TRAIN_NUM_SAMPLES:-2}"
VALIDATION_NUM_SAMPLES="${VALIDATION_NUM_SAMPLES:-16}"
TRAIN_GENERATION_SHARD_SIZE="${TRAIN_GENERATION_SHARD_SIZE:-512}"
VALIDATION_GENERATION_SHARD_SIZE="${VALIDATION_GENERATION_SHARD_SIZE:-256}"
TEMPERATURE="${TEMPERATURE:-1}"
TOP_P="${TOP_P:-1}"
TOP_K="${TOP_K:--1}"
SEED="${SEED:-1}"
LAYERS="${LAYERS:-19}"
PROMPT_LAST_N_VALUES="${PROMPT_LAST_N_VALUES:-10}"
ROLLOUT_COMPONENTS="${ROLLOUT_COMPONENTS:-think_end_last10_hidden}"
PROMPT_BATCH_SIZE="${PROMPT_BATCH_SIZE:-8}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-1}"
ROLLOUT_MAX_BATCH_TOKENS="${ROLLOUT_MAX_BATCH_TOKENS:-12000}"

args=("$@")
for ((idx=0; idx<${#args[@]}; idx++)); do
  case "${args[$idx]}" in
    --gpu-ids) idx=$((idx + 1)); GPU_IDS_CSV="${args[$idx]}" ;;
    --generation-parallelism) idx=$((idx + 1)); GENERATION_PARALLELISM="${args[$idx]}" ;;
    --train-prompts) idx=$((idx + 1)); TRAIN_PROMPTS="${args[$idx]}" ;;
    --validation-prompts) idx=$((idx + 1)); VALIDATION_PROMPTS="${args[$idx]}" ;;
    --train-num-samples) idx=$((idx + 1)); TRAIN_NUM_SAMPLES="${args[$idx]}" ;;
    --validation-num-samples) idx=$((idx + 1)); VALIDATION_NUM_SAMPLES="${args[$idx]}" ;;
    --num-samples) idx=$((idx + 1)); TRAIN_NUM_SAMPLES="${args[$idx]}"; VALIDATION_NUM_SAMPLES="${args[$idx]}" ;;
    --temperature) idx=$((idx + 1)); TEMPERATURE="${args[$idx]}" ;;
    --top-p) idx=$((idx + 1)); TOP_P="${args[$idx]}" ;;
    --top-k) idx=$((idx + 1)); TOP_K="${args[$idx]}" ;;
    --seed) idx=$((idx + 1)); SEED="${args[$idx]}" ;;
    --layers) idx=$((idx + 1)); LAYERS="${args[$idx]}" ;;
    --prompt-last-n-values) idx=$((idx + 1)); PROMPT_LAST_N_VALUES="${args[$idx]}" ;;
    --rollout-components) idx=$((idx + 1)); ROLLOUT_COMPONENTS="${args[$idx]}" ;;
  esac
done

if [[ -z "${TP_SIZE:-}" ]]; then
  if [[ "$GENERATION_PARALLELISM" == "shard" ]]; then
    TP_SIZE=1
  else
    gpu_words="${GPU_IDS_CSV//,/ }"
    read -r -a parsed_gpu_ids <<< "$gpu_words"
    TP_SIZE="${#parsed_gpu_ids[@]}"
  fi
fi
GPU_IDS="$GPU_IDS_CSV"

DATASET_SLUG="${DATASET_SLUG:-deepscaler_train${TRAIN_PROMPTS}_validation${VALIDATION_PROMPTS}_seed${SEED}}"
RUN_SUFFIX="${RUN_SUFFIX:-temp${TEMPERATURE}_topp${TOP_P}_topk${TOP_K}_train${TRAIN_PROMPTS}x${TRAIN_NUM_SAMPLES}_validation${VALIDATION_PROMPTS}x${VALIDATION_NUM_SAMPLES}_vllm_tp${TP_SIZE}_seed${SEED}}"
PROMPT_MODEL_SLUG="${PROMPT_MODEL_SLUG:-deepseek_r1_distill_qwen_1_5b_l${LAYERS}_last${PROMPT_LAST_N_VALUES}mean}"
ROLLOUT_MODEL_SLUG="${ROLLOUT_MODEL_SLUG:-deepseek_r1_distill_qwen_1_5b_l${LAYERS}_thinkendlast10}"
RESPONSE_DATASET_NAME="${RESPONSE_DATASET_NAME:-${DATASET_SLUG}_${RUN_SUFFIX}_thinkendlast10_l${LAYERS}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/classifer_training/artifacts/logs/deepseek_r1_distill_qwen_1_5b_${DATASET_SLUG}_${RUN_SUFFIX}}"

export ROOT MODEL_NAME MODEL_SLUG GPU_IDS TP_SIZE GENERATION_PARALLELISM TRAIN_PROMPTS VALIDATION_PROMPTS
export TRAIN_NUM_SAMPLES VALIDATION_NUM_SAMPLES TRAIN_GENERATION_SHARD_SIZE VALIDATION_GENERATION_SHARD_SIZE
export TEMPERATURE TOP_P TOP_K SEED LAYERS PROMPT_LAST_N_VALUES ROLLOUT_COMPONENTS
export PROMPT_BATCH_SIZE ROLLOUT_BATCH_SIZE ROLLOUT_MAX_BATCH_TOKENS
export DATASET_SLUG RUN_SUFFIX PROMPT_MODEL_SLUG ROLLOUT_MODEL_SLUG RESPONSE_DATASET_NAME LOG_ROOT

exec bash "${SCRIPT_DIR}/run_deepscaler_custom_qwen3_4b_base_4gpu.sh" "$@"
