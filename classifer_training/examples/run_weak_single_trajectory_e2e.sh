#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data2/jongwonlim/verl/yoonho/verl}"
PYTHON_BIN="${PYTHON_BIN:-/home/jongwonlim/anaconda3/bin/python}"
MODEL_NAME="${MODEL_NAME:-/data2/sangjunsong/.cache/hf_hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}"
MODEL_SLUG="${MODEL_SLUG:-qwen3_4b_instruct_2507}"
GPU_IDS_STR="${GPU_IDS:-0 1 2 3}"
VAL_RATIO="${VAL_RATIO:-0.2}"
OVERWRITE="${OVERWRITE:-0}"

PROMPT_BATCH_SIZE="${PROMPT_BATCH_SIZE:-8}"
RESPONSE_BATCH_SIZE="${RESPONSE_BATCH_SIZE:-4}"
MAX_BATCH_TOKENS="${MAX_BATCH_TOKENS:-16000}"

PROMPT_LAYER_INDEX="${PROMPT_LAYER_INDEX:-26}"
PROMPT_LAST_N="${PROMPT_LAST_N:-6}"
ROLLOUT_COMPONENT="${ROLLOUT_COMPONENT:-think_end_hidden}"
ROLLOUT_POOL_MODE="${ROLLOUT_POOL_MODE:-mean}"
PROMPT_HIDDEN_PCA_DIM="${PROMPT_HIDDEN_PCA_DIM:-0}"
ROLLOUT_HIDDEN_PCA_DIM="${ROLLOUT_HIDDEN_PCA_DIM:-32}"
SINGLE_ROLLOUT_STRATEGY="${SINGLE_ROLLOUT_STRATEGY:-first}"
ALPHAS_STR="${ALPHAS:-100 300 1000 3000 10000}"

WEAK_RUN_DIR0="${WEAK_RUN_DIR0:-${ROOT}/classifer_training/artifacts/runs/dapo_math_17k/qwen3_4b_instruct_2507/weak4_runs/0}"
WEAK_RUN_DIR1="${WEAK_RUN_DIR1:-${ROOT}/classifer_training/artifacts/runs/dapo_math_17k/qwen3_4b_instruct_2507/weak4_runs/1}"

DATASET_NAME="${DATASET_NAME:-dapo_math_17k_weak4_val20}"
DATASET_DIR="${DATASET_DIR:-${ROOT}/classifer_training/artifacts/datasets/${DATASET_NAME}}"
LABELS_PATH="${LABELS_PATH:-${ROOT}/classifer_training/artifacts/labels/dapo_math_17k/qwen3_4b_instruct_2507/weak4_labels_val20.jsonl}"
SHARD_DIR="${SHARD_DIR:-${ROOT}/classifer_training/artifacts/datasets/${DATASET_NAME}_shards}"
LOG_DIR="${LOG_DIR:-${ROOT}/classifer_training/artifacts/logs/${DATASET_NAME}_e2e}"

PROMPT_DATASET_PREFIX="${PROMPT_DATASET_PREFIX:-${DATASET_NAME}_shard}"
PROMPT_MODEL_SLUG="${PROMPT_MODEL_SLUG:-${MODEL_SLUG}_last6mean}"
RESPONSE_DATASET_NAME="${RESPONSE_DATASET_NAME:-${DATASET_NAME}_response_l${PROMPT_LAYER_INDEX}}"
MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR:-${ROOT}/classifer_training/artifacts/models/${DATASET_NAME}_e2e_${ROLLOUT_COMPONENT}_ppca${PROMPT_HIDDEN_PCA_DIM}_rpca${ROLLOUT_HIDDEN_PCA_DIM}}"

read -r -a GPU_IDS <<<"${GPU_IDS_STR}"
NUM_SHARDS="${NUM_SHARDS:-${#GPU_IDS[@]}}"
if [ "${NUM_SHARDS}" -lt 1 ]; then
  echo "NUM_SHARDS must be at least 1." >&2
  exit 1
fi

OVERWRITE_FLAG=()
if [ "${OVERWRITE}" = "1" ]; then
  OVERWRITE_FLAG+=(--overwrite)
fi

mkdir -p "${LOG_DIR}"
mkdir -p "$(dirname "${LABELS_PATH}")"
cd "${ROOT}"

WEAK_RUN_DIRS=(
  "${WEAK_RUN_DIR0}"
  "${WEAK_RUN_DIR1}"
)

PROMPT_SCALAR_KEYS=()
ROLLOUT_SCALAR_KEYS=(
  output_length
  think_tokens
  answer_tokens
  has_complete_answer
  has_reasoning_content
  output_mean_token_entropy
  reasoning_mean_token_entropy
  answer_mean_token_entropy
  output_unique_token_ratio
  answer_unique_token_ratio
  output_repetition_ratio
  reasoning_repetition_ratio
  duplicate_line_ratio
)
DERIVED_ROLLOUT_SCALAR_KEYS=(
  think_ratio
  answer_ratio
  entropy_gap_reasoning_answer
  unique_gap_reasoning_output
  repetition_gap_reasoning_output
  reasoning_x_log_output_length
  answer_entropy_gap_vs_output
)

build_dataset() {
  local summary_path="${DATASET_DIR}/summary.json"
  if [ "${OVERWRITE}" != "1" ] && [ -f "${DATASET_DIR}/train.jsonl" ] && [ -f "${DATASET_DIR}/validation.jsonl" ] && [ -f "${LABELS_PATH}" ] && [ -f "${summary_path}" ]; then
    echo "[skip] weak dataset already exists: ${DATASET_DIR}"
    return
  fi

  "${PYTHON_BIN}" -u -m classifer_training.build_weak_prompt_dataset_and_labels \
    --run_dirs "${WEAK_RUN_DIRS[@]}" \
    --prompt_dataset_dir "${DATASET_DIR}" \
    --labels_path "${LABELS_PATH}" \
    --summary_path "${summary_path}" \
    --val_ratio "${VAL_RATIO}" \
    --ignore_existing_split
}

prepare_shards() {
  "${PYTHON_BIN}" -u -m classifer_training.prepare_weak4_shards \
    --input_dir "${DATASET_DIR}" \
    --output_dir "${SHARD_DIR}" \
    --num_shards "${NUM_SHARDS}" \
    "${OVERWRITE_FLAG[@]}"
}

extract_prompt_hidden() {
  local -a pids=()
  local shard_index
  for shard_index in $(seq 0 $((NUM_SHARDS - 1))); do
    local gpu="${GPU_IDS[$((shard_index % ${#GPU_IDS[@]}))]}"
    local dataset_name="${PROMPT_DATASET_PREFIX}${shard_index}"
    local shard_file="${SHARD_DIR}/shard${shard_index}.jsonl"
    local hidden_path="${ROOT}/classifer_training/artifacts/hidden/${dataset_name}/${PROMPT_MODEL_SLUG}/hidden_states.pt"
    local index_path="${ROOT}/classifer_training/artifacts/index/${dataset_name}/${PROMPT_MODEL_SLUG}/index.jsonl"
    local log_path="${LOG_DIR}/prompt_hidden.shard${shard_index}.log"

    if [ "${OVERWRITE}" != "1" ] && [ -f "${hidden_path}" ] && [ -f "${index_path}" ]; then
      echo "[skip] prompt hidden shard${shard_index} already exists"
      continue
    fi

    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=. TOKENIZERS_PARALLELISM=false \
      "${PYTHON_BIN}" -u -m classifer_training.extract_hidden_states \
        --model_name_or_path "${MODEL_NAME}" \
        --input_path "${shard_file}" \
        --dataset_name "${dataset_name}" \
        --model_slug "${PROMPT_MODEL_SLUG}" \
        --components hidden \
        --token_pooling lastn_mean \
        --last_n "${PROMPT_LAST_N}" \
        --batch_size "${PROMPT_BATCH_SIZE}" \
        --trust_remote_code \
        "${OVERWRITE_FLAG[@]}" \
        > "${log_path}" 2>&1 &
    pids+=("$!")
  done

  if [ "${#pids[@]}" -gt 0 ]; then
    wait "${pids[@]}"
  fi
}

extract_response_hidden() {
  local -a pids=()
  local shard_index
  local model_slug
  model_slug="$("${PYTHON_BIN}" - <<'PY' "${MODEL_NAME}"
import sys
from classifer_training.utils import sanitize_name
print(sanitize_name(sys.argv[1]))
PY
)"

  for shard_index in $(seq 0 $((NUM_SHARDS - 1))); do
    local gpu="${GPU_IDS[$((shard_index % ${#GPU_IDS[@]}))]}"
    local hidden_path="${ROOT}/classifer_training/artifacts/rollout_hidden/${RESPONSE_DATASET_NAME}/${model_slug}/rollout_hidden_states.shard$(printf '%02d' "${shard_index}")of$(printf '%02d' "${NUM_SHARDS}").pt"
    local index_path="${ROOT}/classifer_training/artifacts/rollout_index/${RESPONSE_DATASET_NAME}/${model_slug}/rollout_index.shard$(printf '%02d' "${shard_index}")of$(printf '%02d' "${NUM_SHARDS}").jsonl"
    local log_path="${LOG_DIR}/response_hidden.shard${shard_index}.log"

    if [ "${OVERWRITE}" != "1" ] && [ -f "${hidden_path}" ] && [ -f "${index_path}" ]; then
      echo "[skip] response hidden shard${shard_index} already exists"
      continue
    fi

    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=. TOKENIZERS_PARALLELISM=false \
      "${PYTHON_BIN}" -u -m classifer_training.extract_rollout_hidden_states \
        --model_name_or_path "${MODEL_NAME}" \
        --run_dirs "${WEAK_RUN_DIRS[@]}" \
        --dataset_name "${RESPONSE_DATASET_NAME}" \
        --components think_end_hidden think_end_last10_hidden \
        --layers "${PROMPT_LAYER_INDEX}" \
        --response_anchor reasoning \
        --hidden_root "${ROOT}/classifer_training/artifacts/rollout_hidden" \
        --index_root "${ROOT}/classifer_training/artifacts/rollout_index" \
        --hidden_filename rollout_hidden_states.pt \
        --index_filename rollout_index.jsonl \
        --num_shards "${NUM_SHARDS}" \
        --shard_index "${shard_index}" \
        --batch_size "${RESPONSE_BATCH_SIZE}" \
        --max_batch_tokens "${MAX_BATCH_TOKENS}" \
        --trust_remote_code \
        --local_files_only \
        "${OVERWRITE_FLAG[@]}" \
        > "${log_path}" 2>&1 &
    pids+=("$!")
  done

  if [ "${#pids[@]}" -gt 0 ]; then
    wait "${pids[@]}"
  fi
}

train_model() {
  local -a prompt_hidden_paths=()
  local -a prompt_index_paths=()
  local -a response_hidden_paths=()
  local -a response_index_paths=()
  local shard_index
  local model_slug
  model_slug="$("${PYTHON_BIN}" - <<'PY' "${MODEL_NAME}"
import sys
from classifer_training.utils import sanitize_name
print(sanitize_name(sys.argv[1]))
PY
)"

  for shard_index in $(seq 0 $((NUM_SHARDS - 1))); do
    local dataset_name="${PROMPT_DATASET_PREFIX}${shard_index}"
    prompt_hidden_paths+=("${ROOT}/classifer_training/artifacts/hidden/${dataset_name}/${PROMPT_MODEL_SLUG}/hidden_states.pt")
    prompt_index_paths+=("${ROOT}/classifer_training/artifacts/index/${dataset_name}/${PROMPT_MODEL_SLUG}/index.jsonl")
    response_hidden_paths+=("${ROOT}/classifer_training/artifacts/rollout_hidden/${RESPONSE_DATASET_NAME}/${model_slug}/rollout_hidden_states.shard$(printf '%02d' "${shard_index}")of$(printf '%02d' "${NUM_SHARDS}").pt")
    response_index_paths+=("${ROOT}/classifer_training/artifacts/rollout_index/${RESPONSE_DATASET_NAME}/${model_slug}/rollout_index.shard$(printf '%02d' "${shard_index}")of$(printf '%02d' "${NUM_SHARDS}").jsonl")
  done

  read -r -a alpha_args <<<"${ALPHAS_STR}"

  PYTHONPATH=. "${PYTHON_BIN}" -u -m classifer_training.train_weak_only_single_rollout_hidden \
    --weak_run_dirs "${WEAK_RUN_DIRS[@]}" \
    --weak_prompt_dataset_dir "${DATASET_DIR}" \
    --weak_labels_path "${LABELS_PATH}" \
    --weak_prompt_hidden_paths "${prompt_hidden_paths[@]}" \
    --weak_prompt_index_paths "${prompt_index_paths[@]}" \
    --weak_rollout_hidden_paths "${response_hidden_paths[@]}" \
    --weak_rollout_index_paths "${response_index_paths[@]}" \
    --output_dir "${MODEL_OUTPUT_DIR}" \
    --prompt_layer_index "${PROMPT_LAYER_INDEX}" \
    --feature_mode prompt_plus_rollout \
    --rollout_component "${ROLLOUT_COMPONENT}" \
    --rollout_pool_mode "${ROLLOUT_POOL_MODE}" \
    --prompt_hidden_pca_dim "${PROMPT_HIDDEN_PCA_DIM}" \
    --rollout_hidden_pca_dim "${ROLLOUT_HIDDEN_PCA_DIM}" \
    --single_rollout_strategy "${SINGLE_ROLLOUT_STRATEGY}" \
    --prompt_feature_keys "${PROMPT_SCALAR_KEYS[@]}" \
    --rollout_scalar_keys "${ROLLOUT_SCALAR_KEYS[@]}" \
    --derived_rollout_scalar_keys "${DERIVED_ROLLOUT_SCALAR_KEYS[@]}" \
    --alphas "${alpha_args[@]}" \
    > "${LOG_DIR}/train.log" 2>&1
}

build_dataset
prepare_shards
extract_prompt_hidden
extract_response_hidden
train_model

echo "done"
