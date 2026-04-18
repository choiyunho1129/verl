#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/jongwonlim/anaconda3/envs/CB/bin/python}"
BASE_MODEL_NAME="${BASE_MODEL_NAME:-/data2/sangjunsong/.cache/hf_hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}"
BASE_MODEL_SLUG="${BASE_MODEL_SLUG:-qwen3_4b_instruct_2507}"
TRAINED_MODEL_NAME="${TRAINED_MODEL_NAME:-jaygala24/Qwen3-4B-GRPO-math-reasoning}"
TRAINED_MODEL_SLUG="${TRAINED_MODEL_SLUG:-jaygala24_Qwen3-4B-GRPO-math-reasoning}"
DEEPSCALER_DATASET_DIR="${DEEPSCALER_DATASET_DIR:-${REPO_ROOT}/classifer_training/artifacts/datasets/deepscaler}"
DEEPSCALER_SHARD_DIR="${DEEPSCALER_SHARD_DIR:-${REPO_ROOT}/classifer_training/artifacts/datasets/deepscaler_val500_test500_shards4}"
GPU_IDS="${GPU_IDS:-0 1 2 3}"
NUM_SAMPLES="${NUM_SAMPLES:-4}"
SEED="${SEED:-1}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
MATH_MAX_NEW_TOKENS="${MATH_MAX_NEW_TOKENS:-8192}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-64}"
EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE:-4}"
MAX_BATCH_TOKENS="${MAX_BATCH_TOKENS:-16000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"
RUN_SUFFIX="${RUN_SUFFIX:-temp${TEMPERATURE}_topp${TOP_P}_topk${TOP_K}_multisample${NUM_SAMPLES}_val500_test500_vllm_tp4_seed${SEED}}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/classifer_training/artifacts/logs/math_transfer_gpu_chain_${RUN_SUFFIX}}"

export PYTHONPATH="${REPO_ROOT}"
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p "${LOG_ROOT}"
PIPELINE_LOG="${LOG_ROOT}/pipeline.log"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

read -r GPU0 GPU1 GPU2 GPU3 <<<"${GPU_IDS}"

BASE_RUN_DIR="${REPO_ROOT}/classifer_training/artifacts/runs/deepscaler/${BASE_MODEL_SLUG}/${RUN_SUFFIX}"
BASE_LABELS_PATH="${REPO_ROOT}/classifer_training/artifacts/labels/deepscaler/${BASE_MODEL_SLUG}/deepscaler_${RUN_SUFFIX}_labels.jsonl"
BASE_LABELS_SUMMARY="${REPO_ROOT}/classifer_training/artifacts/labels/deepscaler/${BASE_MODEL_SLUG}/deepscaler_${RUN_SUFFIX}_summary.json"
BASE_PROMPT_DATASET_SCRATCH="${REPO_ROOT}/classifer_training/artifacts/datasets/deepscaler_${RUN_SUFFIX}_${BASE_MODEL_SLUG}_labels_scratch"
BASE_RESPONSE_DATASET_NAME="deepscaler_${RUN_SUFFIX}_${BASE_MODEL_SLUG}_response_l26"

TRAINED_RUN_DIR="${REPO_ROOT}/classifer_training/artifacts/runs/deepscaler/${TRAINED_MODEL_SLUG}/${RUN_SUFFIX}"
TRAINED_LABELS_PATH="${REPO_ROOT}/classifer_training/artifacts/labels/deepscaler/${TRAINED_MODEL_SLUG}/deepscaler_${RUN_SUFFIX}_labels.jsonl"
TRAINED_LABELS_SUMMARY="${REPO_ROOT}/classifer_training/artifacts/labels/deepscaler/${TRAINED_MODEL_SLUG}/deepscaler_${RUN_SUFFIX}_summary.json"
TRAINED_PROMPT_DATASET_SCRATCH="${REPO_ROOT}/classifer_training/artifacts/datasets/deepscaler_${RUN_SUFFIX}_${TRAINED_MODEL_SLUG}_labels_scratch"
TRAINED_RESPONSE_DATASET_NAME="deepscaler_${RUN_SUFFIX}_${TRAINED_MODEL_SLUG}_response_l26"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

all_exist() {
  local path
  for path in "$@"; do
    if [[ ! -e "${path}" ]]; then
      return 1
    fi
  done
  return 0
}

find_one() {
  local root="$1"
  local pattern="$2"
  find "${root}" -name "${pattern}" 2>/dev/null | head -n 1
}

run_prompt_hidden_shards() {
  local model_name="$1"
  local shard_dir="$2"
  local dataset_prefix="$3"
  local model_slug="$4"
  local log_prefix="$5"

  local pids=()
  for shard_idx in 0 1 2 3; do
    local gpu_var="GPU${shard_idx}"
    local gpu_id="${!gpu_var}"
    local shard_path="${shard_dir}/shard${shard_idx}.jsonl"
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" -u -m classifer_training.extract_hidden_states \
      --model_name_or_path "${model_name}" \
      --input_path "${shard_path}" \
      --dataset_name "${dataset_prefix}_shard${shard_idx}" \
      --model_slug "${model_slug}" \
      --token_pooling lastn_mean \
      --last_n 6 \
      --batch_size 16 \
      --trust_remote_code \
      --overwrite \
      > "${LOG_ROOT}/${log_prefix}.shard${shard_idx}.log" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
}

run_response_hidden_shards() {
  local model_name="$1"
  local run_dir="$2"
  local dataset_name="$3"
  local log_prefix="$4"

  local pids=()
  for shard_idx in 0 1 2 3; do
    local gpu_var="GPU${shard_idx}"
    local gpu_id="${!gpu_var}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" -u -m classifer_training.extract_rollout_hidden_states \
      --model_name_or_path "${model_name}" \
      --run_dirs "${run_dir}" \
      --dataset_name "${dataset_name}" \
      --components think_end_hidden \
      --layers 26 \
      --response_anchor reasoning_or_answer \
      --hidden_root "${REPO_ROOT}/classifer_training/artifacts/rollout_hidden" \
      --index_root "${REPO_ROOT}/classifer_training/artifacts/rollout_index" \
      --num_shards 4 \
      --shard_index "${shard_idx}" \
      --batch_size "${EXTRACT_BATCH_SIZE}" \
      --max_batch_tokens "${MAX_BATCH_TOKENS}" \
      --trust_remote_code \
      --overwrite \
      > "${LOG_ROOT}/${log_prefix}.shard${shard_idx}.log" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
}

echo "[start] $(timestamp) math transfer gpu chain"

if all_exist "${BASE_RUN_DIR}/all_experiments.jsonl" "${BASE_RUN_DIR}/evaluation_results.jsonl"; then
  echo "[skip] DeepScaleR base-model generation already exists $(timestamp)"
else
  echo "[stage] DeepScaleR base-model generation $(timestamp)"
  "${PYTHON_BIN}" -u -m classifer_training.sample \
    --model_name_or_path "${BASE_MODEL_NAME}" \
    --input_path "${DEEPSCALER_DATASET_DIR}" \
    --dataset_name deepscaler \
    --output_dir "${BASE_RUN_DIR}" \
    --backend vllm \
    --grader math_verify \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --top_k "${TOP_K}" \
    --max_new_tokens "${MATH_MAX_NEW_TOKENS}" \
    --batch_size "${GEN_BATCH_SIZE}" \
    --seed "${SEED}" \
    --num_samples "${NUM_SAMPLES}" \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --trust_remote_code \
    --overwrite
fi

if all_exist "${BASE_LABELS_PATH}" "${BASE_LABELS_SUMMARY}"; then
  echo "[skip] DeepScaleR base-model labels already exist $(timestamp)"
else
  echo "[stage] DeepScaleR base-model labels $(timestamp)"
  "${PYTHON_BIN}" -u -m classifer_training.build_weak_prompt_dataset_and_labels \
    --run_dirs "${BASE_RUN_DIR}" \
    --prompt_dataset_dir "${BASE_PROMPT_DATASET_SCRATCH}" \
    --labels_path "${BASE_LABELS_PATH}" \
    --summary_path "${BASE_LABELS_SUMMARY}"
fi

if all_exist \
  "${REPO_ROOT}/classifer_training/artifacts/hidden/deepscaler_val500_test500_shard0/${BASE_MODEL_SLUG}/hidden_states.pt" \
  "${REPO_ROOT}/classifer_training/artifacts/hidden/deepscaler_val500_test500_shard1/${BASE_MODEL_SLUG}/hidden_states.pt" \
  "${REPO_ROOT}/classifer_training/artifacts/hidden/deepscaler_val500_test500_shard2/${BASE_MODEL_SLUG}/hidden_states.pt" \
  "${REPO_ROOT}/classifer_training/artifacts/hidden/deepscaler_val500_test500_shard3/${BASE_MODEL_SLUG}/hidden_states.pt"; then
  echo "[skip] DeepScaleR base-model prompt hidden already exists $(timestamp)"
else
  echo "[stage] DeepScaleR base-model prompt hidden extraction $(timestamp)"
  run_prompt_hidden_shards "${BASE_MODEL_NAME}" "${DEEPSCALER_SHARD_DIR}" "deepscaler_val500_test500" "${BASE_MODEL_SLUG}" "deepscaler_base_prompt_hidden"
fi

if all_exist \
  "$(find_one "${REPO_ROOT}/classifer_training/artifacts/rollout_hidden/${BASE_RESPONSE_DATASET_NAME}" 'rollout_hidden_states.shard00of04.pt')" \
  "$(find_one "${REPO_ROOT}/classifer_training/artifacts/rollout_hidden/${BASE_RESPONSE_DATASET_NAME}" 'rollout_hidden_states.shard01of04.pt')" \
  "$(find_one "${REPO_ROOT}/classifer_training/artifacts/rollout_hidden/${BASE_RESPONSE_DATASET_NAME}" 'rollout_hidden_states.shard02of04.pt')" \
  "$(find_one "${REPO_ROOT}/classifer_training/artifacts/rollout_hidden/${BASE_RESPONSE_DATASET_NAME}" 'rollout_hidden_states.shard03of04.pt')" \
  "$(find_one "${REPO_ROOT}/classifer_training/artifacts/rollout_index/${BASE_RESPONSE_DATASET_NAME}" 'rollout_index.shard00of04.jsonl')" \
  "$(find_one "${REPO_ROOT}/classifer_training/artifacts/rollout_index/${BASE_RESPONSE_DATASET_NAME}" 'rollout_index.shard03of04.jsonl')"; then
  echo "[skip] DeepScaleR base-model response hidden already exists $(timestamp)"
else
  echo "[stage] DeepScaleR base-model response hidden extraction $(timestamp)"
  run_response_hidden_shards "${BASE_MODEL_NAME}" "${BASE_RUN_DIR}" "${BASE_RESPONSE_DATASET_NAME}" "deepscaler_base_response_hidden"
fi

if all_exist "${TRAINED_RUN_DIR}/all_experiments.jsonl" "${TRAINED_RUN_DIR}/evaluation_results.jsonl"; then
  echo "[skip] DeepScaleR trained-model generation already exists $(timestamp)"
else
  echo "[stage] DeepScaleR trained-model generation $(timestamp)"
  "${PYTHON_BIN}" -u -m classifer_training.sample \
    --model_name_or_path "${TRAINED_MODEL_NAME}" \
    --input_path "${DEEPSCALER_DATASET_DIR}" \
    --dataset_name deepscaler \
    --output_dir "${TRAINED_RUN_DIR}" \
    --backend vllm \
    --grader math_verify \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --top_k "${TOP_K}" \
    --max_new_tokens "${MATH_MAX_NEW_TOKENS}" \
    --batch_size "${GEN_BATCH_SIZE}" \
    --seed "${SEED}" \
    --num_samples "${NUM_SAMPLES}" \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --trust_remote_code \
    --overwrite
fi

if all_exist "${TRAINED_LABELS_PATH}" "${TRAINED_LABELS_SUMMARY}"; then
  echo "[skip] DeepScaleR trained-model labels already exist $(timestamp)"
else
  echo "[stage] DeepScaleR trained-model labels $(timestamp)"
  "${PYTHON_BIN}" -u -m classifer_training.build_weak_prompt_dataset_and_labels \
    --run_dirs "${TRAINED_RUN_DIR}" \
    --prompt_dataset_dir "${TRAINED_PROMPT_DATASET_SCRATCH}" \
    --labels_path "${TRAINED_LABELS_PATH}" \
    --summary_path "${TRAINED_LABELS_SUMMARY}"
fi

if all_exist \
  "${REPO_ROOT}/classifer_training/artifacts/hidden/deepscaler_val500_test500_shard0/${TRAINED_MODEL_SLUG}/hidden_states.pt" \
  "${REPO_ROOT}/classifer_training/artifacts/hidden/deepscaler_val500_test500_shard1/${TRAINED_MODEL_SLUG}/hidden_states.pt" \
  "${REPO_ROOT}/classifer_training/artifacts/hidden/deepscaler_val500_test500_shard2/${TRAINED_MODEL_SLUG}/hidden_states.pt" \
  "${REPO_ROOT}/classifer_training/artifacts/hidden/deepscaler_val500_test500_shard3/${TRAINED_MODEL_SLUG}/hidden_states.pt"; then
  echo "[skip] DeepScaleR trained-model prompt hidden already exists $(timestamp)"
else
  echo "[stage] DeepScaleR trained-model prompt hidden extraction $(timestamp)"
  run_prompt_hidden_shards "${TRAINED_MODEL_NAME}" "${DEEPSCALER_SHARD_DIR}" "deepscaler_val500_test500" "${TRAINED_MODEL_SLUG}" "deepscaler_trained_prompt_hidden"
fi

if all_exist \
  "$(find_one "${REPO_ROOT}/classifer_training/artifacts/rollout_hidden/${TRAINED_RESPONSE_DATASET_NAME}" 'rollout_hidden_states.shard00of04.pt')" \
  "$(find_one "${REPO_ROOT}/classifer_training/artifacts/rollout_hidden/${TRAINED_RESPONSE_DATASET_NAME}" 'rollout_hidden_states.shard01of04.pt')" \
  "$(find_one "${REPO_ROOT}/classifer_training/artifacts/rollout_hidden/${TRAINED_RESPONSE_DATASET_NAME}" 'rollout_hidden_states.shard02of04.pt')" \
  "$(find_one "${REPO_ROOT}/classifer_training/artifacts/rollout_hidden/${TRAINED_RESPONSE_DATASET_NAME}" 'rollout_hidden_states.shard03of04.pt')" \
  "$(find_one "${REPO_ROOT}/classifer_training/artifacts/rollout_index/${TRAINED_RESPONSE_DATASET_NAME}" 'rollout_index.shard00of04.jsonl')" \
  "$(find_one "${REPO_ROOT}/classifer_training/artifacts/rollout_index/${TRAINED_RESPONSE_DATASET_NAME}" 'rollout_index.shard03of04.jsonl')"; then
  echo "[skip] DeepScaleR trained-model response hidden already exists $(timestamp)"
else
  echo "[stage] DeepScaleR trained-model response hidden extraction $(timestamp)"
  run_response_hidden_shards "${TRAINED_MODEL_NAME}" "${TRAINED_RUN_DIR}" "${TRAINED_RESPONSE_DATASET_NAME}" "deepscaler_trained_response_hidden"
fi

echo "[done] $(timestamp) math transfer gpu chain"
