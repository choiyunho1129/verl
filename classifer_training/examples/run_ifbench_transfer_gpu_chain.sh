#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/jongwonlim/anaconda3/envs/CB/bin/python}"
BASE_MODEL_NAME="${BASE_MODEL_NAME:-/data2/sangjunsong/.cache/hf_hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}"
BASE_MODEL_SLUG="${BASE_MODEL_SLUG:-qwen3_4b_instruct_2507}"
IFBENCH_INPUT_PATH="${IFBENCH_INPUT_PATH:-${REPO_ROOT}/classifer_training/external/IFBench/data/IFBench_test.jsonl}"
IFBENCH_DATASET_DIR="${IFBENCH_DATASET_DIR:-${REPO_ROOT}/classifer_training/artifacts/datasets/ifbench_test}"
IFBENCH_SHARD_DIR="${IFBENCH_SHARD_DIR:-${REPO_ROOT}/classifer_training/artifacts/datasets/ifbench_test_shards4}"
GPU_IDS="${GPU_IDS:-0 1 2 3}"
NUM_SAMPLES="${NUM_SAMPLES:-4}"
SEED="${SEED:-1}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
IFBENCH_MAX_NEW_TOKENS="${IFBENCH_MAX_NEW_TOKENS:-8192}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-64}"
EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE:-4}"
MAX_BATCH_TOKENS="${MAX_BATCH_TOKENS:-16000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"
RUN_SUFFIX="${RUN_SUFFIX:-temp${TEMPERATURE}_topp${TOP_P}_topk${TOP_K}_multisample${NUM_SAMPLES}_test300_vllm_tp4_seed${SEED}}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/classifer_training/artifacts/logs/ifbench_transfer_gpu_chain_${RUN_SUFFIX}}"

export PYTHONPATH="${REPO_ROOT}"
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p "${LOG_ROOT}"
PIPELINE_LOG="${LOG_ROOT}/pipeline.log"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

read -r GPU0 GPU1 GPU2 GPU3 <<<"${GPU_IDS}"

IFBENCH_RUN_DIR="${REPO_ROOT}/classifer_training/artifacts/runs/ifbench/${BASE_MODEL_SLUG}/${RUN_SUFFIX}"
IFBENCH_LABELS_PATH="${REPO_ROOT}/classifer_training/artifacts/labels/ifbench/${BASE_MODEL_SLUG}/ifbench_${RUN_SUFFIX}_labels.jsonl"
IFBENCH_LABELS_SUMMARY="${REPO_ROOT}/classifer_training/artifacts/labels/ifbench/${BASE_MODEL_SLUG}/ifbench_${RUN_SUFFIX}_summary.json"
IFBENCH_PROMPT_DATASET_SCRATCH="${REPO_ROOT}/classifer_training/artifacts/datasets/ifbench_${RUN_SUFFIX}_${BASE_MODEL_SLUG}_labels_scratch"
IFBENCH_RESPONSE_DATASET_NAME="ifbench_${RUN_SUFFIX}_response_l26"

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
  local pids=()
  for shard_idx in 0 1 2 3; do
    local gpu_var="GPU${shard_idx}"
    local gpu_id="${!gpu_var}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" -u -m classifer_training.extract_hidden_states \
      --model_name_or_path "${BASE_MODEL_NAME}" \
      --input_path "${IFBENCH_SHARD_DIR}/shard${shard_idx}.jsonl" \
      --dataset_name "ifbench_test_shard${shard_idx}" \
      --model_slug "${BASE_MODEL_SLUG}" \
      --token_pooling lastn_mean \
      --last_n 6 \
      --batch_size 16 \
      --trust_remote_code \
      --overwrite \
      > "${LOG_ROOT}/ifbench_prompt_hidden.shard${shard_idx}.log" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
}

run_response_hidden_shards() {
  local pids=()
  for shard_idx in 0 1 2 3; do
    local gpu_var="GPU${shard_idx}"
    local gpu_id="${!gpu_var}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" -u -m classifer_training.extract_rollout_hidden_states \
      --model_name_or_path "${BASE_MODEL_NAME}" \
      --run_dirs "${IFBENCH_RUN_DIR}" \
      --dataset_name "${IFBENCH_RESPONSE_DATASET_NAME}" \
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
      > "${LOG_ROOT}/ifbench_response_hidden.shard${shard_idx}.log" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
}

echo "[start] $(timestamp) ifbench transfer gpu chain"

if all_exist "${IFBENCH_DATASET_DIR}/test.jsonl" "${IFBENCH_SHARD_DIR}/shard0.jsonl" "${IFBENCH_SHARD_DIR}/shard1.jsonl" "${IFBENCH_SHARD_DIR}/shard2.jsonl" "${IFBENCH_SHARD_DIR}/shard3.jsonl"; then
  echo "[skip] IFBench dataset already prepared $(timestamp)"
else
  echo "[stage] prepare IFBench dataset $(timestamp)"
  "${PYTHON_BIN}" -u -m classifer_training.prepare_ifbench_dataset \
    --input_path "${IFBENCH_INPUT_PATH}" \
    --output_dir "${IFBENCH_DATASET_DIR}" \
    --shard_dir "${IFBENCH_SHARD_DIR}" \
    --dataset_name ifbench_test \
    --num_shards 4
fi

if all_exist "${IFBENCH_RUN_DIR}/all_experiments.jsonl" "${IFBENCH_RUN_DIR}/evaluation_results.jsonl"; then
  echo "[skip] IFBench generation already exists $(timestamp)"
else
  echo "[stage] IFBench generation $(timestamp)"
  "${PYTHON_BIN}" -u -m classifer_training.sample \
    --model_name_or_path "${BASE_MODEL_NAME}" \
    --input_path "${IFBENCH_DATASET_DIR}" \
    --dataset_name ifbench_test \
    --output_dir "${IFBENCH_RUN_DIR}" \
    --backend vllm \
    --grader exact \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --top_k "${TOP_K}" \
    --max_new_tokens "${IFBENCH_MAX_NEW_TOKENS}" \
    --batch_size "${GEN_BATCH_SIZE}" \
    --seed "${SEED}" \
    --num_samples "${NUM_SAMPLES}" \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --trust_remote_code \
    --overwrite
fi

if all_exist "${IFBENCH_LABELS_PATH}" "${IFBENCH_LABELS_SUMMARY}" "${IFBENCH_RUN_DIR}/ifbench_loose_summary.json"; then
  echo "[skip] IFBench rescore + labels already exist $(timestamp)"
else
  echo "[stage] IFBench rescore + labels $(timestamp)"
  "${PYTHON_BIN}" -u -m classifer_training.rescore_ifbench_run \
    --run_dir "${IFBENCH_RUN_DIR}" \
    --ifbench_input_path "${IFBENCH_INPUT_PATH}" \
    --mode loose \
    --overwrite
  "${PYTHON_BIN}" -u -m classifer_training.build_weak_prompt_dataset_and_labels \
    --run_dirs "${IFBENCH_RUN_DIR}" \
    --prompt_dataset_dir "${IFBENCH_PROMPT_DATASET_SCRATCH}" \
    --labels_path "${IFBENCH_LABELS_PATH}" \
    --summary_path "${IFBENCH_LABELS_SUMMARY}"
fi

if all_exist \
  "${REPO_ROOT}/classifer_training/artifacts/hidden/ifbench_test_shard0/${BASE_MODEL_SLUG}/hidden_states.pt" \
  "${REPO_ROOT}/classifer_training/artifacts/hidden/ifbench_test_shard1/${BASE_MODEL_SLUG}/hidden_states.pt" \
  "${REPO_ROOT}/classifer_training/artifacts/hidden/ifbench_test_shard2/${BASE_MODEL_SLUG}/hidden_states.pt" \
  "${REPO_ROOT}/classifer_training/artifacts/hidden/ifbench_test_shard3/${BASE_MODEL_SLUG}/hidden_states.pt"; then
  echo "[skip] IFBench prompt hidden already exists $(timestamp)"
else
  echo "[stage] IFBench prompt hidden extraction $(timestamp)"
  run_prompt_hidden_shards
fi

if all_exist \
  "$(find_one "${REPO_ROOT}/classifer_training/artifacts/rollout_hidden/${IFBENCH_RESPONSE_DATASET_NAME}" 'rollout_hidden_states.shard00of04.pt')" \
  "$(find_one "${REPO_ROOT}/classifer_training/artifacts/rollout_hidden/${IFBENCH_RESPONSE_DATASET_NAME}" 'rollout_hidden_states.shard01of04.pt')" \
  "$(find_one "${REPO_ROOT}/classifer_training/artifacts/rollout_hidden/${IFBENCH_RESPONSE_DATASET_NAME}" 'rollout_hidden_states.shard02of04.pt')" \
  "$(find_one "${REPO_ROOT}/classifer_training/artifacts/rollout_hidden/${IFBENCH_RESPONSE_DATASET_NAME}" 'rollout_hidden_states.shard03of04.pt')" \
  "$(find_one "${REPO_ROOT}/classifer_training/artifacts/rollout_index/${IFBENCH_RESPONSE_DATASET_NAME}" 'rollout_index.shard00of04.jsonl')" \
  "$(find_one "${REPO_ROOT}/classifer_training/artifacts/rollout_index/${IFBENCH_RESPONSE_DATASET_NAME}" 'rollout_index.shard03of04.jsonl')"; then
  echo "[skip] IFBench response hidden already exists $(timestamp)"
else
  echo "[stage] IFBench response hidden extraction $(timestamp)"
  run_response_hidden_shards
fi

echo "[done] $(timestamp) ifbench transfer gpu chain"
