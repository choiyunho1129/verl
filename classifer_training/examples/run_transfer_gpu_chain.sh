#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/jongwonlim/anaconda3/envs/CB/bin/python}"
BASE_MODEL_NAME="${BASE_MODEL_NAME:-/data2/sangjunsong/.cache/hf_hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}"
BASE_MODEL_SLUG="${BASE_MODEL_SLUG:-qwen3_4b_instruct_2507}"
TRAINED_MODEL_NAME="${TRAINED_MODEL_NAME:-jaygala24/Qwen3-4B-GRPO-math-reasoning}"
TRAINED_MODEL_SLUG="${TRAINED_MODEL_SLUG:-jaygala24_Qwen3-4B-GRPO-math-reasoning}"
IFBENCH_INPUT_PATH="${IFBENCH_INPUT_PATH:-${REPO_ROOT}/classifer_training/external/IFBench/data/IFBench_test.jsonl}"
DEEPSCALER_DATASET_DIR="${DEEPSCALER_DATASET_DIR:-${REPO_ROOT}/classifer_training/artifacts/datasets/deepscaler}"
DEEPSCALER_SHARD_DIR="${DEEPSCALER_SHARD_DIR:-${REPO_ROOT}/classifer_training/artifacts/datasets/deepscaler_val500_test500_shards4}"
IFBENCH_DATASET_DIR="${IFBENCH_DATASET_DIR:-${REPO_ROOT}/classifer_training/artifacts/datasets/ifbench_test}"
IFBENCH_SHARD_DIR="${IFBENCH_SHARD_DIR:-${REPO_ROOT}/classifer_training/artifacts/datasets/ifbench_test_shards4}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/classifer_training/artifacts/logs/transfer_gpu_chain_20260418}"
GPU_IDS="${GPU_IDS:-0 1 2 3}"
NUM_SAMPLES="${NUM_SAMPLES:-4}"
SEED="${SEED:-1}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-0}"
IFBENCH_MAX_NEW_TOKENS="${IFBENCH_MAX_NEW_TOKENS:-4096}"
MATH_MAX_NEW_TOKENS="${MATH_MAX_NEW_TOKENS:-8192}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-64}"
EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE:-4}"
MAX_BATCH_TOKENS="${MAX_BATCH_TOKENS:-16000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"

export PYTHONPATH="${REPO_ROOT}"
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p "${LOG_ROOT}"
PIPELINE_LOG="${LOG_ROOT}/pipeline.log"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

read -r GPU0 GPU1 GPU2 GPU3 <<<"${GPU_IDS}"

IFBENCH_RUN_DIR="${REPO_ROOT}/classifer_training/artifacts/runs/ifbench/${BASE_MODEL_SLUG}/temp${TEMPERATURE}_multisample${NUM_SAMPLES}_test300_vllm_tp4_seed${SEED}"
IFBENCH_LABELS_PATH="${REPO_ROOT}/classifer_training/artifacts/labels/ifbench/${BASE_MODEL_SLUG}/ifbench_test300_vllm_tp4_seed${SEED}_labels.jsonl"
IFBENCH_LABELS_SUMMARY="${REPO_ROOT}/classifer_training/artifacts/labels/ifbench/${BASE_MODEL_SLUG}/ifbench_test300_vllm_tp4_seed${SEED}_summary.json"
IFBENCH_PROMPT_DATASET_SCRATCH="${REPO_ROOT}/classifer_training/artifacts/datasets/ifbench_test_labels_scratch"
IFBENCH_RESPONSE_DATASET_NAME="ifbench_test300_response_l26"

TRAINED_RUN_DIR="${REPO_ROOT}/classifer_training/artifacts/runs/deepscaler/${TRAINED_MODEL_SLUG}/temp${TEMPERATURE}_multisample${NUM_SAMPLES}_val500_test500_vllm_tp4_seed${SEED}"
TRAINED_LABELS_PATH="${REPO_ROOT}/classifer_training/artifacts/labels/deepscaler/${TRAINED_MODEL_SLUG}/deepscaler_val500_test500_vllm_tp4_seed${SEED}_labels.jsonl"
TRAINED_LABELS_SUMMARY="${REPO_ROOT}/classifer_training/artifacts/labels/deepscaler/${TRAINED_MODEL_SLUG}/deepscaler_val500_test500_vllm_tp4_seed${SEED}_summary.json"
TRAINED_PROMPT_DATASET_SCRATCH="${REPO_ROOT}/classifer_training/artifacts/datasets/deepscaler_val500_test500_labels_${TRAINED_MODEL_SLUG}_scratch"
TRAINED_RESPONSE_DATASET_NAME="deepscaler_val500_test500_${TRAINED_MODEL_SLUG}_response_l26"

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
  find "${root}" -name "${pattern}" | head -n 1
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

echo "[start] $(timestamp) transfer gpu chain"

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
  echo "[skip] IFBench base-model generation already exists $(timestamp)"
else
  echo "[stage] IFBench base-model generation $(timestamp)"
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
  run_prompt_hidden_shards "${BASE_MODEL_NAME}" "${IFBENCH_SHARD_DIR}" "ifbench_test" "${BASE_MODEL_SLUG}" "ifbench_prompt_hidden"
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
  run_response_hidden_shards "${BASE_MODEL_NAME}" "${IFBENCH_RUN_DIR}" "${IFBENCH_RESPONSE_DATASET_NAME}" "ifbench_response_hidden"
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
  "$(find_one "${REPO_ROOT}/classifer_training/artifacts/rollout_hidden/${TRAINED_RESPONSE_DATASET_NAME}" 'rollout_hidden_states.shard03of04.pt')" \
  "$(find_one "${REPO_ROOT}/classifer_training/artifacts/rollout_index/${TRAINED_RESPONSE_DATASET_NAME}" 'rollout_index.shard00of04.jsonl')" \
  "$(find_one "${REPO_ROOT}/classifer_training/artifacts/rollout_index/${TRAINED_RESPONSE_DATASET_NAME}" 'rollout_index.shard03of04.jsonl')"; then
  echo "[skip] DeepScaleR trained-model response hidden already exists $(timestamp)"
else
  echo "[stage] DeepScaleR trained-model response hidden extraction $(timestamp)"
  run_response_hidden_shards "${TRAINED_MODEL_NAME}" "${TRAINED_RUN_DIR}" "${TRAINED_RESPONSE_DATASET_NAME}" "deepscaler_trained_response_hidden"
fi

echo "[done] $(timestamp) transfer gpu chain"
