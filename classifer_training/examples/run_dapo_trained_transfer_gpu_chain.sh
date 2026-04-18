#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/jongwonlim/anaconda3/envs/CB/bin/python}"
TRAINED_MODEL_NAME="${TRAINED_MODEL_NAME:-jaygala24/Qwen3-4B-GRPO-math-reasoning}"
TRAINED_MODEL_SLUG="${TRAINED_MODEL_SLUG:-jaygala24_Qwen3-4B-GRPO-math-reasoning}"
GPU_IDS="${GPU_IDS:-0 1 2 3}"
NUM_SAMPLES="${NUM_SAMPLES:-4}"
SEED="${SEED:-1}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-64}"
EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE:-4}"
MAX_BATCH_TOKENS="${MAX_BATCH_TOKENS:-16000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"
RUN_SUFFIX="${RUN_SUFFIX:-temp${TEMPERATURE}_topp${TOP_P}_topk${TOP_K}_multisample${NUM_SAMPLES}_vllm_tp4_seed${SEED}}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/classifer_training/artifacts/logs/dapo_trained_transfer_gpu_chain_${RUN_SUFFIX}}"

DAPO_TEST_INPUT="${DAPO_TEST_INPUT:-${REPO_ROOT}/classifer_training/deprecated/artifacts/datasets/dapo_math_17k/test.jsonl}"
DAPO_TRAIN_INPUT="${DAPO_TRAIN_INPUT:-${REPO_ROOT}/classifer_training/deprecated/artifacts/datasets/dapo_math_17k/train.jsonl}"
DAPO_TEST_SHARDS="${DAPO_TEST_SHARDS:-${REPO_ROOT}/classifer_training/artifacts/datasets/dapo_math_17k_test_shards4}"
DAPO_TRAIN_SHARDS="${DAPO_TRAIN_SHARDS:-${REPO_ROOT}/classifer_training/artifacts/datasets/dapo_math_17k_train_shards4}"

export PYTHONPATH="${REPO_ROOT}"
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p "${LOG_ROOT}"
PIPELINE_LOG="${LOG_ROOT}/pipeline.log"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

read -r GPU0 GPU1 GPU2 GPU3 <<<"${GPU_IDS}"

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

prepare_shards() {
  local input_path="$1"
  local output_dir="$2"
  local dataset_name="$3"
  local num_shards="$4"
  mkdir -p "${output_dir}"
  "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
input_path = Path(${input_path@Q})
output_dir = Path(${output_dir@Q})
dataset_name = ${dataset_name@Q}
num_shards = int(${num_shards@Q})
rows = []
with input_path.open() as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))
shards = [[] for _ in range(num_shards)]
for idx, row in enumerate(rows):
    shards[idx % num_shards].append(row)
for shard_idx, shard_rows in enumerate(shards):
    with (output_dir / f"shard{shard_idx}.jsonl").open("w") as f:
        for row in shard_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\\n")
with (output_dir / "summary.json").open("w") as f:
    json.dump({"dataset_name": dataset_name, "num_rows_total": len(rows), "num_shards": num_shards, "shard_sizes": [len(s) for s in shards]}, f, indent=2)
PY
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
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" -u -m classifer_training.extract_hidden_states \
      --model_name_or_path "${model_name}" \
      --input_path "${shard_dir}/shard${shard_idx}.jsonl" \
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

run_split_pipeline() {
  local split_name="$1"
  local input_path="$2"
  local shard_dir="$3"
  local dataset_name="dapo_math_17k_${split_name}"
  local run_dir="${REPO_ROOT}/classifer_training/artifacts/runs/dapo_math_17k/${TRAINED_MODEL_SLUG}/${RUN_SUFFIX}_${split_name}"
  local labels_path="${REPO_ROOT}/classifer_training/artifacts/labels/dapo_math_17k/${TRAINED_MODEL_SLUG}/${dataset_name}_${RUN_SUFFIX}_labels.jsonl"
  local labels_summary="${REPO_ROOT}/classifer_training/artifacts/labels/dapo_math_17k/${TRAINED_MODEL_SLUG}/${dataset_name}_${RUN_SUFFIX}_summary.json"
  local prompt_dataset_scratch="${REPO_ROOT}/classifer_training/artifacts/datasets/${dataset_name}_${RUN_SUFFIX}_${TRAINED_MODEL_SLUG}_labels_scratch"
  local response_dataset_name="${dataset_name}_${TRAINED_MODEL_SLUG}_${RUN_SUFFIX}_response_l26"

  if all_exist "${shard_dir}/shard0.jsonl" "${shard_dir}/shard1.jsonl" "${shard_dir}/shard2.jsonl" "${shard_dir}/shard3.jsonl"; then
    echo "[skip] ${dataset_name} shards already exist $(timestamp)"
  else
    echo "[stage] ${dataset_name} shard prep $(timestamp)"
    prepare_shards "${input_path}" "${shard_dir}" "${dataset_name}" 4
  fi

  if all_exist "${run_dir}/all_experiments.jsonl" "${run_dir}/evaluation_results.jsonl"; then
    echo "[skip] ${dataset_name} generation already exists $(timestamp)"
  else
    echo "[stage] ${dataset_name} generation $(timestamp)"
    "${PYTHON_BIN}" -u -m classifer_training.sample \
      --model_name_or_path "${TRAINED_MODEL_NAME}" \
      --input_path "${input_path}" \
      --dataset_name "${dataset_name}" \
      --output_dir "${run_dir}" \
      --backend vllm \
      --grader math_verify \
      --temperature "${TEMPERATURE}" \
      --top_p "${TOP_P}" \
      --top_k "${TOP_K}" \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      --batch_size "${GEN_BATCH_SIZE}" \
      --seed "${SEED}" \
      --num_samples "${NUM_SAMPLES}" \
      --tensor_parallel_size 4 \
      --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
      --trust_remote_code \
      --overwrite
  fi

  if all_exist "${labels_path}" "${labels_summary}"; then
    echo "[skip] ${dataset_name} labels already exist $(timestamp)"
  else
    echo "[stage] ${dataset_name} labels $(timestamp)"
    "${PYTHON_BIN}" -u -m classifer_training.build_weak_prompt_dataset_and_labels \
      --run_dirs "${run_dir}" \
      --prompt_dataset_dir "${prompt_dataset_scratch}" \
      --labels_path "${labels_path}" \
      --summary_path "${labels_summary}"
  fi

  if all_exist \
    "${REPO_ROOT}/classifer_training/artifacts/hidden/${dataset_name}_shard0/${TRAINED_MODEL_SLUG}/hidden_states.pt" \
    "${REPO_ROOT}/classifer_training/artifacts/hidden/${dataset_name}_shard1/${TRAINED_MODEL_SLUG}/hidden_states.pt" \
    "${REPO_ROOT}/classifer_training/artifacts/hidden/${dataset_name}_shard2/${TRAINED_MODEL_SLUG}/hidden_states.pt" \
    "${REPO_ROOT}/classifer_training/artifacts/hidden/${dataset_name}_shard3/${TRAINED_MODEL_SLUG}/hidden_states.pt"; then
    echo "[skip] ${dataset_name} prompt hidden already exists $(timestamp)"
  else
    echo "[stage] ${dataset_name} prompt hidden extraction $(timestamp)"
    run_prompt_hidden_shards "${TRAINED_MODEL_NAME}" "${shard_dir}" "${dataset_name}" "${TRAINED_MODEL_SLUG}" "${dataset_name}_prompt_hidden"
  fi

  if all_exist \
    "$(find_one "${REPO_ROOT}/classifer_training/artifacts/rollout_hidden/${response_dataset_name}" 'rollout_hidden_states.shard00of04.pt')" \
    "$(find_one "${REPO_ROOT}/classifer_training/artifacts/rollout_hidden/${response_dataset_name}" 'rollout_hidden_states.shard03of04.pt')" \
    "$(find_one "${REPO_ROOT}/classifer_training/artifacts/rollout_index/${response_dataset_name}" 'rollout_index.shard00of04.jsonl')" \
    "$(find_one "${REPO_ROOT}/classifer_training/artifacts/rollout_index/${response_dataset_name}" 'rollout_index.shard03of04.jsonl')"; then
    echo "[skip] ${dataset_name} response hidden already exists $(timestamp)"
  else
    echo "[stage] ${dataset_name} response hidden extraction $(timestamp)"
    run_response_hidden_shards "${TRAINED_MODEL_NAME}" "${run_dir}" "${response_dataset_name}" "${dataset_name}_response_hidden"
  fi
}

echo "[start] $(timestamp) dapo trained transfer gpu chain"
run_split_pipeline "test" "${DAPO_TEST_INPUT}" "${DAPO_TEST_SHARDS}"
run_split_pipeline "train" "${DAPO_TRAIN_INPUT}" "${DAPO_TRAIN_SHARDS}"
echo "[done] $(timestamp) dapo trained transfer gpu chain"
