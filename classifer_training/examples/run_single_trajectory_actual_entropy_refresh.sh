#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data2/jongwonlim/verl/yoonho/verl}"
PYTHON_BIN="${PYTHON_BIN:-/home/jongwonlim/anaconda3/bin/python}"
MODEL_NAME="${MODEL_NAME:-/data2/sangjunsong/.cache/hf_hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}"
CLEAN_GPU_IDS="${CLEAN_GPU_IDS:-3 1}"
WEAK_GPU_IDS="${WEAK_GPU_IDS:-3 1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_BATCH_TOKENS="${MAX_BATCH_TOKENS:-16000}"
OVERWRITE="${OVERWRITE:-0}"

LOG_ROOT="${ROOT}/classifer_training/artifacts/logs/single_traj_actual_entropy_refresh"
MODEL_OUTPUT_DIR="${ROOT}/classifer_training/artifacts/models/weak4_val20_feature_growth_search/pca16_actual_token_entropy_reduced20"

CLEAN_HIDDEN_FILENAME="finished16_plus_extra2000v2_actual_entropy_l26.pt"
CLEAN_INDEX_FILENAME="finished16_plus_extra2000v2_actual_entropy_l26.jsonl"
WEAK_HIDDEN_FILENAME="weak4_actual_entropy_l26.pt"
WEAK_INDEX_FILENAME="weak4_actual_entropy_l26.jsonl"

mkdir -p "${LOG_ROOT}"
cd "${ROOT}"

OVERWRITE_FLAG=()
if [[ "${OVERWRITE}" == "1" ]]; then
  OVERWRITE_FLAG+=(--overwrite)
fi

launch_extract_phase() {
  local phase_name="$1"
  local dataset_name="$2"
  local hidden_filename="$3"
  local index_filename="$4"
  local gpu_ids_str="$5"
  shift 5
  local -a run_dirs=("$@")

  read -r -a gpu_ids <<<"${gpu_ids_str}"
  local num_shards="${#gpu_ids[@]}"
  if [[ "${num_shards}" -lt 1 ]]; then
    echo "[${phase_name}] no GPU ids provided" >&2
    exit 1
  fi

  local -a pids=()
  for shard_index in "${!gpu_ids[@]}"; do
    local gpu_id="${gpu_ids[$shard_index]}"
    local log_path="${LOG_ROOT}/${phase_name}.shard${shard_index}.log"
    echo "[${phase_name}] launch shard ${shard_index}/${num_shards} on GPU ${gpu_id}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" TOKENIZERS_PARALLELISM=false PYTHONPATH=. \
      "${PYTHON_BIN}" -u -m classifer_training.extract_rollout_hidden_states \
        --model_name_or_path "${MODEL_NAME}" \
        --run_dirs "${run_dirs[@]}" \
        --dataset_name "${dataset_name}" \
        --components think_end_hidden \
        --layers 26 \
        --response_anchor reasoning \
        --hidden_root "${ROOT}/classifer_training/artifacts/rollout_hidden" \
        --index_root "${ROOT}/classifer_training/artifacts/rollout_index" \
        --hidden_filename "${hidden_filename}" \
        --index_filename "${index_filename}" \
        --num_shards "${num_shards}" \
        --shard_index "${shard_index}" \
        --batch_size "${BATCH_SIZE}" \
        --max_batch_tokens "${MAX_BATCH_TOKENS}" \
        --trust_remote_code \
        --local_files_only \
        "${OVERWRITE_FLAG[@]}" \
        > "${log_path}" 2>&1 &
    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
  echo "[${phase_name}] done"
}

train_model() {
  local -a weak_rollout_index_paths=()
  local -a clean_rollout_hidden_paths=()
  local -a clean_rollout_index_paths=()

  read -r -a weak_gpu_ids <<<"${WEAK_GPU_IDS}"
  read -r -a clean_gpu_ids <<<"${CLEAN_GPU_IDS}"

  for shard_index in "${!weak_gpu_ids[@]}"; do
    weak_rollout_index_paths+=(
      "${ROOT}/classifer_training/artifacts/rollout_index/dapo_math_17k_weak4_think_end_l26/_data2_sangjunsong__cache_hf_hub_models--Qwen--Qwen3-4B-Instruct-2507_snapshots_cdbee75f17c01a7cc42f958dc650907174af0554/${WEAK_INDEX_FILENAME%.jsonl}.shard$(printf '%02d' "${shard_index}")of$(printf '%02d' "${#weak_gpu_ids[@]}").jsonl"
    )
  done

  for shard_index in "${!clean_gpu_ids[@]}"; do
    clean_rollout_hidden_paths+=(
      "${ROOT}/classifer_training/artifacts/rollout_hidden/dapo_math_17k/_data2_sangjunsong__cache_hf_hub_models--Qwen--Qwen3-4B-Instruct-2507_snapshots_cdbee75f17c01a7cc42f958dc650907174af0554/${CLEAN_HIDDEN_FILENAME%.pt}.shard$(printf '%02d' "${shard_index}")of$(printf '%02d' "${#clean_gpu_ids[@]}").pt"
    )
    clean_rollout_index_paths+=(
      "${ROOT}/classifer_training/artifacts/rollout_index/dapo_math_17k/_data2_sangjunsong__cache_hf_hub_models--Qwen--Qwen3-4B-Instruct-2507_snapshots_cdbee75f17c01a7cc42f958dc650907174af0554/${CLEAN_INDEX_FILENAME%.jsonl}.shard$(printf '%02d' "${shard_index}")of$(printf '%02d' "${#clean_gpu_ids[@]}").jsonl"
    )
  done

  PYTHONPATH=. "${PYTHON_BIN}" -u -m classifer_training.train_weak_only_single_rollout_hidden \
    --weak_run_dirs \
      "${ROOT}/classifer_training/artifacts/runs/dapo_math_17k/qwen3_4b_instruct_2507/weak4_runs/0" \
      "${ROOT}/classifer_training/artifacts/runs/dapo_math_17k/qwen3_4b_instruct_2507/weak4_runs/1" \
    --weak_prompt_dataset_dir "${ROOT}/classifer_training/artifacts/datasets/dapo_math_17k_weak4_val20" \
    --weak_labels_path "${ROOT}/classifer_training/artifacts/labels/dapo_math_17k/qwen3_4b_instruct_2507/weak4_labels_val20.jsonl" \
    --weak_prompt_hidden_paths \
      "${ROOT}/classifer_training/artifacts/hidden/dapo_math_17k_weak4_shard0/qwen3_4b_instruct_2507_last6mean/hidden_states.pt" \
      "${ROOT}/classifer_training/artifacts/hidden/dapo_math_17k_weak4_shard1/qwen3_4b_instruct_2507_last6mean/hidden_states.pt" \
      "${ROOT}/classifer_training/artifacts/hidden/dapo_math_17k_weak4_shard2/qwen3_4b_instruct_2507_last6mean/hidden_states.pt" \
      "${ROOT}/classifer_training/artifacts/hidden/dapo_math_17k_weak4_shard3/qwen3_4b_instruct_2507_last6mean/hidden_states.pt" \
    --weak_prompt_index_paths \
      "${ROOT}/classifer_training/artifacts/index/dapo_math_17k_weak4_shard0/qwen3_4b_instruct_2507_last6mean/index.jsonl" \
      "${ROOT}/classifer_training/artifacts/index/dapo_math_17k_weak4_shard1/qwen3_4b_instruct_2507_last6mean/index.jsonl" \
      "${ROOT}/classifer_training/artifacts/index/dapo_math_17k_weak4_shard2/qwen3_4b_instruct_2507_last6mean/index.jsonl" \
      "${ROOT}/classifer_training/artifacts/index/dapo_math_17k_weak4_shard3/qwen3_4b_instruct_2507_last6mean/index.jsonl" \
    --weak_rollout_index_paths "${weak_rollout_index_paths[@]}" \
    --clean_labels_path "${ROOT}/classifer_training/artifacts/labels/dapo_math_17k/qwen3_4b_instruct_2507/sampling_labels_16seeds.jsonl" \
    --clean_prompt_hidden_paths "${ROOT}/classifer_training/artifacts/hidden/dapo_math_17k/qwen3_4b_instruct_2507_last6mean/hidden_states.pt" \
    --clean_prompt_index_paths "${ROOT}/classifer_training/artifacts/index/dapo_math_17k/qwen3_4b_instruct_2507_last6mean/index.jsonl" \
    --clean_rollout_hidden_paths "${clean_rollout_hidden_paths[@]}" \
    --clean_rollout_index_paths "${clean_rollout_index_paths[@]}" \
    --output_dir "${MODEL_OUTPUT_DIR}" \
    --prompt_layer_index 26 \
    --feature_mode prompt_only \
    --prompt_hidden_pca_dim 16 \
    --rollout_scalar_keys \
      output_length think_tokens answer_tokens has_complete_answer has_reasoning_content \
      output_mean_token_entropy reasoning_mean_token_entropy answer_mean_token_entropy \
      output_unique_token_ratio answer_unique_token_ratio output_repetition_ratio reasoning_repetition_ratio duplicate_line_ratio \
    --derived_rollout_scalar_keys \
      think_ratio answer_ratio mean_token_entropy_gap_reasoning_answer unique_gap_reasoning_output \
      repetition_gap_reasoning_output reasoning_x_log_output_length answer_mean_token_entropy_gap_vs_output \
    --single_rollout_strategy first \
    --alphas 100 300 1000 3000 10000 \
    > "${LOG_ROOT}/train_actual_entropy.log" 2>&1
}

declare -a CLEAN_RUN_DIRS=()
for seed in $(seq 1 16); do
  CLEAN_RUN_DIRS+=("${ROOT}/classifer_training/artifacts/runs/dapo_math_17k/qwen3_4b_instruct_2507/temp0.7_seed${seed}")
done
CLEAN_RUN_DIRS+=(
  "${ROOT}/classifer_training/artifacts/runs/dapo_math_17k/qwen3_4b_instruct_2507/temp0.7_multisample4_extra2000_v2_shard0_len12288_bs32_seed1"
  "${ROOT}/classifer_training/artifacts/runs/dapo_math_17k/qwen3_4b_instruct_2507/temp0.7_multisample4_extra2000_v2_shard1_len12288_bs32_seed1"
)

declare -a WEAK_RUN_DIRS=(
  "${ROOT}/classifer_training/artifacts/runs/dapo_math_17k/qwen3_4b_instruct_2507/weak4_runs/0"
  "${ROOT}/classifer_training/artifacts/runs/dapo_math_17k/qwen3_4b_instruct_2507/weak4_runs/1"
)

launch_extract_phase "clean_actual_entropy" "dapo_math_17k" "${CLEAN_HIDDEN_FILENAME}" "${CLEAN_INDEX_FILENAME}" "${CLEAN_GPU_IDS}" "${CLEAN_RUN_DIRS[@]}"
launch_extract_phase "weak_actual_entropy" "dapo_math_17k_weak4_think_end_l26" "${WEAK_HIDDEN_FILENAME}" "${WEAK_INDEX_FILENAME}" "${WEAK_GPU_IDS}" "${WEAK_RUN_DIRS[@]}"
train_model

echo "done"
