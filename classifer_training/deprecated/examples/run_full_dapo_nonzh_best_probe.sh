#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/jongwonlim/verl/yoonho/verl}"
PYTHON_BIN="${PYTHON_BIN:-/home/jongwonlim/anaconda3/envs/CB/bin/python}"
MODEL_NAME="${MODEL_NAME:-/data2/sangjunsong/.cache/hf_hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}"
MODEL_SLUG="${MODEL_SLUG:-qwen3_4b_instruct_2507}"
DATASET_ID="${DATASET_ID:-open-r1/DAPO-Math-17k-Processed}"
NUM_SHARDS="${NUM_SHARDS:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_SAMPLES="${NUM_SAMPLES:-2}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-1.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
# Shared GPUs currently expose about 13 GiB free, so keep the default conservative.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.25}"
FORCE_FILTER="${FORCE_FILTER:-0}"
HIDDEN_BATCH_SIZE="${HIDDEN_BATCH_SIZE:-8}"

FILTER_DIR="${ROOT}/classifer_training/artifacts/datasets/dapo_math_17k_full_nonzh_shards"
RUN_ROOT="${ROOT}/classifer_training/artifacts/runs/dapo_math_17k/${MODEL_SLUG}"
PRED_DIR="${ROOT}/classifer_training/artifacts/predictions/dapo_math_17k/${MODEL_SLUG}_two_rollout_full_nonzh"
LOG_DIR="${PRED_DIR}/logs"
mkdir -p "${LOG_DIR}" "${PRED_DIR}"

if [ "${FORCE_FILTER}" != "1" ] && [ -f "${FILTER_DIR}/summary.json" ] && [ -f "${FILTER_DIR}/shard0.jsonl" ] && [ -f "${FILTER_DIR}/shard1.jsonl" ] && [ -f "${FILTER_DIR}/shard2.jsonl" ] && [ -f "${FILTER_DIR}/shard3.jsonl" ]; then
  echo "[skip] filter/full-shard dataset already exists: ${FILTER_DIR}"
else
  "${PYTHON_BIN}" -m classifer_training.filter_full_dapo_nonchinese \
    --hf_dataset_id "${DATASET_ID}" \
    --hf_split train \
    --output_dir "${FILTER_DIR}" \
    --num_shards "${NUM_SHARDS}" \
    --overwrite
fi

run_shard () {
  local gpu="$1"
  local shard="$2"
  local shard_file="${FILTER_DIR}/shard${shard}.jsonl"
  local ds_name="dapo_math_17k_full_nonzh_shard${shard}"
  local run_dir="${RUN_ROOT}/full_nonzh_shard${shard}_multisample${NUM_SAMPLES}_exact"
  local raw_hidden="${ROOT}/classifer_training/artifacts/hidden/${ds_name}/${MODEL_SLUG}/hidden_states.pt"
  local raw_index="${ROOT}/classifer_training/artifacts/index/${ds_name}/${MODEL_SLUG}/index.jsonl"
  local last6_hidden="${ROOT}/classifer_training/artifacts/hidden/${ds_name}/${MODEL_SLUG}_last6mean/hidden_states.pt"
  local last6_index="${ROOT}/classifer_training/artifacts/index/${ds_name}/${MODEL_SLUG}_last6mean/index.jsonl"
  local exp_jsonl="${run_dir}/all_experiments.jsonl"
  local eval_jsonl="${run_dir}/evaluation_results.jsonl"

  if [ -f "${raw_hidden}" ] && [ -f "${raw_index}" ]; then
    echo "[skip][shard${shard}] raw hidden exists"
  else
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m classifer_training.extract_hidden_states \
      --model_name_or_path "${MODEL_NAME}" \
      --input_path "${shard_file}" \
      --dataset_name "${ds_name}" \
      --model_slug "${MODEL_SLUG}" \
      --token_pooling last \
      --batch_size "${HIDDEN_BATCH_SIZE}" \
      --trust_remote_code \
      --overwrite \
      > "${LOG_DIR}/shard${shard}_extract_raw.log" 2>&1
  fi

  if [ -f "${last6_hidden}" ] && [ -f "${last6_index}" ]; then
    echo "[skip][shard${shard}] last6 hidden exists"
  else
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m classifer_training.extract_hidden_states \
      --model_name_or_path "${MODEL_NAME}" \
      --input_path "${shard_file}" \
      --dataset_name "${ds_name}" \
      --model_slug "${MODEL_SLUG}_last6mean" \
      --token_pooling lastn_mean \
      --last_n 6 \
      --batch_size "${HIDDEN_BATCH_SIZE}" \
      --trust_remote_code \
      --overwrite \
      > "${LOG_DIR}/shard${shard}_extract_last6.log" 2>&1
  fi

  if [ -f "${exp_jsonl}" ] && [ -f "${eval_jsonl}" ]; then
    echo "[skip][shard${shard}] samples already exist"
  else
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    TOKENIZERS_PARALLELISM=false \
    VLLM_TARGET_DEVICE=cuda \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON_BIN}" -u -m classifer_training.sample \
      --model_name_or_path "${MODEL_NAME}" \
      --input_path "${shard_file}" \
      --dataset_name "${ds_name}" \
      --output_dir "${run_dir}" \
      --backend vllm \
      --grader exact \
      --temperature "${TEMPERATURE}" \
      --top_p "${TOP_P}" \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      --batch_size "${BATCH_SIZE}" \
      --seed 1 \
      --num_samples "${NUM_SAMPLES}" \
      --tensor_parallel_size 1 \
      --gpu_memory_utilization "${GPU_MEM_UTIL}" \
      --max_model_len "${MAX_MODEL_LEN}" \
      --trust_remote_code \
      --enforce_eager \
      --overwrite \
      > "${LOG_DIR}/shard${shard}_sample.log" 2>&1
  fi
}

run_shard 0 0 &
PID0=$!
run_shard 1 1 &
PID1=$!
run_shard 2 2 &
PID2=$!
run_shard 3 3 &
PID3=$!

wait "${PID0}" "${PID1}" "${PID2}" "${PID3}"

if [ -f "${PRED_DIR}/predicted_difficulty.jsonl" ]; then
  echo "[skip] final scored predictions already exist: ${PRED_DIR}/predicted_difficulty.jsonl"
else
  "${PYTHON_BIN}" -m classifer_training.score_two_rollout_best_probe \
    --repo_root "${ROOT}" \
    --target_run_dirs \
      "${RUN_ROOT}/full_nonzh_shard0_multisample${NUM_SAMPLES}_exact" \
      "${RUN_ROOT}/full_nonzh_shard1_multisample${NUM_SAMPLES}_exact" \
      "${RUN_ROOT}/full_nonzh_shard2_multisample${NUM_SAMPLES}_exact" \
      "${RUN_ROOT}/full_nonzh_shard3_multisample${NUM_SAMPLES}_exact" \
    --target_raw_hidden_paths \
      "${ROOT}/classifer_training/artifacts/hidden/dapo_math_17k_full_nonzh_shard0/${MODEL_SLUG}/hidden_states.pt" \
      "${ROOT}/classifer_training/artifacts/hidden/dapo_math_17k_full_nonzh_shard1/${MODEL_SLUG}/hidden_states.pt" \
      "${ROOT}/classifer_training/artifacts/hidden/dapo_math_17k_full_nonzh_shard2/${MODEL_SLUG}/hidden_states.pt" \
      "${ROOT}/classifer_training/artifacts/hidden/dapo_math_17k_full_nonzh_shard3/${MODEL_SLUG}/hidden_states.pt" \
    --target_raw_index_paths \
      "${ROOT}/classifer_training/artifacts/index/dapo_math_17k_full_nonzh_shard0/${MODEL_SLUG}/index.jsonl" \
      "${ROOT}/classifer_training/artifacts/index/dapo_math_17k_full_nonzh_shard1/${MODEL_SLUG}/index.jsonl" \
      "${ROOT}/classifer_training/artifacts/index/dapo_math_17k_full_nonzh_shard2/${MODEL_SLUG}/index.jsonl" \
      "${ROOT}/classifer_training/artifacts/index/dapo_math_17k_full_nonzh_shard3/${MODEL_SLUG}/index.jsonl" \
    --target_last6_hidden_paths \
      "${ROOT}/classifer_training/artifacts/hidden/dapo_math_17k_full_nonzh_shard0/${MODEL_SLUG}_last6mean/hidden_states.pt" \
      "${ROOT}/classifer_training/artifacts/hidden/dapo_math_17k_full_nonzh_shard1/${MODEL_SLUG}_last6mean/hidden_states.pt" \
      "${ROOT}/classifer_training/artifacts/hidden/dapo_math_17k_full_nonzh_shard2/${MODEL_SLUG}_last6mean/hidden_states.pt" \
      "${ROOT}/classifer_training/artifacts/hidden/dapo_math_17k_full_nonzh_shard3/${MODEL_SLUG}_last6mean/hidden_states.pt" \
    --target_last6_index_paths \
      "${ROOT}/classifer_training/artifacts/index/dapo_math_17k_full_nonzh_shard0/${MODEL_SLUG}_last6mean/index.jsonl" \
      "${ROOT}/classifer_training/artifacts/index/dapo_math_17k_full_nonzh_shard1/${MODEL_SLUG}_last6mean/index.jsonl" \
      "${ROOT}/classifer_training/artifacts/index/dapo_math_17k_full_nonzh_shard2/${MODEL_SLUG}_last6mean/index.jsonl" \
      "${ROOT}/classifer_training/artifacts/index/dapo_math_17k_full_nonzh_shard3/${MODEL_SLUG}_last6mean/index.jsonl" \
    --model_cache_path "${PRED_DIR}/best_probe_model.joblib" \
    --output_path "${PRED_DIR}/predicted_difficulty.jsonl" \
    > "${LOG_DIR}/score.log" 2>&1
fi

echo "done"
echo "${PRED_DIR}/predicted_difficulty.jsonl"
