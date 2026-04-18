#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/jongwonlim/verl/yoonho/verl}"
PYTHON_BIN="${PYTHON_BIN:-/home/jongwonlim/anaconda3/envs/CB/bin/python}"
MODEL_NAME="${MODEL_NAME:-/data2/sangjunsong/.cache/hf_hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}"
MODEL_SLUG="${MODEL_SLUG:-qwen3_4b_instruct_2507}"
SHARD_DIR="${SHARD_DIR:-${ROOT}/classifer_training/artifacts/datasets/dapo_math_17k_weak4_shards}"
LOG_DIR="${LOG_DIR:-${ROOT}/classifer_training/artifacts/logs/dapo_math_17k_weak4_hidden}"
HIDDEN_BATCH_SIZE="${HIDDEN_BATCH_SIZE:-8}"
OVERWRITE="${OVERWRITE:-0}"

GPU="${1:?gpu index required}"
SHARD="${2:?shard index required}"

mkdir -p "${LOG_DIR}"
cd "${ROOT}"

OVERWRITE_FLAG=()
if [ "${OVERWRITE}" = "1" ]; then
  OVERWRITE_FLAG+=(--overwrite)
fi

SHARD_FILE="${SHARD_DIR}/shard${SHARD}.jsonl"
DATASET_NAME="dapo_math_17k_weak4_shard${SHARD}"

RAW_HIDDEN="${ROOT}/classifer_training/artifacts/hidden/${DATASET_NAME}/${MODEL_SLUG}/hidden_states.pt"
RAW_INDEX="${ROOT}/classifer_training/artifacts/index/${DATASET_NAME}/${MODEL_SLUG}/index.jsonl"
LAST6_HIDDEN="${ROOT}/classifer_training/artifacts/hidden/${DATASET_NAME}/${MODEL_SLUG}_last6mean/hidden_states.pt"
LAST6_INDEX="${ROOT}/classifer_training/artifacts/index/${DATASET_NAME}/${MODEL_SLUG}_last6mean/index.jsonl"

if [ ! -f "${RAW_HIDDEN}" ] || [ ! -f "${RAW_INDEX}" ] || [ "${OVERWRITE}" = "1" ]; then
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONNOUSERSITE=1 "${PYTHON_BIN}" -u -m classifer_training.extract_hidden_states \
    --model_name_or_path "${MODEL_NAME}" \
    --input_path "${SHARD_FILE}" \
    --dataset_name "${DATASET_NAME}" \
    --model_slug "${MODEL_SLUG}" \
    --components hidden \
    --token_pooling last \
    --batch_size "${HIDDEN_BATCH_SIZE}" \
    --trust_remote_code \
    "${OVERWRITE_FLAG[@]}" \
    > "${LOG_DIR}/shard${SHARD}_extract_raw.log" 2>&1
fi

if [ ! -f "${LAST6_HIDDEN}" ] || [ ! -f "${LAST6_INDEX}" ] || [ "${OVERWRITE}" = "1" ]; then
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONNOUSERSITE=1 "${PYTHON_BIN}" -u -m classifer_training.extract_hidden_states \
    --model_name_or_path "${MODEL_NAME}" \
    --input_path "${SHARD_FILE}" \
    --dataset_name "${DATASET_NAME}" \
    --model_slug "${MODEL_SLUG}_last6mean" \
    --components hidden \
    --token_pooling lastn_mean \
    --last_n 6 \
    --batch_size "${HIDDEN_BATCH_SIZE}" \
    --trust_remote_code \
    "${OVERWRITE_FLAG[@]}" \
    > "${LOG_DIR}/shard${SHARD}_extract_last6.log" 2>&1
fi

echo "done shard${SHARD}"
