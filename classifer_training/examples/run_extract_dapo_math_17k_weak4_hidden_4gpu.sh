#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/jongwonlim/verl/yoonho/verl}"
PYTHON_BIN="${PYTHON_BIN:-/home/jongwonlim/anaconda3/envs/CB/bin/python}"
MODEL_NAME="${MODEL_NAME:-/data2/sangjunsong/.cache/hf_hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}"
MODEL_SLUG="${MODEL_SLUG:-qwen3_4b_instruct_2507}"
WEAK4_DIR="${WEAK4_DIR:-${ROOT}/classifer_training/artifacts/datasets/dapo_math_17k_weak4}"
SHARD_DIR="${SHARD_DIR:-${ROOT}/classifer_training/artifacts/datasets/dapo_math_17k_weak4_shards}"
LOG_DIR="${LOG_DIR:-${ROOT}/classifer_training/artifacts/logs/dapo_math_17k_weak4_hidden}"
NUM_SHARDS="${NUM_SHARDS:-4}"
HIDDEN_BATCH_SIZE="${HIDDEN_BATCH_SIZE:-16}"
MIN_FREE_MIB="${MIN_FREE_MIB:-24000}"
MAX_GPU_UTIL="${MAX_GPU_UTIL:-20}"
POLL_SEC="${POLL_SEC:-30}"
OVERWRITE="${OVERWRITE:-0}"
SKIP_WAIT="${SKIP_WAIT:-0}"

mkdir -p "${LOG_DIR}"

OVERWRITE_FLAG=()
if [ "${OVERWRITE}" = "1" ]; then
  OVERWRITE_FLAG+=(--overwrite)
fi

prepare_shards() {
  if [ "${OVERWRITE}" != "1" ] && [ -f "${SHARD_DIR}/summary.json" ] && [ -f "${SHARD_DIR}/shard0.jsonl" ] && [ -f "${SHARD_DIR}/shard1.jsonl" ] && [ -f "${SHARD_DIR}/shard2.jsonl" ] && [ -f "${SHARD_DIR}/shard3.jsonl" ]; then
    echo "[skip] weak4 shard dataset already exists: ${SHARD_DIR}"
    return
  fi
  "${PYTHON_BIN}" -m classifer_training.prepare_weak4_shards \
    --input_dir "${WEAK4_DIR}" \
    --output_dir "${SHARD_DIR}" \
    --num_shards "${NUM_SHARDS}" \
    "${OVERWRITE_FLAG[@]}"
}

wait_for_gpu() {
  local gpu="$1"
  if [ "${SKIP_WAIT}" = "1" ]; then
    echo "[gpu${gpu}] skip wait"
    return 0
  fi
  while true; do
    local stats
    stats="$(nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits -i "${gpu}" | head -n 1 | tr -d ' ')"
    local free_mem="${stats%%,*}"
    local util="${stats##*,}"
    if [ -n "${free_mem}" ] && [ -n "${util}" ] && [ "${free_mem}" -ge "${MIN_FREE_MIB}" ] && [ "${util}" -le "${MAX_GPU_UTIL}" ]; then
      echo "[gpu${gpu}] ready: free=${free_mem}MiB util=${util}%"
      break
    fi
    echo "[gpu${gpu}] waiting: free=${free_mem:-NA}MiB util=${util:-NA}%"
    sleep "${POLL_SEC}"
  done
}

run_extract() {
  local gpu="$1"
  local shard="$2"
  local shard_file="${SHARD_DIR}/shard${shard}.jsonl"
  local ds_name="dapo_math_17k_weak4_shard${shard}"
  local raw_hidden="${ROOT}/classifer_training/artifacts/hidden/${ds_name}/${MODEL_SLUG}/hidden_states.pt"
  local raw_index="${ROOT}/classifer_training/artifacts/index/${ds_name}/${MODEL_SLUG}/index.jsonl"
  local last6_hidden="${ROOT}/classifer_training/artifacts/hidden/${ds_name}/${MODEL_SLUG}_last6mean/hidden_states.pt"
  local last6_index="${ROOT}/classifer_training/artifacts/index/${ds_name}/${MODEL_SLUG}_last6mean/index.jsonl"

  wait_for_gpu "${gpu}" | tee -a "${LOG_DIR}/gpu${gpu}_wait.log"

  if [ -f "${raw_hidden}" ] && [ -f "${raw_index}" ] && [ "${OVERWRITE}" != "1" ]; then
    echo "[skip][gpu${gpu}][shard${shard}] raw hidden exists"
  else
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONNOUSERSITE=1 "${PYTHON_BIN}" -m classifer_training.extract_hidden_states \
      --model_name_or_path "${MODEL_NAME}" \
      --input_path "${shard_file}" \
      --dataset_name "${ds_name}" \
      --model_slug "${MODEL_SLUG}" \
      --components hidden \
      --token_pooling last \
      --batch_size "${HIDDEN_BATCH_SIZE}" \
      --trust_remote_code \
      "${OVERWRITE_FLAG[@]}" \
      > "${LOG_DIR}/shard${shard}_extract_raw.log" 2>&1
  fi

  if [ -f "${last6_hidden}" ] && [ -f "${last6_index}" ] && [ "${OVERWRITE}" != "1" ]; then
    echo "[skip][gpu${gpu}][shard${shard}] last6 hidden exists"
  else
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONNOUSERSITE=1 "${PYTHON_BIN}" -m classifer_training.extract_hidden_states \
      --model_name_or_path "${MODEL_NAME}" \
      --input_path "${shard_file}" \
      --dataset_name "${ds_name}" \
      --model_slug "${MODEL_SLUG}_last6mean" \
      --components hidden \
      --token_pooling lastn_mean \
      --last_n 6 \
      --batch_size "${HIDDEN_BATCH_SIZE}" \
      --trust_remote_code \
      "${OVERWRITE_FLAG[@]}" \
      > "${LOG_DIR}/shard${shard}_extract_last6.log" 2>&1
  fi
}

write_manifest() {
  MANIFEST_PATH="${LOG_DIR}/weak4_hidden_manifest.json" ROOT="${ROOT}" MODEL_SLUG="${MODEL_SLUG}" SHARD_DIR="${SHARD_DIR}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["ROOT"])
model_slug = os.environ["MODEL_SLUG"]
shard_dir = Path(os.environ["SHARD_DIR"])
manifest_path = Path(os.environ["MANIFEST_PATH"])
entries = []
for shard_path in sorted(shard_dir.glob("shard*.jsonl")):
    shard_name = shard_path.stem
    dataset_name = f"dapo_math_17k_weak4_{shard_name}"
    entries.append(
        {
            "dataset_name": dataset_name,
            "input_path": str(shard_path.resolve()),
            "raw_hidden_path": str((root / "classifer_training/artifacts/hidden" / dataset_name / model_slug / "hidden_states.pt").resolve()),
            "raw_index_path": str((root / "classifer_training/artifacts/index" / dataset_name / model_slug / "index.jsonl").resolve()),
            "last6_hidden_path": str((root / "classifer_training/artifacts/hidden" / dataset_name / f"{model_slug}_last6mean" / "hidden_states.pt").resolve()),
            "last6_index_path": str((root / "classifer_training/artifacts/index" / dataset_name / f"{model_slug}_last6mean" / "index.jsonl").resolve()),
        }
    )
manifest = {"dataset_name": "dapo_math_17k_weak4", "num_shards": len(entries), "entries": entries}
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(manifest_path)
PY
}

prepare_shards

run_extract 0 0 &
PID0=$!
run_extract 1 1 &
PID1=$!
run_extract 2 2 &
PID2=$!
run_extract 3 3 &
PID3=$!

wait "${PID0}" "${PID1}" "${PID2}" "${PID3}"
write_manifest
echo "done"
