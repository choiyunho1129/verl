#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

ROOT="${ROOT:-/home/jongwonlim/verl/yoonho/verl}"
PYTHON_BIN="${PYTHON_BIN:-/home/jongwonlim/anaconda3/envs/CB/bin/python}"
MODEL_NAME="${MODEL_NAME:-/data2/sangjunsong/.cache/hf_hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}"
MODEL_SLUG="${MODEL_SLUG:-qwen3_4b_instruct_2507}"
DATASET_ID="${DATASET_ID:-open-r1/DAPO-Math-17k-Processed}"
NUM_SHARDS="${NUM_SHARDS:-4}"
CHUNK_SIZE="${CHUNK_SIZE:-512}"
NUM_SAMPLES="${NUM_SAMPLES:-2}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-1.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
BATCH_SIZE="${BATCH_SIZE:-64}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
ROLLOUT_HIDDEN_BATCH_SIZE="${ROLLOUT_HIDDEN_BATCH_SIZE:-4}"
ROLLOUT_MAX_BATCH_TOKENS="${ROLLOUT_MAX_BATCH_TOKENS:-24000}"
FORCE_FILTER="${FORCE_FILTER:-0}"
FORCE_CHUNKS="${FORCE_CHUNKS:-0}"

FILTER_DIR="${ROOT}/classifer_training/artifacts/datasets/dapo_math_17k_full_nonzh_shards"
CHUNK_ROOT="${ROOT}/classifer_training/artifacts/datasets/dapo_math_17k_full_nonzh_chunks"
RUN_ROOT="${ROOT}/classifer_training/artifacts/runs/dapo_math_17k/${MODEL_SLUG}"
ROLLOUT_HIDDEN_ROOT="${ROOT}/classifer_training/artifacts/rollout_hidden/dapo_math_17k_full_nonzh"
ROLLOUT_INDEX_ROOT="${ROOT}/classifer_training/artifacts/rollout_index/dapo_math_17k_full_nonzh"
PRED_DIR="${ROOT}/classifer_training/artifacts/predictions/dapo_math_17k/${MODEL_SLUG}_two_rollout_full_nonzh_think_tail_balanced"
LOG_DIR="${PRED_DIR}/logs"
mkdir -p "${LOG_DIR}" "${PRED_DIR}" "${CHUNK_ROOT}"
cd "${ROOT}"

EXTRACT_MODEL_SLUG="$(MODEL_NAME="${MODEL_NAME}" "${PYTHON_BIN}" - <<'PY'
from classifer_training.utils import sanitize_name
import os
print(sanitize_name(os.environ["MODEL_NAME"]))
PY
)"

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

build_chunks() {
  local shard="$1"
  local shard_file="${FILTER_DIR}/shard${shard}.jsonl"
  local chunk_dir="${CHUNK_ROOT}/shard${shard}"
  local marker="${chunk_dir}/chunk_manifest.json"
  if [ "${FORCE_CHUNKS}" != "1" ] && [ -f "${marker}" ]; then
    echo "[skip][shard${shard}] chunk files already exist"
    return
  fi
  rm -rf "${chunk_dir}"
  mkdir -p "${chunk_dir}"
  SHARD_FILE="${shard_file}" CHUNK_DIR="${chunk_dir}" CHUNK_SIZE="${CHUNK_SIZE}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

shard_file = Path(os.environ["SHARD_FILE"])
chunk_dir = Path(os.environ["CHUNK_DIR"])
chunk_size = int(os.environ["CHUNK_SIZE"])
rows = [json.loads(line) for line in shard_file.read_text(encoding="utf-8").splitlines() if line.strip()]
chunks = []
for idx in range(0, len(rows), chunk_size):
    chunk_rows = rows[idx : idx + chunk_size]
    chunk_name = f"chunk{idx // chunk_size:02d}.jsonl"
    chunk_path = chunk_dir / chunk_name
    with chunk_path.open("w", encoding="utf-8") as handle:
        for row in chunk_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    chunks.append({"path": str(chunk_path), "num_rows": len(chunk_rows)})
(chunk_dir / "chunk_manifest.json").write_text(
    json.dumps({"source": str(shard_file), "chunk_size": chunk_size, "num_chunks": len(chunks), "chunks": chunks}, indent=2),
    encoding="utf-8",
)
PY
}

process_shard() {
  local gpu="$1"
  local shard="$2"
  local chunk_dir="${CHUNK_ROOT}/shard${shard}"
  build_chunks "${shard}"
  for chunk_file in "${chunk_dir}"/chunk*.jsonl; do
    local chunk_name
    chunk_name="$(basename "${chunk_file}" .jsonl)"
    local ds_name="dapo_math_17k_full_nonzh_shard${shard}_${chunk_name}"
    local run_dir="${RUN_ROOT}/${ds_name}_multisample${NUM_SAMPLES}_exact"
    local exp_jsonl="${run_dir}/all_experiments.jsonl"
    local rollout_hidden_path="${ROLLOUT_HIDDEN_ROOT}/${ds_name}/${EXTRACT_MODEL_SLUG}/rollout_hidden_states.pt"
    local rollout_index_path="${ROLLOUT_INDEX_ROOT}/${ds_name}/${EXTRACT_MODEL_SLUG}/rollout_index.jsonl"

    if [ -f "${exp_jsonl}" ]; then
      echo "[skip][gpu${gpu}][${ds_name}] samples already exist"
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
        --input_path "${chunk_file}" \
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
        > "${LOG_DIR}/${ds_name}_sample.log" 2>&1
    fi

    if [ -f "${rollout_hidden_path}" ] && [ -f "${rollout_index_path}" ]; then
      echo "[skip][gpu${gpu}][${ds_name}] think_end hidden already exists"
    else
      CUDA_VISIBLE_DEVICES="${gpu}" \
      "${PYTHON_BIN}" -u -m classifer_training.extract_rollout_hidden_states \
        --model_name_or_path "${MODEL_NAME}" \
        --run_dirs "${run_dir}" \
        --dataset_name "${ds_name}" \
        --components think_end_hidden \
        --layers 27 \
        --response_anchor reasoning_or_answer \
        --hidden_root "${ROLLOUT_HIDDEN_ROOT}" \
        --index_root "${ROLLOUT_INDEX_ROOT}" \
        --batch_size "${ROLLOUT_HIDDEN_BATCH_SIZE}" \
        --max_batch_tokens "${ROLLOUT_MAX_BATCH_TOKENS}" \
        --trust_remote_code \
        --overwrite \
        > "${LOG_DIR}/${ds_name}_think_extract.log" 2>&1
    fi
  done
}

process_shard 0 0 &
PID0=$!
process_shard 1 1 &
PID1=$!
process_shard 2 2 &
PID2=$!
process_shard 3 3 &
PID3=$!

wait "${PID0}" "${PID1}" "${PID2}" "${PID3}"

OUTPUT_PATH="${PRED_DIR}/predicted_difficulty.jsonl"
if [ -f "${OUTPUT_PATH}" ]; then
  echo "[skip] final scored predictions already exist: ${OUTPUT_PATH}"
else
  mapfile -t TARGET_RUN_DIRS < <(find "${RUN_ROOT}" -maxdepth 1 -type d -name 'dapo_math_17k_full_nonzh_shard*_chunk*_multisample'"${NUM_SAMPLES}"'_exact' | sort)
  mapfile -t TARGET_ROLLOUT_HIDDEN_PATHS < <(find "${ROLLOUT_HIDDEN_ROOT}" -type f -path "*/${EXTRACT_MODEL_SLUG}/rollout_hidden_states.pt" | sort)
  mapfile -t TARGET_ROLLOUT_INDEX_PATHS < <(find "${ROLLOUT_INDEX_ROOT}" -type f -path "*/${EXTRACT_MODEL_SLUG}/rollout_index.jsonl" | sort)
  if [ "${#TARGET_RUN_DIRS[@]}" -eq 0 ]; then
    echo "No target run dirs found under ${RUN_ROOT}" >&2
    exit 1
  fi
  "${PYTHON_BIN}" -m classifer_training.score_two_rollout_think_tail_balanced_best \
    --repo_root "${ROOT}" \
    --target_run_dirs "${TARGET_RUN_DIRS[@]}" \
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
    --target_rollout_hidden_paths "${TARGET_ROLLOUT_HIDDEN_PATHS[@]}" \
    --target_rollout_index_paths "${TARGET_ROLLOUT_INDEX_PATHS[@]}" \
    --output_path "${OUTPUT_PATH}" \
    > "${LOG_DIR}/score.log" 2>&1
fi

FULL_JSONL="${PRED_DIR}/dapo_math_17k_full_nonzh_with_value.jsonl"
TAGS_JSONL="${PRED_DIR}/dapo_math_17k_full_nonzh_value_tags.jsonl"
FULL_PARQUET="${PRED_DIR}/dapo_math_17k_full_nonzh_with_value.parquet"
if [ -f "${FULL_JSONL}" ] && [ -f "${TAGS_JSONL}" ] && [ -f "${FULL_PARQUET}" ]; then
  echo "[skip] integrated outputs already exist"
else
  "${PYTHON_BIN}" -m classifer_training.integrate_full_nonzh_predictions \
    --shard_paths \
      "${FILTER_DIR}/shard0.jsonl" \
      "${FILTER_DIR}/shard1.jsonl" \
      "${FILTER_DIR}/shard2.jsonl" \
      "${FILTER_DIR}/shard3.jsonl" \
    --predictions_path "${OUTPUT_PATH}" \
    --output_full_jsonl "${FULL_JSONL}" \
    --output_tags_jsonl "${TAGS_JSONL}" \
    --output_full_parquet "${FULL_PARQUET}" \
    > "${LOG_DIR}/integrate.log" 2>&1
fi

echo "done"
echo "${OUTPUT_PATH}"
