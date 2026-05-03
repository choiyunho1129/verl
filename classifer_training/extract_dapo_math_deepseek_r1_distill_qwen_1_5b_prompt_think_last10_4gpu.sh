#!/usr/bin/env bash
# Extract prompt last-10 mean hidden states and rollout think-end last-10 hidden
# states for DAPO-Math with DeepSeek-R1-Distill-Qwen-1.5B.
#
# Defaults:
#   - GPUs: 0,1,2,3
#   - layers: 14:27 (0-indexed, inclusive; middle-to-late layers for 28-layer 1.5B)
#   - prompt component: hidden_last10_mean
#   - rollout components: prompt_hidden, think_end_last10_hidden
#   - DAPO DeepSeek rollout dirs: temp0.7_seed1 ... temp0.7_seed16
#
# Example:
#   bash classifer_training/extract_dapo_math_deepseek_r1_distill_qwen_1_5b_prompt_think_last10_4gpu.sh
#
# Useful overrides:
#   GPU_IDS=0,1 LAYERS=18:27 PYTHON=/path/to/python \
#     bash classifer_training/extract_dapo_math_deepseek_r1_distill_qwen_1_5b_prompt_think_last10_4gpu.sh --overwrite
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON="${PYTHON:-python3}"

MODEL_NAME="${MODEL_NAME:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
MODEL_SLUG="${MODEL_SLUG:-deepseek-ai_DeepSeek-R1-Distill-Qwen-1_5B}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/home/holi_models}"
DATASET_NAME="${DATASET_NAME:-dapo_math_17k}"
PROMPT_SOURCE_PATH="${PROMPT_SOURCE_PATH:-${ROOT}/classifer_training/artifacts/index/dapo_math_17k/${MODEL_SLUG}/index.jsonl}"
PROMPT_SHARDS_DIR="${PROMPT_SHARDS_DIR:-${ROOT}/classifer_training/artifacts/datasets/dapo_math_17k_deepseek_1p5b_prompt_shards4}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/classifer_training/artifacts/runs/dapo_math_17k/${MODEL_SLUG}}"

GPU_IDS_CSV="${GPU_IDS:-0,1,2,3}"
LAYERS="${LAYERS:-14:27}"
PROMPT_LAST_N="${PROMPT_LAST_N:-10}"
PROMPT_BATCH_SIZE="${PROMPT_BATCH_SIZE:-8}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
ROLLOUT_MAX_BATCH_TOKENS="${ROLLOUT_MAX_BATCH_TOKENS:-16000}"
ROLLOUT_COMPONENTS="${ROLLOUT_COMPONENTS:-prompt_hidden think_end_last10_hidden}"
RESPONSE_ANCHOR="${RESPONSE_ANCHOR:-reasoning_or_answer}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"

PROMPT_DATASET_PREFIX="${PROMPT_DATASET_PREFIX:-dapo_math_17k_deepseek_1p5b_prompt_last10_l${LAYERS//:/_}}"
ROLLOUT_DATASET_NAME="${ROLLOUT_DATASET_NAME:-dapo_math_17k_deepseek_1p5b_thinkendlast10_l${LAYERS//:/_}}"
PROMPT_MODEL_SLUG="${PROMPT_MODEL_SLUG:-deepseek_r1_distill_qwen_1_5b_l${LAYERS//:/_}_last${PROMPT_LAST_N}mean}"
ROLLOUT_MODEL_SLUG="${ROLLOUT_MODEL_SLUG:-deepseek_r1_distill_qwen_1_5b_l${LAYERS//:/_}_thinkendlast10}"
LOG_DIR="${LOG_DIR:-${ROOT}/classifer_training/artifacts/logs/dapo_math_deepseek_1p5b_prompt_think_last10_l${LAYERS//:/_}_4gpu}"

OVERWRITE=0
LOCAL_FILES_ONLY=0
SKIP_PROMPT=0
SKIP_ROLLOUT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) shift; ROOT="$1" ;;
    --python) shift; PYTHON="$1" ;;
    --model) shift; MODEL_NAME="$1" ;;
    --model-cache-dir) shift; MODEL_CACHE_DIR="$1" ;;
    --prompt-source-path) shift; PROMPT_SOURCE_PATH="$1" ;;
    --prompt-shards-dir) shift; PROMPT_SHARDS_DIR="$1" ;;
    --run-root) shift; RUN_ROOT="$1" ;;
    --gpu-ids) shift; GPU_IDS_CSV="$1" ;;
    --layers) shift; LAYERS="$1" ;;
    --prompt-batch-size) shift; PROMPT_BATCH_SIZE="$1" ;;
    --rollout-batch-size) shift; ROLLOUT_BATCH_SIZE="$1" ;;
    --rollout-max-batch-tokens) shift; ROLLOUT_MAX_BATCH_TOKENS="$1" ;;
    --rollout-components) shift; ROLLOUT_COMPONENTS="$1" ;;
    --attn-implementation) shift; ATTN_IMPLEMENTATION="$1" ;;
    --torch-dtype) shift; TORCH_DTYPE="$1" ;;
    --overwrite) OVERWRITE=1 ;;
    --local-files-only) LOCAL_FILES_ONLY=1 ;;
    --skip-prompt) SKIP_PROMPT=1 ;;
    --skip-rollout) SKIP_ROLLOUT=1 ;;
    -h|--help)
      sed -n '1,40p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
  shift
done

IFS=',' read -r -a GPU_IDS <<< "$GPU_IDS_CSV"
NUM_SHARDS="${#GPU_IDS[@]}"
if [[ "$NUM_SHARDS" -lt 1 ]]; then
  echo "At least one GPU id is required." >&2
  exit 2
fi

mkdir -p "$LOG_DIR"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Missing required directory: $1" >&2
    exit 1
  fi
}

OVERWRITE_FLAG=()
if [[ "$OVERWRITE" == "1" ]]; then
  OVERWRITE_FLAG+=(--overwrite)
fi

LOCAL_ONLY_FLAG=()
if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
  LOCAL_ONLY_FLAG+=(--local_files_only)
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_OFFLINE=1
fi

CACHE_FLAG=()
if [[ -n "$MODEL_CACHE_DIR" ]]; then
  CACHE_FLAG+=(--cache_dir "$MODEL_CACHE_DIR")
fi

ATTN_FLAG=()
if [[ -n "$ATTN_IMPLEMENTATION" ]]; then
  ATTN_FLAG+=(--attn_implementation "$ATTN_IMPLEMENTATION")
fi

require_file "$PROMPT_SOURCE_PATH"
require_dir "$RUN_ROOT"

prepare_prompt_shards() {
  local needs_prepare=0
  for ((shard=0; shard<NUM_SHARDS; shard++)); do
    if [[ ! -f "${PROMPT_SHARDS_DIR}/shard${shard}.jsonl" ]]; then
      needs_prepare=1
    fi
  done
  if [[ "$OVERWRITE" == "1" || "$needs_prepare" == "1" ]]; then
    log "Preparing ${NUM_SHARDS} prompt shards from ${PROMPT_SOURCE_PATH} under ${PROMPT_SHARDS_DIR}"
    PYTHONPATH="$ROOT" "$PYTHON" - "$PROMPT_SOURCE_PATH" "$PROMPT_SHARDS_DIR" "$NUM_SHARDS" <<'PY'
import json
import sys
from pathlib import Path

source_path = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
num_shards = int(sys.argv[3])
rows = []
with source_path.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

output_dir.mkdir(parents=True, exist_ok=True)
for shard_idx in range(num_shards):
    shard_rows = rows[shard_idx::num_shards]
    out_path = output_dir / f"shard{shard_idx}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in shard_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

summary = {
    "source_path": str(source_path),
    "output_dir": str(output_dir),
    "num_rows_total": len(rows),
    "num_shards": num_shards,
    "shard_sizes": [len(rows[i::num_shards]) for i in range(num_shards)],
}
(output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2), flush=True)
PY
  fi

  for ((shard=0; shard<NUM_SHARDS; shard++)); do
    require_file "${PROMPT_SHARDS_DIR}/shard${shard}.jsonl"
  done
}

RUN_DIRS=()
for seed in $(seq 1 16); do
  run_dir="${RUN_ROOT}/temp0.7_seed${seed}"
  require_file "${run_dir}/all_experiments.jsonl"
  RUN_DIRS+=("$run_dir")
done

run_prompt_shard() {
  local gpu="$1"
  local shard="$2"
  local dataset_shard="${PROMPT_DATASET_PREFIX}_shard${shard}"
  local log_path="${LOG_DIR}/prompt_shard${shard}_gpu${gpu}.log"

  log "[prompt][shard${shard}][gpu${gpu}] extracting layers=${LAYERS} last${PROMPT_LAST_N} -> ${log_path}"
  PYTHONPATH="$ROOT" "$PYTHON" -m classifer_training.extract_hidden_states \
    --input_path "${PROMPT_SHARDS_DIR}/shard${shard}.jsonl" \
    --model_name_or_path "$MODEL_NAME" \
    --dataset_name "$dataset_shard" \
    --model_slug "$PROMPT_MODEL_SLUG" \
    --components hidden \
    --layers "$LAYERS" \
    --last_n_values "$PROMPT_LAST_N" \
    --batch_size "$PROMPT_BATCH_SIZE" \
    --cuda_device "$gpu" \
    --torch_dtype "$TORCH_DTYPE" \
    --hidden_root "${ROOT}/classifer_training/artifacts/hidden" \
    --index_root "${ROOT}/classifer_training/artifacts/index" \
    "${CACHE_FLAG[@]}" \
    "${LOCAL_ONLY_FLAG[@]}" \
    "${ATTN_FLAG[@]}" \
    "${OVERWRITE_FLAG[@]}" \
    > "$log_path" 2>&1
}

run_rollout_shard() {
  local gpu="$1"
  local shard="$2"
  local log_path="${LOG_DIR}/rollout_shard${shard}_gpu${gpu}.log"

  read -r -a rollout_components_array <<< "$ROLLOUT_COMPONENTS"
  log "[rollout][shard${shard}][gpu${gpu}] extracting layers=${LAYERS} components=${ROLLOUT_COMPONENTS} -> ${log_path}"
  PYTHONPATH="$ROOT" "$PYTHON" -m classifer_training.extract_rollout_hidden_states \
    --run_dirs "${RUN_DIRS[@]}" \
    --model_name_or_path "$MODEL_NAME" \
    --dataset_name "$ROLLOUT_DATASET_NAME" \
    --model_slug "$ROLLOUT_MODEL_SLUG" \
    --components "${rollout_components_array[@]}" \
    --layers "$LAYERS" \
    --response_anchor "$RESPONSE_ANCHOR" \
    --num_shards "$NUM_SHARDS" \
    --shard_index "$shard" \
    --batch_size "$ROLLOUT_BATCH_SIZE" \
    --max_batch_tokens "$ROLLOUT_MAX_BATCH_TOKENS" \
    --cuda_device "$gpu" \
    --torch_dtype "$TORCH_DTYPE" \
    --hidden_root "${ROOT}/classifer_training/artifacts/rollout_hidden" \
    --index_root "${ROOT}/classifer_training/artifacts/rollout_index" \
    "${CACHE_FLAG[@]}" \
    "${LOCAL_ONLY_FLAG[@]}" \
    "${ATTN_FLAG[@]}" \
    "${OVERWRITE_FLAG[@]}" \
    > "$log_path" 2>&1
}

run_parallel_stage() {
  local stage="$1"
  local pids=()
  for shard in "${!GPU_IDS[@]}"; do
    gpu="${GPU_IDS[$shard]}"
    if [[ "$stage" == "prompt" ]]; then
      run_prompt_shard "$gpu" "$shard" &
    else
      run_rollout_shard "$gpu" "$shard" &
    fi
    pids+=("$!")
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" != "0" ]]; then
    echo "${stage} extraction failed. Check logs under ${LOG_DIR}" >&2
    exit 1
  fi
}

log "ROOT=${ROOT}"
log "PYTHON=${PYTHON}"
log "MODEL_NAME=${MODEL_NAME}"
log "MODEL_CACHE_DIR=${MODEL_CACHE_DIR}"
log "PROMPT_SOURCE_PATH=${PROMPT_SOURCE_PATH}"
log "PROMPT_SHARDS_DIR=${PROMPT_SHARDS_DIR}"
log "RUN_ROOT=${RUN_ROOT}"
log "GPU_IDS=${GPU_IDS_CSV}"
log "LAYERS=${LAYERS}"
log "LOG_DIR=${LOG_DIR}"

if [[ "$SKIP_PROMPT" != "1" ]]; then
  prepare_prompt_shards
  run_parallel_stage prompt
fi

if [[ "$SKIP_ROLLOUT" != "1" ]]; then
  run_parallel_stage rollout
fi

log "Done."
log "Prompt hidden root: ${ROOT}/classifer_training/artifacts/hidden/*/${PROMPT_MODEL_SLUG}"
log "Rollout hidden root: ${ROOT}/classifer_training/artifacts/rollout_hidden/${ROLLOUT_DATASET_NAME}/${ROLLOUT_MODEL_SLUG}"
