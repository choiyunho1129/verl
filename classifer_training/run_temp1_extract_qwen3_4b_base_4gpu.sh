#!/usr/bin/env bash
# Extract prompt and rollout hidden states for SPO temp1 subset0-4 with base Qwen3-4B.
#
# Default model is Qwen/Qwen3-4B, not Qwen3-4B-Instruct-2507.
# The rollout extractor also writes entropy/logprob scalar features into rollout_index.jsonl.
#
# Typical use on a 4-GPU server:
#   bash classifer_training/run_temp1_extract_qwen3_4b_base_4gpu.sh --gpu-ids 0,1,2,3
#
# Useful overrides:
#   ROOT=/path/to/verl PYTHON=/path/to/python MODEL_NAME=/local/Qwen3-4B \
#   MODEL_CACHE_DIR=/path/to/writable/hf-cache \
#   bash classifer_training/run_temp1_extract_qwen3_4b_base_4gpu.sh --local-files-only
#   bash classifer_training/run_temp1_extract_qwen3_4b_base_4gpu.sh --model-load-path /local/Qwen3-4B
#   bash classifer_training/run_temp1_extract_qwen3_4b_base_4gpu.sh --model-cache-dir /path/to/hf-cache
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON="${PYTHON:-python3}"

DATASET_NAME="${DATASET_NAME:-spo_temp1_subset0to4}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B}"
MODEL_LOAD_NAME_OR_PATH="${MODEL_LOAD_NAME_OR_PATH:-}"
PROMPT_MODEL_SLUG="${PROMPT_MODEL_SLUG:-qwen3_4b_base_l18_35_last5_10_15mean}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3}"

DATASET_DIR_ENV_PROVIDED="${DATASET_DIR+x}"
PROMPT_SHARDS_DIR_ENV_PROVIDED="${PROMPT_SHARDS_DIR+x}"
IMPORTED_ROOT_ENV_PROVIDED="${IMPORTED_ROOT+x}"
LOG_DIR_ENV_PROVIDED="${LOG_DIR+x}"
DATASET_DIR="${DATASET_DIR:-${ROOT}/classifer_training/artifacts/datasets/${DATASET_NAME}}"
PROMPT_SHARDS_DIR="${PROMPT_SHARDS_DIR:-${ROOT}/classifer_training/artifacts/datasets/${DATASET_NAME}_qwen3_4b_base_shards}"
IMPORTED_ROOT="${IMPORTED_ROOT:-${ROOT}/classifer_training/artifacts/runs/${DATASET_NAME}/imported_runs}"
RUN_DIR_NAMES_CSV="${RUN_DIR_NAMES_CSV:-offline_value_estimation_subset_0,offline_value_estimation_subset_1,offline_value_estimation_subset_2,offline_value_estimation_subset_3,offline_value_estimation_subset_4}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-}"

PROMPT_BATCH_SIZE="${PROMPT_BATCH_SIZE:-32}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
ROLLOUT_MAX_BATCH_TOKENS="${ROLLOUT_MAX_BATCH_TOKENS:-24000}"
PROMPT_LAST_N_VALUES_CSV="${PROMPT_LAST_N_VALUES:-5,10,15}"
LAYERS="${LAYERS:-18:35}"
ROLLOUT_COMPONENTS="${ROLLOUT_COMPONENTS:-response_last5_mean_hidden response_last10_mean_hidden response_last15_mean_hidden}"

MIN_FREE_MIB="${MIN_FREE_MIB:-20000}"
MAX_GPU_UTIL="${MAX_GPU_UTIL:-20}"
POLL_SEC="${POLL_SEC:-30}"
SKIP_WAIT="${SKIP_WAIT:-0}"
OVERWRITE="${OVERWRITE:-0}"
SKIP_PROMPT="${SKIP_PROMPT:-0}"
SKIP_ROLLOUT="${SKIP_ROLLOUT:-0}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"

LOG_DIR="${LOG_DIR:-${ROOT}/classifer_training/artifacts/logs/${DATASET_NAME}_qwen3_4b_base_extract_4gpu}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) shift; ROOT="$1" ;;
    --python) shift; PYTHON="$1" ;;
    --model) shift; MODEL_NAME="$1" ;;
    --model-load-path|--load-model-name-or-path) shift; MODEL_LOAD_NAME_OR_PATH="$1" ;;
    --dataset-name) shift; DATASET_NAME="$1" ;;
    --gpu-ids) shift; GPU_IDS_CSV="$1" ;;
    --prompt-batch-size) shift; PROMPT_BATCH_SIZE="$1" ;;
    --prompt-last-n-values) shift; PROMPT_LAST_N_VALUES_CSV="$1" ;;
    --rollout-batch-size) shift; ROLLOUT_BATCH_SIZE="$1" ;;
    --rollout-max-batch-tokens) shift; ROLLOUT_MAX_BATCH_TOKENS="$1" ;;
    --layers) shift; LAYERS="$1" ;;
    --rollout-components) shift; ROLLOUT_COMPONENTS="$1" ;;
    --model-cache-dir) shift; MODEL_CACHE_DIR="$1" ;;
    --overwrite) OVERWRITE=1 ;;
    --skip-prompt) SKIP_PROMPT=1 ;;
    --skip-rollout) SKIP_ROLLOUT=1 ;;
    --skip-wait) SKIP_WAIT=1 ;;
    --trust-remote-code) TRUST_REMOTE_CODE=1 ;;
    --local-files-only) LOCAL_FILES_ONLY=1 ;;
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

if [[ -z "$DATASET_DIR_ENV_PROVIDED" ]]; then
  DATASET_DIR="${ROOT}/classifer_training/artifacts/datasets/${DATASET_NAME}"
fi
if [[ -z "$PROMPT_SHARDS_DIR_ENV_PROVIDED" ]]; then
  PROMPT_SHARDS_DIR="${ROOT}/classifer_training/artifacts/datasets/${DATASET_NAME}_qwen3_4b_base_shards"
fi
if [[ -z "$IMPORTED_ROOT_ENV_PROVIDED" ]]; then
  IMPORTED_ROOT="${ROOT}/classifer_training/artifacts/runs/${DATASET_NAME}/imported_runs"
fi
if [[ -z "$LOG_DIR_ENV_PROVIDED" ]]; then
  LOG_DIR="${ROOT}/classifer_training/artifacts/logs/${DATASET_NAME}_qwen3_4b_base_extract_4gpu"
fi
if [[ -z "$MODEL_CACHE_DIR" ]]; then
  MODEL_CACHE_DIR="${ROOT}/classifer_training/artifacts/hf_cache"
fi
if [[ -z "$MODEL_LOAD_NAME_OR_PATH" ]]; then
  DEFAULT_MERGED_MODEL_PATH="${ROOT}/classifer_training/artifacts/models/Qwen_Qwen3-4B_merged_snapshot"
  if [[ "$LOCAL_FILES_ONLY" == "1" && "$MODEL_NAME" == "Qwen/Qwen3-4B" && -f "${DEFAULT_MERGED_MODEL_PATH}/config.json" && -f "${DEFAULT_MERGED_MODEL_PATH}/model.safetensors.index.json" ]]; then
    MODEL_LOAD_NAME_OR_PATH="$DEFAULT_MERGED_MODEL_PATH"
  else
    MODEL_LOAD_NAME_OR_PATH="$MODEL_NAME"
  fi
fi

IFS=',' read -r -a GPU_IDS <<< "$GPU_IDS_CSV"
NUM_SHARDS="${NUM_SHARDS:-${#GPU_IDS[@]}}"
if [[ "${#GPU_IDS[@]}" -lt 1 ]]; then
  echo "At least one GPU id is required." >&2
  exit 2
fi
if [[ "$NUM_SHARDS" -lt 1 ]]; then
  echo "NUM_SHARDS must be at least 1." >&2
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

py_sanitize_model_slug() {
  PYTHONPATH="$ROOT" "$PYTHON" - "$MODEL_NAME" <<'PY'
import sys
from classifer_training.utils import sanitize_name
print(sanitize_name(sys.argv[1]))
PY
}

BASE_MODEL_SLUG="$(py_sanitize_model_slug)"
ROLLOUT_MODEL_SLUG="${ROLLOUT_MODEL_SLUG:-${BASE_MODEL_SLUG}_l18_35_last5_10_15mean}"
IFS=',' read -r -a PROMPT_LAST_N_VALUES <<< "$PROMPT_LAST_N_VALUES_CSV"
PROMPT_LAST_N_VALUES_FLAG=(--last_n_values "${PROMPT_LAST_N_VALUES[@]}")

OVERWRITE_FLAG=()
if [[ "$OVERWRITE" == "1" ]]; then
  OVERWRITE_FLAG+=(--overwrite)
fi

TRUST_FLAG=()
if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  TRUST_FLAG+=(--trust_remote_code)
fi

LOCAL_ONLY_FLAG=()
if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
  LOCAL_ONLY_FLAG+=(--local_files_only)
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_OFFLINE=1
fi

CACHE_DIR_FLAG=(--cache_dir "$MODEL_CACHE_DIR")
mkdir -p "$MODEL_CACHE_DIR"

if [[ "$MODEL_NAME" == *Instruct* || "$MODEL_NAME" == *instruct* ]]; then
  log "WARNING: MODEL_NAME contains 'Instruct': ${MODEL_NAME}"
  log "This script defaults to base Qwen/Qwen3-4B. Continue only if this override is intentional."
fi

require_file "${DATASET_DIR}/train.jsonl"
require_file "${DATASET_DIR}/validation.jsonl"
require_dir "$IMPORTED_ROOT"

IFS=',' read -r -a RUN_DIR_NAMES <<< "$RUN_DIR_NAMES_CSV"
RUN_DIRS=()
for run_name in "${RUN_DIR_NAMES[@]}"; do
  run_dir="${IMPORTED_ROOT}/${run_name}"
  require_file "${run_dir}/all_experiments.jsonl"
  RUN_DIRS+=("$run_dir")
done

wait_for_gpu() {
  local gpu="$1"
  if [[ "$SKIP_WAIT" == "1" ]]; then
    log "[gpu${gpu}] skip wait"
    return 0
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "[gpu${gpu}] nvidia-smi not found; continuing without wait"
    return 0
  fi
  while true; do
    local stats free_mem util
    stats="$(nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits -i "$gpu" | head -n 1 | tr -d ' ')"
    free_mem="${stats%%,*}"
    util="${stats##*,}"
    if [[ -n "$free_mem" && -n "$util" && "$free_mem" -ge "$MIN_FREE_MIB" && "$util" -le "$MAX_GPU_UTIL" ]]; then
      log "[gpu${gpu}] ready: free=${free_mem}MiB util=${util}%"
      return 0
    fi
    log "[gpu${gpu}] waiting: free=${free_mem:-NA}MiB util=${util:-NA}%"
    sleep "$POLL_SEC"
  done
}

prepare_prompt_shards() {
  if [[ "$SKIP_PROMPT" == "1" ]]; then
    return 0
  fi
  if [[ "$OVERWRITE" != "1" && -f "${PROMPT_SHARDS_DIR}/summary.json" ]]; then
    local ok=1
    for ((shard=0; shard<NUM_SHARDS; shard++)); do
      [[ -f "${PROMPT_SHARDS_DIR}/shard${shard}.jsonl" ]] || ok=0
    done
    if [[ "$ok" == "1" ]]; then
      log "Prompt shards already exist: ${PROMPT_SHARDS_DIR}"
      return 0
    fi
  fi
  log "Preparing ${NUM_SHARDS} prompt shards under ${PROMPT_SHARDS_DIR}"
  PYTHONPATH="$ROOT" "$PYTHON" -m classifer_training.prepare_weak4_shards \
    --input_dir "$DATASET_DIR" \
    --output_dir "$PROMPT_SHARDS_DIR" \
    --num_shards "$NUM_SHARDS" \
    "${OVERWRITE_FLAG[@]}"
}

run_prompt_shard() {
  local gpu="$1"
  local shard="$2"
  local dataset_shard="${DATASET_NAME}_shard${shard}"
  local hidden_path="${ROOT}/classifer_training/artifacts/hidden/${dataset_shard}/${PROMPT_MODEL_SLUG}/hidden_states.pt"
  local index_path="${ROOT}/classifer_training/artifacts/index/${dataset_shard}/${PROMPT_MODEL_SLUG}/index.jsonl"
  local log_path="${LOG_DIR}/prompt_shard${shard}_gpu${gpu}.log"

  if [[ "$OVERWRITE" != "1" && -f "$hidden_path" && -f "$index_path" ]]; then
    log "[prompt][shard${shard}][gpu${gpu}] already exists; skipping"
    return 0
  fi

  wait_for_gpu "$gpu"
  log "[prompt][shard${shard}][gpu${gpu}] extracting -> ${log_path}"
  PYTHONPATH="$ROOT" "$PYTHON" -m classifer_training.extract_hidden_states \
    --input_path "${PROMPT_SHARDS_DIR}/shard${shard}.jsonl" \
    --model_name_or_path "$MODEL_NAME" \
    --load_model_name_or_path "$MODEL_LOAD_NAME_OR_PATH" \
    --model_slug "$PROMPT_MODEL_SLUG" \
    --dataset_name "$dataset_shard" \
    --components hidden \
    --layers "$LAYERS" \
    --token_pooling lastn_mean \
    "${PROMPT_LAST_N_VALUES_FLAG[@]}" \
    --batch_size "$PROMPT_BATCH_SIZE" \
    --cuda_device "$gpu" \
    --hidden_root "${ROOT}/classifer_training/artifacts/hidden" \
    --index_root "${ROOT}/classifer_training/artifacts/index" \
    "${TRUST_FLAG[@]}" \
    "${LOCAL_ONLY_FLAG[@]}" \
    "${CACHE_DIR_FLAG[@]}" \
    "${OVERWRITE_FLAG[@]}" \
    > "$log_path" 2>&1
}

shard_suffix() {
  local shard="$1"
  printf "shard%02dof%02d" "$shard" "$NUM_SHARDS"
}

run_rollout_shard() {
  local gpu="$1"
  local shard="$2"
  local suffix
  suffix="$(shard_suffix "$shard")"
  local hidden_path="${ROOT}/classifer_training/artifacts/rollout_hidden/${DATASET_NAME}/${ROLLOUT_MODEL_SLUG}/rollout_hidden_states.${suffix}.pt"
  local index_path="${ROOT}/classifer_training/artifacts/rollout_index/${DATASET_NAME}/${ROLLOUT_MODEL_SLUG}/rollout_index.${suffix}.jsonl"
  local log_path="${LOG_DIR}/rollout_shard${shard}_gpu${gpu}.log"

  if [[ "$OVERWRITE" != "1" && -f "$hidden_path" && -f "$index_path" ]]; then
    log "[rollout][shard${shard}][gpu${gpu}] already exists; skipping"
    return 0
  fi

  wait_for_gpu "$gpu"
  log "[rollout][shard${shard}][gpu${gpu}] extracting hidden + entropy/logprob features -> ${log_path}"
  PYTHONPATH="$ROOT" "$PYTHON" -m classifer_training.extract_rollout_hidden_states \
    --model_name_or_path "$MODEL_NAME" \
    --load_model_name_or_path "$MODEL_LOAD_NAME_OR_PATH" \
    --model_slug "$ROLLOUT_MODEL_SLUG" \
    --run_dirs "${RUN_DIRS[@]}" \
    --dataset_name "$DATASET_NAME" \
    --components $ROLLOUT_COMPONENTS \
    --layers "$LAYERS" \
    --num_shards "$NUM_SHARDS" \
    --shard_index "$shard" \
    --cuda_device "$gpu" \
    --hidden_root "${ROOT}/classifer_training/artifacts/rollout_hidden" \
    --index_root "${ROOT}/classifer_training/artifacts/rollout_index" \
    --batch_size "$ROLLOUT_BATCH_SIZE" \
    --max_batch_tokens "$ROLLOUT_MAX_BATCH_TOKENS" \
    "${TRUST_FLAG[@]}" \
    "${LOCAL_ONLY_FLAG[@]}" \
    "${CACHE_DIR_FLAG[@]}" \
    "${OVERWRITE_FLAG[@]}" \
    > "$log_path" 2>&1
}

run_parallel_shards() {
  local phase="$1"
  local fn="$2"
  local pids=()
  log "Starting ${phase} extraction with ${NUM_SHARDS} shards on GPUs ${GPU_IDS_CSV}"
  for ((shard=0; shard<NUM_SHARDS; shard++)); do
    local gpu="${GPU_IDS[$((shard % ${#GPU_IDS[@]}))]}"
    "$fn" "$gpu" "$shard" &
    pids+=("$!")
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" != "0" ]]; then
    echo "${phase} extraction failed. Check logs under ${LOG_DIR}" >&2
    exit 1
  fi
  log "Finished ${phase} extraction"
}

write_manifest() {
  MANIFEST_PATH="${LOG_DIR}/qwen3_4b_base_hidden_manifest.json" \
  ROOT="$ROOT" \
  DATASET_NAME="$DATASET_NAME" \
  NUM_SHARDS="$NUM_SHARDS" \
  PROMPT_MODEL_SLUG="$PROMPT_MODEL_SLUG" \
  ROLLOUT_MODEL_SLUG="$ROLLOUT_MODEL_SLUG" \
  MODEL_NAME="$MODEL_NAME" \
  LAYERS="$LAYERS" \
  PROMPT_LAST_N_VALUES="$PROMPT_LAST_N_VALUES_CSV" \
  ROLLOUT_COMPONENTS="$ROLLOUT_COMPONENTS" \
  PROMPT_SHARDS_DIR="$PROMPT_SHARDS_DIR" \
  "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["ROOT"])
dataset_name = os.environ["DATASET_NAME"]
num_shards = int(os.environ["NUM_SHARDS"])
prompt_model_slug = os.environ["PROMPT_MODEL_SLUG"]
rollout_model_slug = os.environ["ROLLOUT_MODEL_SLUG"]
prompt_shards_dir = Path(os.environ["PROMPT_SHARDS_DIR"])
manifest_path = Path(os.environ["MANIFEST_PATH"])

prompt_hidden_paths = []
prompt_index_paths = []
rollout_hidden_paths = []
rollout_index_paths = []
prompt_shard_paths = []

for shard in range(num_shards):
    dataset_shard = f"{dataset_name}_shard{shard}"
    prompt_shard_paths.append(str((prompt_shards_dir / f"shard{shard}.jsonl").resolve()))
    prompt_hidden_paths.append(str((root / "classifer_training/artifacts/hidden" / dataset_shard / prompt_model_slug / "hidden_states.pt").resolve()))
    prompt_index_paths.append(str((root / "classifer_training/artifacts/index" / dataset_shard / prompt_model_slug / "index.jsonl").resolve()))
    suffix = f"shard{shard:02d}of{num_shards:02d}"
    rollout_hidden_paths.append(str((root / "classifer_training/artifacts/rollout_hidden" / dataset_name / rollout_model_slug / f"rollout_hidden_states.{suffix}.pt").resolve()))
    rollout_index_paths.append(str((root / "classifer_training/artifacts/rollout_index" / dataset_name / rollout_model_slug / f"rollout_index.{suffix}.jsonl").resolve()))

manifest = {
    "dataset_name": dataset_name,
    "model_name_or_path": os.environ["MODEL_NAME"],
    "prompt_model_slug": prompt_model_slug,
    "rollout_model_slug": rollout_model_slug,
    "num_shards": num_shards,
    "selected_layers": os.environ["LAYERS"],
    "prompt_last_n_values": os.environ["PROMPT_LAST_N_VALUES"],
    "rollout_components": os.environ["ROLLOUT_COMPONENTS"].split(),
    "prompt_shard_paths": prompt_shard_paths,
    "prompt_hidden_paths": prompt_hidden_paths,
    "prompt_index_paths": prompt_index_paths,
    "rollout_hidden_paths": rollout_hidden_paths,
    "rollout_index_paths": rollout_index_paths,
    "rollout_index_contains_entropy_features": True,
    "entropy_feature_examples": [
        "rollout_features.output_mean_token_entropy",
        "rollout_features.reasoning_mean_token_entropy",
        "rollout_features.answer_mean_token_entropy",
        "rollout_features.output_last_token_entropy",
        "rollout_features.reasoning_last_token_entropy",
        "rollout_features.answer_last_token_entropy",
    ],
}
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(manifest_path)
PY
}

log "ROOT=${ROOT}"
log "PYTHON=${PYTHON}"
log "MODEL_NAME=${MODEL_NAME}"
log "MODEL_LOAD_NAME_OR_PATH=${MODEL_LOAD_NAME_OR_PATH}"
log "MODEL_CACHE_DIR=${MODEL_CACHE_DIR}"
log "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
log "PROMPT_MODEL_SLUG=${PROMPT_MODEL_SLUG}"
log "ROLLOUT_MODEL_SLUG=${ROLLOUT_MODEL_SLUG}"
log "DATASET_NAME=${DATASET_NAME}"
log "NUM_SHARDS=${NUM_SHARDS}"
log "LAYERS=${LAYERS}"
log "PROMPT_LAST_N_VALUES=${PROMPT_LAST_N_VALUES_CSV}"
log "ROLLOUT_COMPONENTS=${ROLLOUT_COMPONENTS}"
log "LOG_DIR=${LOG_DIR}"

prepare_prompt_shards

if [[ "$SKIP_PROMPT" != "1" ]]; then
  run_parallel_shards "prompt hidden" run_prompt_shard
else
  log "Skipping prompt hidden extraction"
fi

if [[ "$SKIP_ROLLOUT" != "1" ]]; then
  run_parallel_shards "rollout hidden" run_rollout_shard
else
  log "Skipping rollout hidden extraction"
fi

write_manifest
log "Done. Manifest: ${LOG_DIR}/qwen3_4b_base_hidden_manifest.json"
