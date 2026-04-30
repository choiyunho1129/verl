#!/usr/bin/env bash
# IF_multi_constraints_upto5 generation + official IFEvalG scoring +
# prompt/rollout hidden-state extraction for Qwen3-4B.
#
# Defaults mirror the custom DeepScaleR layout:
#   - train prompts: 4096, rollouts per prompt: 2
#   - validation prompts: 2048, rollouts per prompt: 16
#   - split-specific generation shards are kept as reusable run dirs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
if [[ -n "${PYTHON:-}" ]]; then
  PYTHON="${PYTHON}"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON="$(command -v python)"
else
  echo "Could not find python3 or python on PATH. Pass --python /path/to/python." >&2
  exit 1
fi

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B}"
MODEL_LOAD_NAME_OR_PATH="${MODEL_LOAD_NAME_OR_PATH:-}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-}"
DATASETS_CACHE_DIR="${DATASETS_CACHE_DIR:-}"
OPEN_INSTRUCT_ROOT="${OPEN_INSTRUCT_ROOT:-${ROOT}/classifer_training/external/open-instruct}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3}"

DATASET_ID="${DATASET_ID:-allenai/IF_multi_constraints_upto5}"
SOURCE_SPLIT="${SOURCE_SPLIT:-train}"
DATASET_NAME="${DATASET_NAME:-if_multi_constraints_upto5}"
TRAIN_PROMPTS="${TRAIN_PROMPTS:-4096}"
VALIDATION_PROMPTS="${VALIDATION_PROMPTS:-2048}"
TRAIN_NUM_SAMPLES="${TRAIN_NUM_SAMPLES:-2}"
VALIDATION_NUM_SAMPLES="${VALIDATION_NUM_SAMPLES:-16}"
TRAIN_GENERATION_SHARD_SIZE="${TRAIN_GENERATION_SHARD_SIZE:-512}"
VALIDATION_GENERATION_SHARD_SIZE="${VALIDATION_GENERATION_SHARD_SIZE:-256}"
MAX_VALIDATION_GENERATION_SHARDS="${MAX_VALIDATION_GENERATION_SHARDS:-}"

SEED="${SEED:-1}"
TEMPERATURE="${TEMPERATURE:-1}"
TOP_P="${TOP_P:-1}"
TOP_K="${TOP_K:--1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-64}"
PROMPT_BATCH_SIZE="${PROMPT_BATCH_SIZE:-32}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
ROLLOUT_MAX_BATCH_TOKENS="${ROLLOUT_MAX_BATCH_TOKENS:-24000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"

LAYERS="${LAYERS:-19}"
PROMPT_LAST_N_VALUES_CSV="${PROMPT_LAST_N_VALUES:-10}"
ROLLOUT_COMPONENTS="${ROLLOUT_COMPONENTS:-think_end_last10_hidden}"

MIN_FREE_MIB="${MIN_FREE_MIB:-20000}"
MAX_GPU_UTIL="${MAX_GPU_UTIL:-20}"
POLL_SEC="${POLL_SEC:-30}"
SKIP_WAIT="${SKIP_WAIT:-0}"
OVERWRITE="${OVERWRITE:-0}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-1}"
FILTER_UNSUPPORTED="${FILTER_UNSUPPORTED:-0}"
SKIP_OVERLONG_PROMPTS="${SKIP_OVERLONG_PROMPTS:-1}"

SKIP_PREPARE="${SKIP_PREPARE:-0}"
SKIP_GENERATION="${SKIP_GENERATION:-0}"
SKIP_LABELS="${SKIP_LABELS:-0}"
SKIP_PROMPT="${SKIP_PROMPT:-0}"
SKIP_ROLLOUT="${SKIP_ROLLOUT:-0}"

DATASET_DIR_ENV_PROVIDED="${IF_MULTI_DATASET_DIR+x}"
PROMPT_SHARD_DIR_ENV_PROVIDED="${IF_MULTI_PROMPT_SHARD_DIR+x}"
LOG_ROOT_ENV_PROVIDED="${LOG_ROOT+x}"

usage() {
  cat <<'EOF'
Usage:
  bash classifer_training/run_if_multi_constraints_qwen3_4b_base_4gpu.sh \
    --gpu-ids 0,1,2,3 \
    --python "$(which python)"

Defaults:
  - dataset: allenai/IF_multi_constraints_upto5, source split train
  - train prompts: 4096, train rollouts/prompt: 2
  - validation prompts: 2048, validation rollouts/prompt: 16
  - sampling: temperature=1, top_p=1, top_k=-1
  - hidden extraction: layer 19, prompt last-10 mean, think-end last-10 hidden

Key options:
  --train-prompts 4096
  --validation-prompts 2048
  --train-num-samples 2
  --validation-num-samples 16
  --train-generation-shard-size 512
  --validation-generation-shard-size 256
  --max-validation-generation-shards 4
  --no-skip-overlong-prompts
  --layers 19
  --prompt-last-n-values 10
  --rollout-components "think_end_last10_hidden"
  --skip-generation|--skip-labels|--skip-prompt|--skip-rollout
  --local-files-only --model-cache-dir PATH
  --datasets-cache-dir PATH
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) shift; ROOT="$1" ;;
    --python) shift; PYTHON="$1" ;;
    --model) shift; MODEL_NAME="$1" ;;
    --model-load-path|--load-model-name-or-path) shift; MODEL_LOAD_NAME_OR_PATH="$1" ;;
    --model-cache-dir) shift; MODEL_CACHE_DIR="$1" ;;
    --datasets-cache-dir) shift; DATASETS_CACHE_DIR="$1" ;;
    --open-instruct-root) shift; OPEN_INSTRUCT_ROOT="$1" ;;
    --gpu-ids) shift; GPU_IDS_CSV="$1" ;;
    --dataset-id) shift; DATASET_ID="$1" ;;
    --source-split|--dataset-split) shift; SOURCE_SPLIT="$1" ;;
    --dataset-name) shift; DATASET_NAME="$1" ;;
    --train-prompts) shift; TRAIN_PROMPTS="$1" ;;
    --validation-prompts) shift; VALIDATION_PROMPTS="$1" ;;
    --train-num-samples) shift; TRAIN_NUM_SAMPLES="$1" ;;
    --validation-num-samples) shift; VALIDATION_NUM_SAMPLES="$1" ;;
    --num-samples) shift; TRAIN_NUM_SAMPLES="$1"; VALIDATION_NUM_SAMPLES="$1" ;;
    --train-generation-shard-size) shift; TRAIN_GENERATION_SHARD_SIZE="$1" ;;
    --validation-generation-shard-size) shift; VALIDATION_GENERATION_SHARD_SIZE="$1" ;;
    --max-validation-generation-shards) shift; MAX_VALIDATION_GENERATION_SHARDS="$1" ;;
    --seed) shift; SEED="$1" ;;
    --temperature) shift; TEMPERATURE="$1" ;;
    --top-p) shift; TOP_P="$1" ;;
    --top-k) shift; TOP_K="$1" ;;
    --max-new-tokens) shift; MAX_NEW_TOKENS="$1" ;;
    --gen-batch-size) shift; GEN_BATCH_SIZE="$1" ;;
    --prompt-batch-size) shift; PROMPT_BATCH_SIZE="$1" ;;
    --rollout-batch-size) shift; ROLLOUT_BATCH_SIZE="$1" ;;
    --rollout-max-batch-tokens) shift; ROLLOUT_MAX_BATCH_TOKENS="$1" ;;
    --gpu-memory-utilization) shift; GPU_MEMORY_UTILIZATION="$1" ;;
    --layers) shift; LAYERS="$1" ;;
    --prompt-last-n-values) shift; PROMPT_LAST_N_VALUES_CSV="$1" ;;
    --rollout-components) shift; ROLLOUT_COMPONENTS="$1" ;;
    --if-multi-dataset-dir) shift; IF_MULTI_DATASET_DIR="$1"; DATASET_DIR_ENV_PROVIDED=1 ;;
    --if-multi-prompt-shard-dir|--if-multi-shard-dir) shift; IF_MULTI_PROMPT_SHARD_DIR="$1"; PROMPT_SHARD_DIR_ENV_PROVIDED=1 ;;
    --log-root) shift; LOG_ROOT="$1"; LOG_ROOT_ENV_PROVIDED=1 ;;
    --overwrite) OVERWRITE=1 ;;
    --local-files-only) LOCAL_FILES_ONLY=1 ;;
    --trust-remote-code) TRUST_REMOTE_CODE=1 ;;
    --enforce-eager) ENFORCE_EAGER=1 ;;
    --enable-custom-all-reduce) DISABLE_CUSTOM_ALL_REDUCE=0 ;;
    --filter-unsupported) FILTER_UNSUPPORTED=1 ;;
    --no-skip-overlong-prompts) SKIP_OVERLONG_PROMPTS=0 ;;
    --skip-wait) SKIP_WAIT=1 ;;
    --skip-prepare) SKIP_PREPARE=1 ;;
    --skip-generation) SKIP_GENERATION=1 ;;
    --skip-labels) SKIP_LABELS=1 ;;
    --skip-prompt) SKIP_PROMPT=1 ;;
    --skip-rollout) SKIP_ROLLOUT=1 ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
  shift
done

export PYTHONPATH="${ROOT}"
export OPEN_INSTRUCT_ROOT
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

IFS=',' read -r -a GPU_IDS <<< "$GPU_IDS_CSV"
NUM_SHARDS="${NUM_SHARDS:-${#GPU_IDS[@]}}"
TP_SIZE="${TP_SIZE:-${#GPU_IDS[@]}}"
if [[ "${#GPU_IDS[@]}" -lt 1 ]]; then
  echo "At least one GPU id is required." >&2
  exit 2
fi

if [[ -z "$MODEL_CACHE_DIR" ]]; then
  MODEL_CACHE_DIR="${ROOT}/classifer_training/artifacts/hf_cache"
fi
if [[ -z "$DATASETS_CACHE_DIR" ]]; then
  DATASETS_CACHE_DIR="${ROOT}/classifer_training/artifacts/hf_datasets_cache"
fi
mkdir -p "$MODEL_CACHE_DIR"
mkdir -p "$DATASETS_CACHE_DIR"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$DATASETS_CACHE_DIR}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$MODEL_CACHE_DIR}"

if [[ -z "$MODEL_LOAD_NAME_OR_PATH" ]]; then
  DEFAULT_MERGED_MODEL_PATH="${ROOT}/classifer_training/artifacts/models/Qwen_Qwen3-4B_merged_snapshot"
  if [[ "$LOCAL_FILES_ONLY" == "1" && "$MODEL_NAME" == "Qwen/Qwen3-4B" && -f "${DEFAULT_MERGED_MODEL_PATH}/config.json" ]]; then
    MODEL_LOAD_NAME_OR_PATH="$DEFAULT_MERGED_MODEL_PATH"
  else
    MODEL_LOAD_NAME_OR_PATH="$MODEL_NAME"
  fi
fi

if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_OFFLINE=1
fi

slug_text() {
  local text="$1"
  text="${text//\//_}"
  text="${text//:/_}"
  text="${text//,/_}"
  text="${text// /_}"
  text="${text//[!A-Za-z0-9._-]/}"
  printf '%s' "$text"
}

py_sanitize_model_slug() {
  PYTHONPATH="$ROOT" "$PYTHON" - "$MODEL_NAME" <<'PY'
import sys
from classifer_training.utils import sanitize_name
print(sanitize_name(sys.argv[1]))
PY
}

MODEL_SLUG="${MODEL_SLUG:-$(py_sanitize_model_slug)}"
LAYER_SLUG="$(slug_text "$LAYERS")"
LASTN_SLUG="$(slug_text "$PROMPT_LAST_N_VALUES_CSV")"
DATASET_SLUG="${DATASET_SLUG:-${DATASET_NAME}_train${TRAIN_PROMPTS}_validation${VALIDATION_PROMPTS}_seed${SEED}}"
PROMPT_MODEL_SLUG="${PROMPT_MODEL_SLUG:-qwen3_4b_base_l${LAYER_SLUG}_last${LASTN_SLUG}mean}"
ROLLOUT_MODEL_SLUG="${ROLLOUT_MODEL_SLUG:-${MODEL_SLUG}_l${LAYER_SLUG}_thinkendlast10}"
RUN_SUFFIX="${RUN_SUFFIX:-temp${TEMPERATURE}_topp${TOP_P}_topk${TOP_K}_train${TRAIN_PROMPTS}x${TRAIN_NUM_SAMPLES}_validation${VALIDATION_PROMPTS}x${VALIDATION_NUM_SAMPLES}_vllm_tp${TP_SIZE}_seed${SEED}}"

if [[ -z "$DATASET_DIR_ENV_PROVIDED" ]]; then
  IF_MULTI_DATASET_DIR="${ROOT}/classifer_training/artifacts/datasets/${DATASET_SLUG}"
fi
if [[ -z "$PROMPT_SHARD_DIR_ENV_PROVIDED" ]]; then
  IF_MULTI_PROMPT_SHARD_DIR="${ROOT}/classifer_training/artifacts/datasets/${DATASET_SLUG}_prompt_shards${NUM_SHARDS}"
fi
if [[ -z "$LOG_ROOT_ENV_PROVIDED" ]]; then
  LOG_ROOT="${ROOT}/classifer_training/artifacts/logs/qwen3_4b_base_${DATASET_SLUG}_${RUN_SUFFIX}"
fi

RUN_ROOT="${ROOT}/classifer_training/artifacts/runs/if_multi_constraints/${MODEL_SLUG}/${RUN_SUFFIX}"
TRAIN_RUN_ROOT="${RUN_ROOT}/train_runs"
VALIDATION_RUN_ROOT="${RUN_ROOT}/validation_runs"
LABELS_PATH="${ROOT}/classifer_training/artifacts/labels/if_multi_constraints/${MODEL_SLUG}/${DATASET_SLUG}_${RUN_SUFFIX}_labels.jsonl"
LABELS_SUMMARY="${ROOT}/classifer_training/artifacts/labels/if_multi_constraints/${MODEL_SLUG}/${DATASET_SLUG}_${RUN_SUFFIX}_summary.json"
LABELS_SCRATCH="${ROOT}/classifer_training/artifacts/datasets/${DATASET_SLUG}_${RUN_SUFFIX}_${MODEL_SLUG}_labels_scratch"
RESPONSE_DATASET_NAME="${DATASET_SLUG}_${RUN_SUFFIX}_thinkendlast10_l${LAYER_SLUG}"

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
fi

CACHE_DIR_FLAG=(--cache_dir "$MODEL_CACHE_DIR")

ENFORCE_EAGER_FLAG=()
if [[ "$ENFORCE_EAGER" == "1" ]]; then
  ENFORCE_EAGER_FLAG+=(--enforce_eager)
fi

CUSTOM_ALL_REDUCE_FLAG=()
if [[ "$DISABLE_CUSTOM_ALL_REDUCE" == "1" ]]; then
  CUSTOM_ALL_REDUCE_FLAG+=(--disable_custom_all_reduce)
fi

FILTER_UNSUPPORTED_FLAG=()
if [[ "$FILTER_UNSUPPORTED" == "1" ]]; then
  FILTER_UNSUPPORTED_FLAG+=(--filter-unsupported)
fi

SKIP_OVERLONG_FLAG=()
if [[ "$SKIP_OVERLONG_PROMPTS" == "1" ]]; then
  SKIP_OVERLONG_FLAG+=(--skip_overlong_prompts)
fi

mkdir -p "$LOG_ROOT"
PIPELINE_LOG="${LOG_ROOT}/pipeline.log"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

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

wait_for_all_gpus() {
  local gpu
  for gpu in "${GPU_IDS[@]}"; do
    wait_for_gpu "$gpu"
  done
}

wait_for_pids() {
  local label="$1"
  shift
  local failed=0
  local pid
  for pid in "$@"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" != "0" ]]; then
    echo "${label} failed. Check logs under ${LOG_ROOT}" >&2
    exit 1
  fi
  log "Finished ${label}"
}

collect_run_dirs() {
  local run_root="$1"
  local -n out_ref="$2"
  local max_dirs="${3:-}"
  out_ref=()
  if [[ ! -d "$run_root" ]]; then
    return 0
  fi
  local count=0
  while IFS= read -r dir; do
    if [[ -n "$max_dirs" && "$count" -ge "$max_dirs" ]]; then
      break
    fi
    [[ -n "$dir" ]] && out_ref+=("$dir")
    count=$((count + 1))
  done < <(find "$run_root" -mindepth 1 -maxdepth 1 -type d | sort)
}

shard_suffix() {
  local shard="$1"
  printf "shard%02dof%02d" "$shard" "$NUM_SHARDS"
}

prepare_dataset_if_needed() {
  if [[ "$SKIP_PREPARE" == "1" ]]; then
    log "[prepare] skipping dataset preparation"
    return 0
  fi
  if [[ "$OVERWRITE" != "1" && -f "${IF_MULTI_DATASET_DIR}/summary.json" && -f "${IF_MULTI_PROMPT_SHARD_DIR}/summary.json" ]]; then
    log "[prepare] dataset already exists: ${IF_MULTI_DATASET_DIR}"
    return 0
  fi
  log "[prepare] ${DATASET_ID} -> ${IF_MULTI_DATASET_DIR}"
  PYTHONPATH="$ROOT" "$PYTHON" -u -m classifer_training.prepare_if_multi_constraints_dataset \
    --dataset-id "$DATASET_ID" \
    --split "$SOURCE_SPLIT" \
    --output-dir "$IF_MULTI_DATASET_DIR" \
    --shard-dir "$IF_MULTI_PROMPT_SHARD_DIR" \
    --dataset-name "$DATASET_SLUG" \
    --train-prompts "$TRAIN_PROMPTS" \
    --validation-prompts "$VALIDATION_PROMPTS" \
    --train-generation-shard-size "$TRAIN_GENERATION_SHARD_SIZE" \
    --validation-generation-shard-size "$VALIDATION_GENERATION_SHARD_SIZE" \
    --sample-seed "$SEED" \
    --num-shards "$NUM_SHARDS" \
    --open-instruct-root "$OPEN_INSTRUCT_ROOT" \
    "${FILTER_UNSUPPORTED_FLAG[@]}" \
    "${OVERWRITE_FLAG[@]}"
}

run_generation_shard() {
  local split_label="$1"
  local input_path="$2"
  local run_dir="$3"
  local num_samples="$4"
  local log_path="$5"

  if [[ "$SKIP_GENERATION" == "1" ]]; then
    log "[${split_label}] skipping generation"
    return 0
  fi
  if [[ "$OVERWRITE" != "1" && -f "${run_dir}/all_experiments.jsonl" && -f "${run_dir}/evaluation_results.jsonl" ]]; then
    log "[${split_label}] generation shard already exists: ${run_dir}"
    return 0
  fi

  wait_for_all_gpus
  mkdir -p "$(dirname "$log_path")"
  log "[${split_label}] generation -> ${log_path}"
  CUDA_VISIBLE_DEVICES="$GPU_IDS_CSV" PYTHONPATH="$ROOT" "$PYTHON" -u -m classifer_training.sample \
    --model_name_or_path "$MODEL_LOAD_NAME_OR_PATH" \
    --input_path "$input_path" \
    --dataset_name "$DATASET_SLUG" \
    --output_dir "$run_dir" \
    --backend vllm \
    --grader ifeval \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --top_k "$TOP_K" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --batch_size "$GEN_BATCH_SIZE" \
    --seed "$SEED" \
    --num_samples "$num_samples" \
    --tensor_parallel_size "$TP_SIZE" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    "${SKIP_OVERLONG_FLAG[@]}" \
    "${TRUST_FLAG[@]}" \
    "${ENFORCE_EAGER_FLAG[@]}" \
    "${CUSTOM_ALL_REDUCE_FLAG[@]}" \
    "${OVERWRITE_FLAG[@]}" \
    > "$log_path" 2>&1
}

run_generation_split() {
  local split_name="$1"
  local shard_dir="$2"
  local run_root="$3"
  local num_samples="$4"

  local shard_path shard_base run_dir log_path split_label
  local shard_count=0
  local max_shards=""
  if [[ "$split_name" == "validation" ]]; then
    max_shards="$MAX_VALIDATION_GENERATION_SHARDS"
  fi
  for shard_path in "$shard_dir"/shard*.jsonl; do
    [[ -f "$shard_path" ]] || continue
    if [[ -n "$max_shards" && "$shard_count" -ge "$max_shards" ]]; then
      log "[${split_name}] reached shard limit ${max_shards}; skipping remaining shards"
      break
    fi
    shard_base="$(basename "${shard_path%.jsonl}")"
    run_dir="${run_root}/${split_name}_${shard_base}"
    log_path="${LOG_ROOT}/${split_name}_generation.${shard_base}.log"
    split_label="${split_name}:${shard_base}"
    run_generation_shard "$split_label" "$shard_path" "$run_dir" "$num_samples" "$log_path"
    shard_count=$((shard_count + 1))
  done
}

run_labels() {
  if [[ "$SKIP_LABELS" == "1" ]]; then
    log "[labels] skipping"
    return 0
  fi
  if [[ "$OVERWRITE" != "1" && -f "$LABELS_PATH" && -f "$LABELS_SUMMARY" ]]; then
    log "[labels] already exists: ${LABELS_PATH}"
    return 0
  fi
  mkdir -p "$(dirname "$LABELS_PATH")"

  local train_run_dirs=()
  local validation_run_dirs=()
  collect_run_dirs "$TRAIN_RUN_ROOT" train_run_dirs
  collect_run_dirs "$VALIDATION_RUN_ROOT" validation_run_dirs "$MAX_VALIDATION_GENERATION_SHARDS"
  if [[ "${#train_run_dirs[@]}" -eq 0 || "${#validation_run_dirs[@]}" -eq 0 ]]; then
    echo "Missing train/validation run dirs for label aggregation." >&2
    exit 1
  fi

  local args=(--run_dirs "${train_run_dirs[@]}" "${validation_run_dirs[@]}"
              --prompt_dataset_dir "$LABELS_SCRATCH"
              --labels_path "$LABELS_PATH"
              --summary_path "$LABELS_SUMMARY")
  local run_dir
  args+=(--train_run_dir_names)
  for run_dir in "${train_run_dirs[@]}"; do
    args+=("$(basename "$run_dir")")
  done
  args+=(--validation_run_dir_names)
  for run_dir in "${validation_run_dirs[@]}"; do
    args+=("$(basename "$run_dir")")
  done

  log "[labels] -> ${LOG_ROOT}/labels.log"
  PYTHONPATH="$ROOT" "$PYTHON" -u -m classifer_training.build_weak_prompt_dataset_and_labels \
    "${args[@]}" > "${LOG_ROOT}/labels.log" 2>&1
}

run_prompt_shard() {
  local gpu="$1"
  local shard="$2"
  local dataset_shard="${DATASET_SLUG}_shard${shard}"
  local hidden_path="${ROOT}/classifer_training/artifacts/hidden/${dataset_shard}/${PROMPT_MODEL_SLUG}/hidden_states.pt"
  local index_path="${ROOT}/classifer_training/artifacts/index/${dataset_shard}/${PROMPT_MODEL_SLUG}/index.jsonl"
  local log_path="${LOG_ROOT}/prompt_hidden.shard${shard}.gpu${gpu}.log"
  if [[ "$OVERWRITE" != "1" && -f "$hidden_path" && -f "$index_path" ]]; then
    log "[prompt][shard${shard}][gpu${gpu}] already exists; skipping"
    return 0
  fi
  wait_for_gpu "$gpu"
  log "[prompt][shard${shard}][gpu${gpu}] extracting -> ${log_path}"
  PYTHONPATH="$ROOT" "$PYTHON" -u -m classifer_training.extract_hidden_states \
    --input_path "${IF_MULTI_PROMPT_SHARD_DIR}/shard${shard}.jsonl" \
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

run_parallel_prompt() {
  if [[ "$SKIP_PROMPT" == "1" ]]; then
    log "[prompt] skipping"
    return 0
  fi
  local pids=()
  local shard gpu
  for ((shard=0; shard<NUM_SHARDS; shard++)); do
    gpu="${GPU_IDS[$((shard % ${#GPU_IDS[@]}))]}"
    run_prompt_shard "$gpu" "$shard" &
    pids+=("$!")
  done
  wait_for_pids "prompt hidden extraction" "${pids[@]}"
}

run_rollout_shard() {
  local gpu="$1"
  local shard="$2"
  shift 2
  local run_dirs=("$@")
  local suffix
  suffix="$(shard_suffix "$shard")"
  local hidden_path="${ROOT}/classifer_training/artifacts/rollout_hidden/${RESPONSE_DATASET_NAME}/${ROLLOUT_MODEL_SLUG}/rollout_hidden_states.${suffix}.pt"
  local index_path="${ROOT}/classifer_training/artifacts/rollout_index/${RESPONSE_DATASET_NAME}/${ROLLOUT_MODEL_SLUG}/rollout_index.${suffix}.jsonl"
  local log_path="${LOG_ROOT}/rollout_hidden.shard${shard}.gpu${gpu}.log"
  if [[ "$OVERWRITE" != "1" && -f "$hidden_path" && -f "$index_path" ]]; then
    log "[rollout][shard${shard}][gpu${gpu}] already exists; skipping"
    return 0
  fi
  wait_for_gpu "$gpu"
  log "[rollout][shard${shard}][gpu${gpu}] extracting -> ${log_path}"
  PYTHONPATH="$ROOT" "$PYTHON" -u -m classifer_training.extract_rollout_hidden_states \
    --model_name_or_path "$MODEL_NAME" \
    --load_model_name_or_path "$MODEL_LOAD_NAME_OR_PATH" \
    --model_slug "$ROLLOUT_MODEL_SLUG" \
    --run_dirs "${run_dirs[@]}" \
    --dataset_name "$RESPONSE_DATASET_NAME" \
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

run_parallel_rollout() {
  if [[ "$SKIP_ROLLOUT" == "1" ]]; then
    log "[rollout] skipping"
    return 0
  fi
  local train_run_dirs=()
  local validation_run_dirs=()
  local all_run_dirs=()
  collect_run_dirs "$TRAIN_RUN_ROOT" train_run_dirs
  collect_run_dirs "$VALIDATION_RUN_ROOT" validation_run_dirs "$MAX_VALIDATION_GENERATION_SHARDS"
  all_run_dirs=("${train_run_dirs[@]}" "${validation_run_dirs[@]}")
  if [[ "${#all_run_dirs[@]}" -eq 0 ]]; then
    echo "No generation run dirs available for rollout hidden extraction." >&2
    exit 1
  fi

  local pids=()
  local shard gpu
  for ((shard=0; shard<NUM_SHARDS; shard++)); do
    gpu="${GPU_IDS[$((shard % ${#GPU_IDS[@]}))]}"
    run_rollout_shard "$gpu" "$shard" "${all_run_dirs[@]}" &
    pids+=("$!")
  done
  wait_for_pids "rollout hidden extraction" "${pids[@]}"
}

write_manifest() {
  local train_run_dirs=()
  local validation_run_dirs=()
  collect_run_dirs "$TRAIN_RUN_ROOT" train_run_dirs
  collect_run_dirs "$VALIDATION_RUN_ROOT" validation_run_dirs "$MAX_VALIDATION_GENERATION_SHARDS"
  MANIFEST_PATH="${LOG_ROOT}/qwen3_4b_base_if_multi_constraints_manifest.json" \
  ROOT="$ROOT" \
  MODEL_NAME="$MODEL_NAME" \
  MODEL_LOAD_NAME_OR_PATH="$MODEL_LOAD_NAME_OR_PATH" \
  MODEL_SLUG="$MODEL_SLUG" \
  PROMPT_MODEL_SLUG="$PROMPT_MODEL_SLUG" \
  ROLLOUT_MODEL_SLUG="$ROLLOUT_MODEL_SLUG" \
  NUM_SHARDS="$NUM_SHARDS" \
  LAYERS="$LAYERS" \
  PROMPT_LAST_N_VALUES="$PROMPT_LAST_N_VALUES_CSV" \
  ROLLOUT_COMPONENTS="$ROLLOUT_COMPONENTS" \
  DATASET_SLUG="$DATASET_SLUG" \
  DATASET_DIR="$IF_MULTI_DATASET_DIR" \
  PROMPT_SHARD_DIR="$IF_MULTI_PROMPT_SHARD_DIR" \
  LABELS_PATH="$LABELS_PATH" \
  RESPONSE_DATASET_NAME="$RESPONSE_DATASET_NAME" \
  TRAIN_RUN_DIRS="$(printf '%s\n' "${train_run_dirs[@]}")" \
  VALIDATION_RUN_DIRS="$(printf '%s\n' "${validation_run_dirs[@]}")" \
  "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["ROOT"])
num_shards = int(os.environ["NUM_SHARDS"])
dataset_slug = os.environ["DATASET_SLUG"]
prompt_slug = os.environ["PROMPT_MODEL_SLUG"]
rollout_slug = os.environ["ROLLOUT_MODEL_SLUG"]
response_dataset_name = os.environ["RESPONSE_DATASET_NAME"]

prompt_hidden, prompt_index = [], []
rollout_hidden, rollout_index = [], []
for shard in range(num_shards):
    dataset_shard = f"{dataset_slug}_shard{shard}"
    suffix = f"shard{shard:02d}of{num_shards:02d}"
    prompt_hidden.append(str((root / "classifer_training/artifacts/hidden" / dataset_shard / prompt_slug / "hidden_states.pt").resolve()))
    prompt_index.append(str((root / "classifer_training/artifacts/index" / dataset_shard / prompt_slug / "index.jsonl").resolve()))
    rollout_hidden.append(str((root / "classifer_training/artifacts/rollout_hidden" / response_dataset_name / rollout_slug / f"rollout_hidden_states.{suffix}.pt").resolve()))
    rollout_index.append(str((root / "classifer_training/artifacts/rollout_index" / response_dataset_name / rollout_slug / f"rollout_index.{suffix}.jsonl").resolve()))

manifest = {
    "dataset_slug": dataset_slug,
    "dataset_dir": os.environ["DATASET_DIR"],
    "prompt_shard_dir": os.environ["PROMPT_SHARD_DIR"],
    "model_name_or_path": os.environ["MODEL_NAME"],
    "load_model_name_or_path": os.environ["MODEL_LOAD_NAME_OR_PATH"],
    "model_slug": os.environ["MODEL_SLUG"],
    "prompt_model_slug": prompt_slug,
    "rollout_model_slug": rollout_slug,
    "num_shards": num_shards,
    "selected_layers": os.environ["LAYERS"],
    "prompt_last_n_values": os.environ["PROMPT_LAST_N_VALUES"],
    "rollout_components": os.environ["ROLLOUT_COMPONENTS"].split(),
    "labels_path": os.environ["LABELS_PATH"],
    "train_run_dirs": [line for line in os.environ.get("TRAIN_RUN_DIRS", "").splitlines() if line],
    "validation_run_dirs": [line for line in os.environ.get("VALIDATION_RUN_DIRS", "").splitlines() if line],
    "rollout_dataset_name": response_dataset_name,
    "prompt": {"hidden": prompt_hidden, "index": prompt_index},
    "rollout": {"hidden": rollout_hidden, "index": rollout_index},
}
manifest_path = Path(os.environ["MANIFEST_PATH"])
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(manifest_path)
PY
}

log "ROOT=${ROOT}"
log "PYTHON=${PYTHON}"
log "MODEL_NAME=${MODEL_NAME}"
log "MODEL_LOAD_NAME_OR_PATH=${MODEL_LOAD_NAME_OR_PATH}"
log "MODEL_SLUG=${MODEL_SLUG}"
log "MODEL_CACHE_DIR=${MODEL_CACHE_DIR}"
log "DATASETS_CACHE_DIR=${DATASETS_CACHE_DIR}"
log "OPEN_INSTRUCT_ROOT=${OPEN_INSTRUCT_ROOT}"
log "DATASET_ID=${DATASET_ID}"
log "SOURCE_SPLIT=${SOURCE_SPLIT}"
log "DATASET_SLUG=${DATASET_SLUG}"
log "GPU_IDS=${GPU_IDS_CSV}"
log "NUM_SHARDS=${NUM_SHARDS}"
log "TP_SIZE=${TP_SIZE}"
log "TRAIN_PROMPTS=${TRAIN_PROMPTS}"
log "VALIDATION_PROMPTS=${VALIDATION_PROMPTS}"
log "TRAIN_NUM_SAMPLES=${TRAIN_NUM_SAMPLES}"
log "VALIDATION_NUM_SAMPLES=${VALIDATION_NUM_SAMPLES}"
log "TRAIN_GENERATION_SHARD_SIZE=${TRAIN_GENERATION_SHARD_SIZE}"
log "VALIDATION_GENERATION_SHARD_SIZE=${VALIDATION_GENERATION_SHARD_SIZE}"
log "MAX_VALIDATION_GENERATION_SHARDS=${MAX_VALIDATION_GENERATION_SHARDS:-none}"
log "TEMPERATURE=${TEMPERATURE}"
log "TOP_P=${TOP_P}"
log "TOP_K=${TOP_K}"
log "LAYERS=${LAYERS}"
log "PROMPT_LAST_N_VALUES=${PROMPT_LAST_N_VALUES_CSV}"
log "ROLLOUT_COMPONENTS=${ROLLOUT_COMPONENTS}"
log "RUN_ROOT=${RUN_ROOT}"
log "LOG_ROOT=${LOG_ROOT}"

prepare_dataset_if_needed
run_generation_split "train" "${IF_MULTI_DATASET_DIR}/train_generation_shards" "$TRAIN_RUN_ROOT" "$TRAIN_NUM_SAMPLES"
run_generation_split "validation" "${IF_MULTI_DATASET_DIR}/validation_generation_shards" "$VALIDATION_RUN_ROOT" "$VALIDATION_NUM_SAMPLES"
run_labels
run_parallel_prompt
run_parallel_rollout
write_manifest

log "Done. Manifest: ${LOG_ROOT}/qwen3_4b_base_if_multi_constraints_manifest.json"
