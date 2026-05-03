#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON="${PYTHON:-/home/jongwonlim/anaconda3/envs/vllm311/bin/python}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B}"
MODEL_LOAD_NAME_OR_PATH="${MODEL_LOAD_NAME_OR_PATH:-$MODEL_NAME}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/data2/sangjunsong/.cache/transformers}"
GPU_ID="${GPU_ID:-0}"
DATASET_NAME="${DATASET_NAME:-spo_offline_subset0_1_validation_data}"
WORK_ROOT="${WORK_ROOT:-${ROOT}/classifer_training/artifacts/datasets/${DATASET_NAME}}"
SPO_ROOT="${SPO_ROOT:-$(cd "${ROOT}/.." && pwd)/spo}"
SUBSET0_DIR="${SUBSET0_DIR:-${SPO_ROOT}/offline_value_estimation_subset_0}"
SUBSET1_DIR="${SUBSET1_DIR:-${SPO_ROOT}/offline_value_estimation_subset_1}"
PROMPT_MODEL_SLUG="${PROMPT_MODEL_SLUG:-qwen3_4b_base_l19_last10mean}"
ROLLOUT_MODEL_SLUG="${ROLLOUT_MODEL_SLUG:-Qwen_Qwen3-4B_l19_thinkendlast10}"
RESPONSE_DATASET_NAME="${RESPONSE_DATASET_NAME:-${DATASET_NAME}_response_l19_thinkendlast10}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/classifer_training/artifacts/probe/spo_offline_subset0_1_qwen3_4b_base_L19_promptlast10_thinkendlast10_entropy3_dapo}"
PROMPT_BATCH_SIZE="${PROMPT_BATCH_SIZE:-16}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
ROLLOUT_MAX_BATCH_TOKENS="${ROLLOUT_MAX_BATCH_TOKENS:-24000}"
HOLDOUT_FRACTION="${HOLDOUT_FRACTION:-0}"
MAX_PROMPTS_PER_SUBSET="${MAX_PROMPTS_PER_SUBSET:-}"
LOCAL_FILES_ONLY=0
OVERWRITE=0
KEEP_SOURCE_LABELS=0
SKIP_PREPARE=0
SKIP_PROMPT=0
SKIP_ROLLOUT=0
SKIP_TRAIN=0

usage() {
  cat <<'EOF'
Usage:
  bash classifer_training/run_spo_offline_l19_thinkend_value_estimator.sh [options]

This builds a L19 prompt-last10 + think_end_last10 + entropy3 Ridge value estimator
from only:
  /home/jongwonlim/verl/yoonho/spo/offline_value_estimation_subset_0/validation_data/0.jsonl
  /home/jongwonlim/verl/yoonho/spo/offline_value_estimation_subset_1/validation_data/0.jsonl

Options:
  --gpu-id N                    GPU for hidden extraction. Default: $GPU_ID or 0.
  --python PATH                 Python executable.
  --model NAME                  Display model id. Default: Qwen/Qwen3-4B.
  --load-model PATH             Load model path/id. Default: same as --model.
  --model-cache-dir PATH        HF model cache dir.
  --spo-root PATH               Directory containing offline_value_estimation_subset_*.
  --subset0-dir PATH            offline_value_estimation_subset_0 dir.
  --subset1-dir PATH            offline_value_estimation_subset_1 dir.
  --work-root PATH              Prepared dataset/run root.
  --output-dir PATH             Trained probe output dir.
  --holdout-fraction FLOAT      Optional prompt holdout fraction. Default: 0.
  --max-prompts-per-subset N    Smoke/debug: limit prompts per subset during prepare.
  --keep-source-labels          Use existing score/reward instead of math_dapo relabeling.
  --local-files-only            HF local_files_only.
  --overwrite                   Rebuild existing artifacts.
  --skip-prepare|--skip-prompt|--skip-rollout|--skip-train
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-id) GPU_ID="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --model) MODEL_NAME="$2"; shift 2 ;;
    --load-model) MODEL_LOAD_NAME_OR_PATH="$2"; shift 2 ;;
    --model-cache-dir) MODEL_CACHE_DIR="$2"; shift 2 ;;
    --spo-root) SPO_ROOT="$2"; SUBSET0_DIR="${SPO_ROOT}/offline_value_estimation_subset_0"; SUBSET1_DIR="${SPO_ROOT}/offline_value_estimation_subset_1"; shift 2 ;;
    --subset0-dir) SUBSET0_DIR="$2"; shift 2 ;;
    --subset1-dir) SUBSET1_DIR="$2"; shift 2 ;;
    --work-root) WORK_ROOT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --holdout-fraction) HOLDOUT_FRACTION="$2"; shift 2 ;;
    --max-prompts-per-subset) MAX_PROMPTS_PER_SUBSET="$2"; shift 2 ;;
    --keep-source-labels) KEEP_SOURCE_LABELS=1; shift ;;
    --local-files-only) LOCAL_FILES_ONLY=1; shift ;;
    --overwrite) OVERWRITE=1; shift ;;
    --skip-prepare) SKIP_PREPARE=1; shift ;;
    --skip-prompt) SKIP_PROMPT=1; shift ;;
    --skip-rollout) SKIP_ROLLOUT=1; shift ;;
    --skip-train) SKIP_TRAIN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
}

PROMPT_DATASET_DIR="${WORK_ROOT}/prompt_dataset"
RUN_ROOT="${WORK_ROOT}/runs"
RUN_DIR0="${RUN_ROOT}/offline_value_estimation_subset_0"
RUN_DIR1="${RUN_ROOT}/offline_value_estimation_subset_1"
PROMPT_HIDDEN_PATH="${ROOT}/classifer_training/artifacts/hidden/${DATASET_NAME}/${PROMPT_MODEL_SLUG}/hidden_states.pt"
PROMPT_INDEX_PATH="${ROOT}/classifer_training/artifacts/index/${DATASET_NAME}/${PROMPT_MODEL_SLUG}/index.jsonl"
ROLLOUT_HIDDEN_PATH="${ROOT}/classifer_training/artifacts/rollout_hidden/${RESPONSE_DATASET_NAME}/${ROLLOUT_MODEL_SLUG}/rollout_hidden_states.pt"
ROLLOUT_INDEX_PATH="${ROOT}/classifer_training/artifacts/rollout_index/${RESPONSE_DATASET_NAME}/${ROLLOUT_MODEL_SLUG}/rollout_index.jsonl"

LOCAL_ONLY_FLAG=()
if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
  LOCAL_ONLY_FLAG=(--local_files_only)
fi
CACHE_FLAG=(--cache_dir "$MODEL_CACHE_DIR")
OVERWRITE_FLAG=()
if [[ "$OVERWRITE" == "1" ]]; then
  OVERWRITE_FLAG=(--overwrite)
fi

log "ROOT=${ROOT}"
log "PYTHON=${PYTHON}"
log "MODEL_NAME=${MODEL_NAME}"
log "MODEL_LOAD_NAME_OR_PATH=${MODEL_LOAD_NAME_OR_PATH}"
log "GPU_ID=${GPU_ID}"
log "WORK_ROOT=${WORK_ROOT}"
log "OUTPUT_DIR=${OUTPUT_DIR}"
log "ROLLOUT_COMPONENT=think_end_last10_hidden"
log "LABEL_SOURCE=$([[ "$KEEP_SOURCE_LABELS" == "1" ]] && echo source || echo math_dapo)"

if [[ "$SKIP_PREPARE" != "1" ]]; then
  prepare_args=(
    -m classifer_training.prepare_spo_offline_validation_data
    --subset-dirs "$SUBSET0_DIR" "$SUBSET1_DIR"
    --output-root "$WORK_ROOT"
    --dataset-name "$DATASET_NAME"
  )
  if [[ "$KEEP_SOURCE_LABELS" == "1" ]]; then
    prepare_args+=(--keep-source-labels)
  fi
  if [[ "$OVERWRITE" == "1" ]]; then
    prepare_args+=(--overwrite)
  fi
  if [[ -n "$MAX_PROMPTS_PER_SUBSET" ]]; then
    prepare_args+=(--max-prompts-per-subset "$MAX_PROMPTS_PER_SUBSET")
  fi
  log "[prepare] validation_data -> ${WORK_ROOT}"
  PYTHONPATH="$ROOT" "$PYTHON" "${prepare_args[@]}"
fi

if [[ "$SKIP_PROMPT" != "1" ]]; then
  if [[ "$OVERWRITE" != "1" && -f "$PROMPT_HIDDEN_PATH" && -f "$PROMPT_INDEX_PATH" ]]; then
    log "[prompt] hidden already exists: ${PROMPT_HIDDEN_PATH}"
  else
    log "[prompt] extracting L19 last10 hidden"
    CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONPATH="$ROOT" "$PYTHON" -u -m classifer_training.extract_hidden_states \
      --model_name_or_path "$MODEL_NAME" \
      --load_model_name_or_path "$MODEL_LOAD_NAME_OR_PATH" \
      --input_path "$PROMPT_DATASET_DIR" \
      --dataset_name "$DATASET_NAME" \
      --model_slug "$PROMPT_MODEL_SLUG" \
      --components hidden \
      --layers 19 \
      --last_n_values 10 \
      --batch_size "$PROMPT_BATCH_SIZE" \
      --torch_dtype bfloat16 \
      --disable_chat_template \
      --disable_generation_prompt \
      --disable_thinking \
      "${LOCAL_ONLY_FLAG[@]}" \
      "${CACHE_FLAG[@]}" \
      "${OVERWRITE_FLAG[@]}"
  fi
fi

if [[ "$SKIP_ROLLOUT" != "1" ]]; then
  if [[ "$OVERWRITE" != "1" && -f "$ROLLOUT_HIDDEN_PATH" && -f "$ROLLOUT_INDEX_PATH" ]]; then
    log "[rollout] hidden already exists: ${ROLLOUT_HIDDEN_PATH}"
  else
    log "[rollout] extracting L19 think_end_last10_hidden + entropy/logprob features"
    CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONPATH="$ROOT" "$PYTHON" -u -m classifer_training.extract_rollout_hidden_states \
      --model_name_or_path "$MODEL_NAME" \
      --load_model_name_or_path "$MODEL_LOAD_NAME_OR_PATH" \
      --run_dirs "$RUN_DIR0" "$RUN_DIR1" \
      --dataset_name "$RESPONSE_DATASET_NAME" \
      --model_slug "$ROLLOUT_MODEL_SLUG" \
      --components think_end_last10_hidden \
      --layers 19 \
      --batch_size "$ROLLOUT_BATCH_SIZE" \
      --max_batch_tokens "$ROLLOUT_MAX_BATCH_TOKENS" \
      --torch_dtype bfloat16 \
      --disable_chat_template \
      --disable_generation_prompt \
      --disable_thinking \
      "${LOCAL_ONLY_FLAG[@]}" \
      "${CACHE_FLAG[@]}" \
      "${OVERWRITE_FLAG[@]}"
  fi
fi

if [[ "$SKIP_TRAIN" != "1" ]]; then
  log "[train] Ridge probe"
  train_args=(
    -m classifer_training.train_spo_offline_l19_thinkend_value_estimator
    --prompt-hidden-path "$PROMPT_HIDDEN_PATH"
    --prompt-index-path "$PROMPT_INDEX_PATH"
    --rollout-hidden-path "$ROLLOUT_HIDDEN_PATH"
    --rollout-index-path "$ROLLOUT_INDEX_PATH"
    --output-dir "$OUTPUT_DIR"
    --holdout-fraction "$HOLDOUT_FRACTION"
  )
  if [[ "$OVERWRITE" == "1" ]]; then
    train_args+=(--overwrite)
  fi
  PYTHONPATH="$ROOT" "$PYTHON" "${train_args[@]}"
fi

log "Done. Model: ${OUTPUT_DIR}/model.joblib"
