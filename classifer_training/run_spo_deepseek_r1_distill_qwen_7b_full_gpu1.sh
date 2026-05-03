#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SPO_REPO_ROOT="${SPO_REPO_ROOT:-/home/jongwonlim/verl/yoonho/spo/spo}"
SPO_ESTIMATE_DIR="${SPO_ESTIMATE_DIR:-${SPO_REPO_ROOT}/recipe/spo/estimate_offline_values}"

PYTHON="${PYTHON:-/home/jongwonlim/anaconda3/envs/CB/bin/python}"
GPU_ID="${GPU_ID:-1}"
MODEL_NAME="${MODEL_NAME:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-${ROOT}/classifer_training/artifacts/hf_models}"
HF_HOME_DIR="${HF_HOME_DIR:-${MODEL_CACHE_DIR}}"

RUN_SLUG="${RUN_SLUG:-deepseek_r1_distill_qwen_7b}"
ROLLOUT_OUTPUT_DIR="${ROLLOUT_OUTPUT_DIR:-${SPO_REPO_ROOT}/spo_verl_pr_temp1_${RUN_SLUG}}"
DATA_DIR="${DATA_DIR:-${SPO_REPO_ROOT}/data/DAPO-Math-17k-Processed_Splits}"
SUBSET_START="${SUBSET_START:-0}"
SUBSET_END="${SUBSET_END:-3}"
TARGET_TRAJECTORIES="${TARGET_TRAJECTORIES:-16}"
SUBSET_SPECIAL_TARGET="${SUBSET_SPECIAL_TARGET:-2}"

LAYERS="${LAYERS:-14:27}"
NUM_MODEL_LAYERS="${NUM_MODEL_LAYERS:-28}"
DATASET_NAME="${DATASET_NAME:-spo_${RUN_SLUG}_subset0_1_train_subset2_3_validation}"
WORK_ROOT="${WORK_ROOT:-${ROOT}/classifer_training/artifacts/datasets/${DATASET_NAME}}"
PROMPT_SHARD_DIR="${PROMPT_SHARD_DIR:-${WORK_ROOT}_prompt_shards}"
PROMPT_MODEL_SLUG="${PROMPT_MODEL_SLUG:-${RUN_SLUG}_l14_27_last10mean}"
ROLLOUT_DATASET_NAME="${ROLLOUT_DATASET_NAME:-${DATASET_NAME}_thinkendlast10_l14_27}"
ROLLOUT_MODEL_SLUG="${ROLLOUT_MODEL_SLUG:-${RUN_SLUG}_l14_27_thinkendlast10}"
PROBE_OUTPUT_DIR="${PROBE_OUTPUT_DIR:-${ROOT}/classifer_training/artifacts/probe/${DATASET_NAME}_L14_27_promptlast10_thinkendlast10_p32_r256_entropy3_dapo}"
EXTRACT_LOG_ROOT="${EXTRACT_LOG_ROOT:-${ROOT}/classifer_training/artifacts/logs/${DATASET_NAME}_L14_27_promptlast10_thinkendlast10_gpu${GPU_ID}}"
MAIN_LOG_ROOT="${MAIN_LOG_ROOT:-${ROOT}/classifer_training/artifacts/logs/${DATASET_NAME}_full_gpu${GPU_ID}}"

PROMPT_BATCH_SIZE="${PROMPT_BATCH_SIZE:-2}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-1}"
ROLLOUT_MAX_BATCH_TOKENS="${ROLLOUT_MAX_BATCH_TOKENS:-12000}"
ROLLOUT_AGENT_WORKERS="${ROLLOUT_AGENT_WORKERS:-2}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-4096}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-32}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-8192}"
INFER_TP="${INFER_TP:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-1}"

SKIP_ROLLOUT=0
SKIP_EXTRACT_TRAIN=0
LOCAL_FILES_ONLY=0
OVERWRITE_EXTRACT=0

usage() {
  cat <<'EOF'
Usage:
  bash classifer_training/run_spo_deepseek_r1_distill_qwen_7b_full_gpu1.sh [options]

Runs the same SPO offline-value pipeline as the 1.5B setup, but for
deepseek-ai/DeepSeek-R1-Distill-Qwen-7B on GPU 1:
  1. rollout subset0,1 with 2 samples/prompt and subset2,3 with 16 samples/prompt
  2. extract prompt last10 and rollout think_end_last10 hidden over middle-late layers
  3. train tied layer sweep Ridge probes

Options:
  --gpu-id N
  --python PATH
  --model NAME_OR_PATH
  --model-cache-dir PATH
  --rollout-output-dir PATH
  --data-dir PATH
  --layers SPEC
  --local-files-only
  --overwrite-extract
  --skip-rollout
  --skip-extract-train
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-id) GPU_ID="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --model) MODEL_NAME="$2"; shift 2 ;;
    --model-cache-dir) MODEL_CACHE_DIR="$2"; HF_HOME_DIR="$2"; shift 2 ;;
    --rollout-output-dir) ROLLOUT_OUTPUT_DIR="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --layers) LAYERS="$2"; shift 2 ;;
    --local-files-only) LOCAL_FILES_ONLY=1; shift ;;
    --overwrite-extract) OVERWRITE_EXTRACT=1; shift ;;
    --skip-rollout) SKIP_ROLLOUT=1; shift ;;
    --skip-extract-train) SKIP_EXTRACT_TRAIN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
}

mkdir -p "$MODEL_CACHE_DIR" "$MAIN_LOG_ROOT" "$EXTRACT_LOG_ROOT"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PATH="$(dirname "$PYTHON"):${PATH}"
export PYTHONPATH="$SPO_REPO_ROOT:$ROOT:${PYTHONPATH:-}"
export HF_HOME="$HF_HOME_DIR"
export HF_HUB_CACHE="$MODEL_CACHE_DIR"
export TRANSFORMERS_CACHE="$MODEL_CACHE_DIR"
export TOKENIZERS_PARALLELISM=false

log "ROOT=${ROOT}"
log "SPO_REPO_ROOT=${SPO_REPO_ROOT}"
log "PYTHON=${PYTHON}"
log "GPU_ID=${GPU_ID}"
log "MODEL_NAME=${MODEL_NAME}"
log "MODEL_CACHE_DIR=${MODEL_CACHE_DIR}"
log "ROLLOUT_OUTPUT_DIR=${ROLLOUT_OUTPUT_DIR}"
log "DATA_DIR=${DATA_DIR}"
log "LAYERS=${LAYERS}"
log "DATASET_NAME=${DATASET_NAME}"
log "PROBE_OUTPUT_DIR=${PROBE_OUTPUT_DIR}"

if [[ "$SKIP_ROLLOUT" != "1" ]]; then
  log "[rollout] subset ${SUBSET_START}..${SUBSET_END} on GPU ${GPU_ID}"
  (
    cd "$SPO_REPO_ROOT"
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    HF_HOME="$HF_HOME_DIR" \
    HF_HUB_CACHE="$MODEL_CACHE_DIR" \
    TRANSFORMERS_CACHE="$MODEL_CACHE_DIR" \
    PYTHON_BIN="$PYTHON" \
    OUTPUT_DIR="$ROLLOUT_OUTPUT_DIR" \
    DATA_DIR="$DATA_DIR" \
    MODEL_PATH="$MODEL_NAME" \
    SUBSET_START="$SUBSET_START" \
    SUBSET_END="$SUBSET_END" \
    TARGET_TRAJECTORIES="$TARGET_TRAJECTORIES" \
    SUBSET_SPECIAL_TARGET="$SUBSET_SPECIAL_TARGET" \
    RESPONSE_LENGTH="$RESPONSE_LENGTH" \
    INFER_TP="$INFER_TP" \
    N_GPUS_PER_NODE="$N_GPUS_PER_NODE" \
    ROLLOUT_AGENT_WORKERS="$ROLLOUT_AGENT_WORKERS" \
    ROLLOUT_MAX_NUM_BATCHED_TOKENS="$ROLLOUT_MAX_NUM_BATCHED_TOKENS" \
    ROLLOUT_MAX_NUM_SEQS="$ROLLOUT_MAX_NUM_SEQS" \
    bash "${SPO_ESTIMATE_DIR}/run_eval_subsets.sh"
  ) 2>&1 | tee "${MAIN_LOG_ROOT}/rollout.log"
else
  log "[rollout] skipped"
fi

if [[ "$SKIP_EXTRACT_TRAIN" != "1" ]]; then
  extract_args=(
    --gpu-ids "$GPU_ID"
    --python "$PYTHON"
    --model "$MODEL_NAME"
    --load-model "$MODEL_NAME"
    --model-cache-dir "$MODEL_CACHE_DIR"
    --spo-output-root "${ROLLOUT_OUTPUT_DIR}/spo"
    --layers "$LAYERS"
    --prompt-batch-size "$PROMPT_BATCH_SIZE"
    --rollout-batch-size "$ROLLOUT_BATCH_SIZE"
    --rollout-max-batch-tokens "$ROLLOUT_MAX_BATCH_TOKENS"
    --work-root "$WORK_ROOT"
    --output-dir "$PROBE_OUTPUT_DIR"
  )
  if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
    extract_args+=(--local-files-only)
  fi
  if [[ "$OVERWRITE_EXTRACT" == "1" ]]; then
    extract_args+=(--overwrite)
  fi

  log "[extract-train] prompt/rollout hidden extraction + layer sweep"
  DATASET_NAME="$DATASET_NAME" \
  PROMPT_SHARD_DIR="$PROMPT_SHARD_DIR" \
  PROMPT_MODEL_SLUG="$PROMPT_MODEL_SLUG" \
  ROLLOUT_DATASET_NAME="$ROLLOUT_DATASET_NAME" \
  ROLLOUT_MODEL_SLUG="$ROLLOUT_MODEL_SLUG" \
  LOG_ROOT="$EXTRACT_LOG_ROOT" \
  NUM_MODEL_LAYERS="$NUM_MODEL_LAYERS" \
  bash "${ROOT}/classifer_training/run_spo_deepseek_r1_offline_layer_sweep_2gpu.sh" "${extract_args[@]}" \
    2>&1 | tee "${MAIN_LOG_ROOT}/extract_train.log"
else
  log "[extract-train] skipped"
fi

log "Done. Probe summary: ${PROBE_OUTPUT_DIR}/layer_sweep_summary.md"
