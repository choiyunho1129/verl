#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -x /home/jongwonlim/anaconda3/envs/CB/bin/python ]]; then
  DEFAULT_PYTHON="/home/jongwonlim/anaconda3/envs/CB/bin/python"
else
  DEFAULT_PYTHON="python"
fi

PYTHON="${PYTHON:-$DEFAULT_PYTHON}"
MODEL_NAME="${MODEL_NAME:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
MODEL_LOAD_NAME_OR_PATH="${MODEL_LOAD_NAME_OR_PATH:-$MODEL_NAME}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-${HF_HUB_CACHE:-/home/holi_models}}"
GPU_IDS="${GPU_IDS:-0,3}"
LAYERS="${LAYERS:-14:27}"
NUM_MODEL_LAYERS="${NUM_MODEL_LAYERS:-28}"
PROMPT_LAST_N_VALUES="${PROMPT_LAST_N_VALUES:-10}"
ROLLOUT_COMPONENTS="${ROLLOUT_COMPONENTS:-think_end_last10_hidden}"
PROMPT_BATCH_SIZE="${PROMPT_BATCH_SIZE:-8}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-1}"
ROLLOUT_MAX_BATCH_TOKENS="${ROLLOUT_MAX_BATCH_TOKENS:-12000}"
PROMPT_PCA_DIM="${PROMPT_PCA_DIM:-32}"
ROLLOUT_PCA_DIM="${ROLLOUT_PCA_DIM:-256}"
SCALAR_KEYS="${SCALAR_KEYS:-output_mean_token_entropy,reasoning_mean_token_entropy,answer_mean_token_entropy}"
DATASET_NAME="${DATASET_NAME:-spo_deepseek_r1_subset0_1_train_subset2_3_validation}"
WORK_ROOT="${WORK_ROOT:-${ROOT}/classifer_training/artifacts/datasets/${DATASET_NAME}}"
SPO_OUTPUT_ROOT="${SPO_OUTPUT_ROOT:-/data2/jongwonlim/verl/yoonho/spo/spo/spo_verl_pr_temp1_deepseek_r1_distill_qwen_1_5b/spo}"
PROMPT_SHARD_DIR="${PROMPT_SHARD_DIR:-${WORK_ROOT}_prompt_shards}"
PROMPT_MODEL_SLUG="${PROMPT_MODEL_SLUG:-deepseek_r1_distill_qwen_1_5b_l14_27_last10mean}"
ROLLOUT_DATASET_NAME="${ROLLOUT_DATASET_NAME:-${DATASET_NAME}_thinkendlast10_l14_27}"
ROLLOUT_MODEL_SLUG="${ROLLOUT_MODEL_SLUG:-deepseek_r1_distill_qwen_1_5b_l14_27_thinkendlast10}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/classifer_training/artifacts/probe/${DATASET_NAME}_L14_27_promptlast10_thinkendlast10_p32_r256_entropy3_dapo}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/classifer_training/artifacts/logs/${DATASET_NAME}_L14_27_promptlast10_thinkendlast10_2gpu}"
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
  bash classifer_training/run_spo_deepseek_r1_offline_layer_sweep_2gpu.sh [options]

Builds SPO DeepSeek-R1-Distill-Qwen-1.5B offline value-estimator artifacts:
  train:      subset 0,1 = 4096 prompts, 2 rollouts each
  validation: subset 2,3 = 1024 prompts, 16 rollouts each
  hidden: prompt last10 mean + rollout think_end_last10_hidden mean
  sweep: tied prompt/rollout layers over --layers, default 14:27

Options:
  --gpu-ids IDS                 Comma-separated GPU ids. Default: 0,3.
  --python PATH                 Python executable.
  --model NAME                  Metadata model id. Default: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B.
  --load-model PATH             Model path/id to load. Default: same as --model.
  --model-cache-dir PATH        HF model cache dir.
  --spo-output-root PATH        Directory containing offline_value_estimation_subset_0..3.
  --layers SPEC                 Layer spec, 0-indexed. Default: 14:27.
  --prompt-batch-size N         Prompt extraction batch size. Default: 8.
  --rollout-batch-size N        Rollout extraction batch size. Default: 1.
  --rollout-max-batch-tokens N  Rollout extraction token budget. Default: 12000.
  --output-dir PATH             Layer-sweep probe output dir.
  --work-root PATH              Prepared dataset root.
  --local-files-only            HF local_files_only.
  --keep-source-labels          Use source reward/score instead of math_dapo relabeling.
  --overwrite                   Rebuild existing artifacts.
  --skip-prepare|--skip-prompt|--skip-rollout|--skip-train
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-ids) GPU_IDS="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --model) MODEL_NAME="$2"; shift 2 ;;
    --load-model) MODEL_LOAD_NAME_OR_PATH="$2"; shift 2 ;;
    --model-cache-dir) MODEL_CACHE_DIR="$2"; shift 2 ;;
    --spo-output-root) SPO_OUTPUT_ROOT="$2"; shift 2 ;;
    --layers) LAYERS="$2"; shift 2 ;;
    --prompt-batch-size) PROMPT_BATCH_SIZE="$2"; shift 2 ;;
    --rollout-batch-size) ROLLOUT_BATCH_SIZE="$2"; shift 2 ;;
    --rollout-max-batch-tokens) ROLLOUT_MAX_BATCH_TOKENS="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --work-root) WORK_ROOT="$2"; shift 2 ;;
    --local-files-only) LOCAL_FILES_ONLY=1; shift ;;
    --keep-source-labels) KEEP_SOURCE_LABELS=1; shift ;;
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

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
NUM_SHARDS="${#GPU_ARRAY[@]}"
if [[ "$NUM_SHARDS" -lt 1 ]]; then
  echo "No GPU ids were provided." >&2
  exit 2
fi

mkdir -p "$LOG_ROOT"

LOCAL_ONLY_FLAG=()
if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
  LOCAL_ONLY_FLAG=(--local_files_only)
fi
CACHE_FLAG=()
if [[ -n "$MODEL_CACHE_DIR" ]]; then
  CACHE_FLAG=(--cache_dir "$MODEL_CACHE_DIR")
fi
OVERWRITE_FLAG=()
if [[ "$OVERWRITE" == "1" ]]; then
  OVERWRITE_FLAG=(--overwrite)
fi

SUBSET_DIRS=(
  "${SPO_OUTPUT_ROOT}/offline_value_estimation_subset_0"
  "${SPO_OUTPUT_ROOT}/offline_value_estimation_subset_1"
  "${SPO_OUTPUT_ROOT}/offline_value_estimation_subset_2"
  "${SPO_OUTPUT_ROOT}/offline_value_estimation_subset_3"
)
PROMPT_DATASET_DIR="${WORK_ROOT}/prompt_dataset"
RUN_ROOT="${WORK_ROOT}/runs"
RUN_DIRS=(
  "${RUN_ROOT}/offline_value_estimation_subset_0"
  "${RUN_ROOT}/offline_value_estimation_subset_1"
  "${RUN_ROOT}/offline_value_estimation_subset_2"
  "${RUN_ROOT}/offline_value_estimation_subset_3"
)

prepare_prompt_shards() {
  if [[ -f "${PROMPT_SHARD_DIR}/manifest.json" && "$OVERWRITE" != "1" ]]; then
    log "[prompt-shards] already exist: ${PROMPT_SHARD_DIR}"
    return
  fi
  log "[prompt-shards] preparing ${NUM_SHARDS} shards under ${PROMPT_SHARD_DIR}"
  NUM_SHARDS="$NUM_SHARDS" PROMPT_DATASET_DIR="$PROMPT_DATASET_DIR" PROMPT_SHARD_DIR="$PROMPT_SHARD_DIR" \
    PYTHONPATH="$ROOT" "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

num_shards = int(os.environ["NUM_SHARDS"])
input_dir = Path(os.environ["PROMPT_DATASET_DIR"])
output_dir = Path(os.environ["PROMPT_SHARD_DIR"])

rows_by_split = {}
all_rows = []
for split in ("train", "validation", "test"):
    path = input_dir / f"{split}.jsonl"
    if not path.exists():
        continue
    with path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows_by_split[split] = rows
    all_rows.extend(rows)

if not all_rows:
    raise SystemExit(f"No prompt rows found under {input_dir}")

shards = [dict() for _ in range(num_shards)]
for split, rows in rows_by_split.items():
    split_shards = [[] for _ in range(num_shards)]
    for idx, row in enumerate(rows):
        split_shards[idx % num_shards].append(row)
    for shard_idx, shard_rows in enumerate(split_shards):
        shards[shard_idx][split] = shard_rows

output_dir.mkdir(parents=True, exist_ok=True)
sizes = []
for shard_idx, split_map in enumerate(shards):
    shard_dir = output_dir / f"shard{shard_idx}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for split, rows in sorted(split_map.items()):
        with (shard_dir / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        total += len(rows)
    sizes.append(total)

manifest = {
    "input_dir": str(input_dir),
    "output_dir": str(output_dir),
    "num_rows_total": len(all_rows),
    "split_counts": {split: len(rows) for split, rows in rows_by_split.items()},
    "num_shards": num_shards,
    "shard_sizes": sizes,
}
(output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(json.dumps(manifest, indent=2))
PY
}

wait_for_pids() {
  local label="$1"
  shift
  local pids=("$@")
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" != "0" ]]; then
    echo "${label} failed. Check logs under ${LOG_ROOT}" >&2
    exit 1
  fi
}

log "ROOT=${ROOT}"
log "PYTHON=${PYTHON}"
log "MODEL_NAME=${MODEL_NAME}"
log "MODEL_LOAD_NAME_OR_PATH=${MODEL_LOAD_NAME_OR_PATH}"
log "GPU_IDS=${GPU_IDS}"
log "NUM_SHARDS=${NUM_SHARDS}"
log "LAYERS=${LAYERS}"
log "PROMPT_LAST_N_VALUES=${PROMPT_LAST_N_VALUES}"
log "ROLLOUT_COMPONENTS=${ROLLOUT_COMPONENTS}"
log "WORK_ROOT=${WORK_ROOT}"
log "LOG_ROOT=${LOG_ROOT}"
log "OUTPUT_DIR=${OUTPUT_DIR}"

if [[ "$SKIP_PREPARE" != "1" ]]; then
  prepare_args=(
    -m classifer_training.prepare_spo_offline_validation_data
    --subset-dirs "${SUBSET_DIRS[@]}"
    --output-root "$WORK_ROOT"
    --dataset-name "$DATASET_NAME"
    --validation-subset-ids 2 3
  )
  if [[ "$KEEP_SOURCE_LABELS" == "1" ]]; then
    prepare_args+=(--keep-source-labels)
  fi
  if [[ "$OVERWRITE" == "1" ]]; then
    prepare_args+=(--overwrite)
  fi
  log "[prepare] SPO validation_data -> ${WORK_ROOT}"
  PYTHONPATH="$ROOT" "$PYTHON" "${prepare_args[@]}"
fi

prepare_prompt_shards

PROMPT_HIDDEN_PATHS=()
PROMPT_INDEX_PATHS=()
for shard in "${!GPU_ARRAY[@]}"; do
  shard_dataset="${DATASET_NAME}_prompt_shard${shard}"
  PROMPT_HIDDEN_PATHS+=("${ROOT}/classifer_training/artifacts/hidden/${shard_dataset}/${PROMPT_MODEL_SLUG}/hidden_states.pt")
  PROMPT_INDEX_PATHS+=("${ROOT}/classifer_training/artifacts/index/${shard_dataset}/${PROMPT_MODEL_SLUG}/index.jsonl")
done

ROLLOUT_HIDDEN_PATHS=()
ROLLOUT_INDEX_PATHS=()
for shard in "${!GPU_ARRAY[@]}"; do
  suffix="shard$(printf '%02d' "$shard")of$(printf '%02d' "$NUM_SHARDS")"
  ROLLOUT_HIDDEN_PATHS+=("${ROOT}/classifer_training/artifacts/rollout_hidden/${ROLLOUT_DATASET_NAME}/${ROLLOUT_MODEL_SLUG}/rollout_hidden_states.${suffix}.pt")
  ROLLOUT_INDEX_PATHS+=("${ROOT}/classifer_training/artifacts/rollout_index/${ROLLOUT_DATASET_NAME}/${ROLLOUT_MODEL_SLUG}/rollout_index.${suffix}.jsonl")
done

if [[ "$SKIP_PROMPT" != "1" ]]; then
  prompt_pids=()
  for shard in "${!GPU_ARRAY[@]}"; do
    gpu="${GPU_ARRAY[$shard]}"
    shard_dataset="${DATASET_NAME}_prompt_shard${shard}"
    if [[ "$OVERWRITE" != "1" && -f "${PROMPT_HIDDEN_PATHS[$shard]}" && -f "${PROMPT_INDEX_PATHS[$shard]}" ]]; then
      log "[prompt][shard${shard}] hidden already exists: ${PROMPT_HIDDEN_PATHS[$shard]}"
      continue
    fi
    log "[prompt][shard${shard}][gpu${gpu}] extracting -> ${LOG_ROOT}/prompt_shard${shard}_gpu${gpu}.log"
    (
      CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT" "$PYTHON" -u -m classifer_training.extract_hidden_states \
        --model_name_or_path "$MODEL_NAME" \
        --load_model_name_or_path "$MODEL_LOAD_NAME_OR_PATH" \
        --input_path "${PROMPT_SHARD_DIR}/shard${shard}" \
        --dataset_name "$shard_dataset" \
        --model_slug "$PROMPT_MODEL_SLUG" \
        --components hidden \
        --layers "$LAYERS" \
        --last_n_values "$PROMPT_LAST_N_VALUES" \
        --batch_size "$PROMPT_BATCH_SIZE" \
        --torch_dtype bfloat16 \
        --disable_chat_template \
        --disable_generation_prompt \
        --disable_thinking \
        "${LOCAL_ONLY_FLAG[@]}" \
        "${CACHE_FLAG[@]}" \
        "${OVERWRITE_FLAG[@]}"
    ) >"${LOG_ROOT}/prompt_shard${shard}_gpu${gpu}.log" 2>&1 &
    prompt_pids+=("$!")
  done
  wait_for_pids "prompt hidden extraction" "${prompt_pids[@]}"
  log "Finished prompt hidden extraction"
fi

if [[ "$SKIP_ROLLOUT" != "1" ]]; then
  rollout_pids=()
  for shard in "${!GPU_ARRAY[@]}"; do
    gpu="${GPU_ARRAY[$shard]}"
    if [[ "$OVERWRITE" != "1" && -f "${ROLLOUT_HIDDEN_PATHS[$shard]}" && -f "${ROLLOUT_INDEX_PATHS[$shard]}" ]]; then
      log "[rollout][shard${shard}] hidden already exists: ${ROLLOUT_HIDDEN_PATHS[$shard]}"
      continue
    fi
    log "[rollout][shard${shard}][gpu${gpu}] extracting hidden + entropy/logprob -> ${LOG_ROOT}/rollout_shard${shard}_gpu${gpu}.log"
    (
      CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT" "$PYTHON" -u -m classifer_training.extract_rollout_hidden_states \
        --model_name_or_path "$MODEL_NAME" \
        --load_model_name_or_path "$MODEL_LOAD_NAME_OR_PATH" \
        --run_dirs "${RUN_DIRS[@]}" \
        --dataset_name "$ROLLOUT_DATASET_NAME" \
        --model_slug "$ROLLOUT_MODEL_SLUG" \
        --components $ROLLOUT_COMPONENTS \
        --layers "$LAYERS" \
        --num_shards "$NUM_SHARDS" \
        --shard_index "$shard" \
        --batch_size "$ROLLOUT_BATCH_SIZE" \
        --max_batch_tokens "$ROLLOUT_MAX_BATCH_TOKENS" \
        --torch_dtype bfloat16 \
        --disable_chat_template \
        --disable_generation_prompt \
        --disable_thinking \
        "${LOCAL_ONLY_FLAG[@]}" \
        "${CACHE_FLAG[@]}" \
        "${OVERWRITE_FLAG[@]}"
    ) >"${LOG_ROOT}/rollout_shard${shard}_gpu${gpu}.log" 2>&1 &
    rollout_pids+=("$!")
  done
  wait_for_pids "rollout hidden extraction" "${rollout_pids[@]}"
  log "Finished rollout hidden extraction"
fi

if [[ "$SKIP_TRAIN" != "1" ]]; then
  IFS=',' read -r -a SCALAR_KEY_ARRAY <<< "$SCALAR_KEYS"
  log "[train] layer sweep"
  PYTHONPATH="$ROOT" "$PYTHON" -u -m classifer_training.train_spo_offline_thinkend_layer_sweep \
    --prompt-hidden-paths "${PROMPT_HIDDEN_PATHS[@]}" \
    --prompt-index-paths "${PROMPT_INDEX_PATHS[@]}" \
    --rollout-hidden-paths "${ROLLOUT_HIDDEN_PATHS[@]}" \
    --rollout-index-paths "${ROLLOUT_INDEX_PATHS[@]}" \
    --output-dir "$OUTPUT_DIR" \
    --prompt-component "hidden_last10_mean" \
    --rollout-component "think_end_last10_hidden" \
    --layers "$LAYERS" \
    --num-model-layers "$NUM_MODEL_LAYERS" \
    --train-subsets 0 1 \
    --validation-subsets 2 3 \
    --prompt-pca-dim "$PROMPT_PCA_DIM" \
    --rollout-pca-dim "$ROLLOUT_PCA_DIM" \
    --scalar-keys "${SCALAR_KEY_ARRAY[@]}" \
    "${OVERWRITE_FLAG[@]}" \
    >"${LOG_ROOT}/train_layer_sweep.log" 2>&1
fi

log "Done. Summary: ${OUTPUT_DIR}/layer_sweep_summary.md"
