#!/usr/bin/env bash
# Custom DeepScaleR pipeline with split-specific prompt counts and sampling rates.
# The long generation stage is broken into shard-level run dirs so reruns keep
# completed work instead of restarting from scratch.
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
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3}"

TRAIN_PROMPTS="${TRAIN_PROMPTS:-5500}"
VALIDATION_PROMPTS="${VALIDATION_PROMPTS:-2000}"
TRAIN_NUM_SAMPLES="${TRAIN_NUM_SAMPLES:-2}"
VALIDATION_NUM_SAMPLES="${VALIDATION_NUM_SAMPLES:-16}"
TRAIN_GENERATION_SHARD_SIZE="${TRAIN_GENERATION_SHARD_SIZE:-500}"
VALIDATION_GENERATION_SHARD_SIZE="${VALIDATION_GENERATION_SHARD_SIZE:-250}"
GENERATION_PARALLELISM="${GENERATION_PARALLELISM:-tp}"
REUSE_TRAIN_FROM_VALIDATION_PROMPTS="${REUSE_TRAIN_FROM_VALIDATION_PROMPTS:-}"
REUSE_TRAIN_DATASET_DIR="${REUSE_TRAIN_DATASET_DIR:-}"
REUSE_TRAIN_RUN_ROOT="${REUSE_TRAIN_RUN_ROOT:-}"

SEED="${SEED:-1}"
TEMPERATURE="${TEMPERATURE:-1}"
TOP_P="${TOP_P:-1}"
TOP_K="${TOP_K:--1}"
MATH_MAX_NEW_TOKENS="${MATH_MAX_NEW_TOKENS:-8192}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-64}"
PROMPT_BATCH_SIZE="${PROMPT_BATCH_SIZE:-32}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
ROLLOUT_MAX_BATCH_TOKENS="${ROLLOUT_MAX_BATCH_TOKENS:-24000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"

LAYERS="${LAYERS:-18:35}"
PROMPT_LAST_N_VALUES_CSV="${PROMPT_LAST_N_VALUES:-5,10,15}"
ROLLOUT_COMPONENTS="${ROLLOUT_COMPONENTS:-response_last5_mean_hidden response_last10_mean_hidden response_last15_mean_hidden}"

MIN_FREE_MIB="${MIN_FREE_MIB:-20000}"
MAX_GPU_UTIL="${MAX_GPU_UTIL:-20}"
POLL_SEC="${POLL_SEC:-30}"
SKIP_WAIT="${SKIP_WAIT:-0}"
OVERWRITE="${OVERWRITE:-0}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-1}"

SKIP_PREPARE="${SKIP_PREPARE:-0}"
SKIP_GENERATION="${SKIP_GENERATION:-0}"
SKIP_LABELS="${SKIP_LABELS:-0}"
SKIP_PROMPT="${SKIP_PROMPT:-0}"
SKIP_ROLLOUT="${SKIP_ROLLOUT:-0}"

DATASET_DIR_ENV_PROVIDED="${DEEPSCALER_DATASET_DIR+x}"
PROMPT_SHARD_DIR_ENV_PROVIDED="${DEEPSCALER_PROMPT_SHARD_DIR+x}"
LOG_ROOT_ENV_PROVIDED="${LOG_ROOT+x}"

usage() {
  cat <<'EOF'
Usage:
  bash classifer_training/run_deepscaler_custom_qwen3_4b_base_4gpu.sh \
    --gpu-ids 0,1 \
    --generation-parallelism shard \
    --local-files-only \
    --model-cache-dir /data2/sangjunsong/.cache/transformers \
    --reuse-train-from-validation-prompts 2048

Defaults:
  - train prompts: 5500, num_samples: 2
  - validation prompts: 2000, num_samples: 16
  - generation parallelism: tp
  - generation shards are preserved on rerun
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) shift; ROOT="$1" ;;
    --python) shift; PYTHON="$1" ;;
    --model) shift; MODEL_NAME="$1" ;;
    --model-load-path|--load-model-name-or-path) shift; MODEL_LOAD_NAME_OR_PATH="$1" ;;
    --model-cache-dir) shift; MODEL_CACHE_DIR="$1" ;;
    --gpu-ids) shift; GPU_IDS_CSV="$1" ;;
    --train-prompts) shift; TRAIN_PROMPTS="$1" ;;
    --validation-prompts) shift; VALIDATION_PROMPTS="$1" ;;
    --train-num-samples) shift; TRAIN_NUM_SAMPLES="$1" ;;
    --validation-num-samples) shift; VALIDATION_NUM_SAMPLES="$1" ;;
    --train-generation-shard-size) shift; TRAIN_GENERATION_SHARD_SIZE="$1" ;;
    --validation-generation-shard-size) shift; VALIDATION_GENERATION_SHARD_SIZE="$1" ;;
    --generation-parallelism) shift; GENERATION_PARALLELISM="$1" ;;
    --seed) shift; SEED="$1" ;;
    --temperature) shift; TEMPERATURE="$1" ;;
    --top-p) shift; TOP_P="$1" ;;
    --top-k) shift; TOP_K="$1" ;;
    --math-max-new-tokens) shift; MATH_MAX_NEW_TOKENS="$1" ;;
    --gen-batch-size) shift; GEN_BATCH_SIZE="$1" ;;
    --prompt-batch-size) shift; PROMPT_BATCH_SIZE="$1" ;;
    --rollout-batch-size) shift; ROLLOUT_BATCH_SIZE="$1" ;;
    --rollout-max-batch-tokens) shift; ROLLOUT_MAX_BATCH_TOKENS="$1" ;;
    --gpu-memory-utilization) shift; GPU_MEMORY_UTILIZATION="$1" ;;
    --layers) shift; LAYERS="$1" ;;
    --prompt-last-n-values) shift; PROMPT_LAST_N_VALUES_CSV="$1" ;;
    --rollout-components) shift; ROLLOUT_COMPONENTS="$1" ;;
    --reuse-train-from-validation-prompts) shift; REUSE_TRAIN_FROM_VALIDATION_PROMPTS="$1" ;;
    --reuse-train-dataset-dir) shift; REUSE_TRAIN_DATASET_DIR="$1" ;;
    --reuse-train-run-root) shift; REUSE_TRAIN_RUN_ROOT="$1" ;;
    --deepscaler-dataset-dir) shift; DEEPSCALER_DATASET_DIR="$1"; DATASET_DIR_ENV_PROVIDED=1 ;;
    --deepscaler-prompt-shard-dir) shift; DEEPSCALER_PROMPT_SHARD_DIR="$1"; PROMPT_SHARD_DIR_ENV_PROVIDED=1 ;;
    --log-root) shift; LOG_ROOT="$1"; LOG_ROOT_ENV_PROVIDED=1 ;;
    --overwrite) OVERWRITE=1 ;;
    --local-files-only) LOCAL_FILES_ONLY=1 ;;
    --trust-remote-code) TRUST_REMOTE_CODE=1 ;;
    --enforce-eager) ENFORCE_EAGER=1 ;;
    --enable-custom-all-reduce) DISABLE_CUSTOM_ALL_REDUCE=0 ;;
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
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

normalize_gpu_ids_csv() {
  local raw="$1"
  local raw_spaces="${raw//,/ }"
  local ids=()
  local gpu_id
  read -r -a ids <<< "$raw_spaces"
  if [[ "${#ids[@]}" -lt 1 ]]; then
    echo "" >&2
    return 1
  fi
  local normalized=""
  for gpu_id in "${ids[@]}"; do
    [[ -n "$normalized" ]] && normalized+=","
    normalized+="$gpu_id"
  done
  printf '%s\n' "$normalized"
}

GPU_IDS_CSV="$(normalize_gpu_ids_csv "$GPU_IDS_CSV")"
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

if [[ -z "${TP_SIZE:-}" ]]; then
  if [[ "$GENERATION_PARALLELISM" == "shard" ]]; then
    TP_SIZE=1
  else
    TP_SIZE="${#GPU_IDS[@]}"
  fi
fi

case "$GENERATION_PARALLELISM" in
  tp|shard)
    ;;
  *)
    echo "GENERATION_PARALLELISM must be one of: tp, shard" >&2
    exit 2
    ;;
esac

if [[ "$GENERATION_PARALLELISM" == "shard" && "$TP_SIZE" != "1" ]]; then
  echo "GENERATION_PARALLELISM=shard currently requires TP_SIZE=1. Use tp mode for multi-GPU tensor parallel generation." >&2
  exit 2
fi

if [[ -z "$MODEL_CACHE_DIR" ]]; then
  MODEL_CACHE_DIR="${ROOT}/classifer_training/artifacts/hf_cache"
fi
mkdir -p "$MODEL_CACHE_DIR"

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

py_sanitize_model_slug() {
  PYTHONPATH="$ROOT" "$PYTHON" - "$MODEL_NAME" <<'PY'
import sys
from classifer_training.utils import sanitize_name
print(sanitize_name(sys.argv[1]))
PY
}

MODEL_SLUG="${MODEL_SLUG:-$(py_sanitize_model_slug)}"
DATASET_SLUG="${DATASET_SLUG:-deepscaler_train${TRAIN_PROMPTS}_validation${VALIDATION_PROMPTS}_seed${SEED}}"
PROMPT_MODEL_SLUG="${PROMPT_MODEL_SLUG:-qwen3_4b_base_l18_35_last5_10_15mean}"
ROLLOUT_MODEL_SLUG="${ROLLOUT_MODEL_SLUG:-${MODEL_SLUG}_l18_35_last5_10_15mean}"
RUN_SUFFIX="${RUN_SUFFIX:-temp${TEMPERATURE}_topp${TOP_P}_topk${TOP_K}_train${TRAIN_PROMPTS}x${TRAIN_NUM_SAMPLES}_validation${VALIDATION_PROMPTS}x${VALIDATION_NUM_SAMPLES}_vllm_tp${TP_SIZE}_seed${SEED}}"

if [[ -n "$REUSE_TRAIN_FROM_VALIDATION_PROMPTS" ]]; then
  if [[ -z "$REUSE_TRAIN_DATASET_DIR" ]]; then
    REUSE_TRAIN_DATASET_SLUG="deepscaler_train${TRAIN_PROMPTS}_validation${REUSE_TRAIN_FROM_VALIDATION_PROMPTS}_seed${SEED}"
    REUSE_TRAIN_DATASET_DIR="${ROOT}/classifer_training/artifacts/datasets/${REUSE_TRAIN_DATASET_SLUG}"
  fi
  if [[ -z "$REUSE_TRAIN_RUN_ROOT" ]]; then
    REUSE_TRAIN_RUN_SUFFIX="temp${TEMPERATURE}_topp${TOP_P}_topk${TOP_K}_train${TRAIN_PROMPTS}x${TRAIN_NUM_SAMPLES}_validation${REUSE_TRAIN_FROM_VALIDATION_PROMPTS}x${VALIDATION_NUM_SAMPLES}_vllm_tp${TP_SIZE}_seed${SEED}"
    REUSE_TRAIN_RUN_ROOT="${ROOT}/classifer_training/artifacts/runs/deepscaler/${MODEL_SLUG}/${REUSE_TRAIN_RUN_SUFFIX}/train_runs"
  fi
fi

if [[ -z "$DATASET_DIR_ENV_PROVIDED" ]]; then
  DEEPSCALER_DATASET_DIR="${ROOT}/classifer_training/artifacts/datasets/${DATASET_SLUG}"
fi
if [[ -z "$PROMPT_SHARD_DIR_ENV_PROVIDED" ]]; then
  DEEPSCALER_PROMPT_SHARD_DIR="${ROOT}/classifer_training/artifacts/datasets/${DATASET_SLUG}_prompt_shards${NUM_SHARDS}"
fi
if [[ -z "$LOG_ROOT_ENV_PROVIDED" ]]; then
  LOG_ROOT="${ROOT}/classifer_training/artifacts/logs/qwen3_4b_base_${DATASET_SLUG}_${RUN_SUFFIX}"
fi

mkdir -p "$LOG_ROOT"
PIPELINE_LOG="${LOG_ROOT}/pipeline.log"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

RUN_ROOT="${ROOT}/classifer_training/artifacts/runs/deepscaler/${MODEL_SLUG}/${RUN_SUFFIX}"
DEFAULT_TRAIN_RUN_ROOT="${RUN_ROOT}/train_runs"
TRAIN_RUN_ROOT="${REUSE_TRAIN_RUN_ROOT:-$DEFAULT_TRAIN_RUN_ROOT}"
VALIDATION_RUN_ROOT="${RUN_ROOT}/validation_runs"
LABELS_PATH="${ROOT}/classifer_training/artifacts/labels/deepscaler/${MODEL_SLUG}/${DATASET_SLUG}_${RUN_SUFFIX}_labels.jsonl"
LABELS_SUMMARY="${ROOT}/classifer_training/artifacts/labels/deepscaler/${MODEL_SLUG}/${DATASET_SLUG}_${RUN_SUFFIX}_summary.json"
LABELS_SCRATCH="${ROOT}/classifer_training/artifacts/datasets/${DATASET_SLUG}_${RUN_SUFFIX}_${MODEL_SLUG}_labels_scratch"
RESPONSE_DATASET_NAME="${RESPONSE_DATASET_NAME:-${DATASET_SLUG}_${RUN_SUFFIX}_response_l18_35_last5_10_15mean}"

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

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Missing required directory: $1" >&2
    exit 1
  fi
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

wait_for_gpu_group() {
  local gpu_csv="$1"
  local gpu_group=()
  local gpu
  IFS=',' read -r -a gpu_group <<< "$gpu_csv"
  for gpu in "${gpu_group[@]}"; do
    wait_for_gpu "$gpu"
  done
}

prepare_custom_dataset() {
  local reuse_train_flag=()
  if [[ "$SKIP_PREPARE" == "1" ]]; then
    log "[prepare] skipping custom dataset prep"
    return 0
  fi
  if [[ "$OVERWRITE" != "1" && -f "${DEEPSCALER_DATASET_DIR}/summary.json" ]]; then
    log "[prepare] custom dataset already exists: ${DEEPSCALER_DATASET_DIR}"
    return 0
  fi
  if [[ -n "$REUSE_TRAIN_DATASET_DIR" ]]; then
    reuse_train_flag=(--reuse_train_dataset_dir "$REUSE_TRAIN_DATASET_DIR")
  fi
  log "[prepare] custom DeepScaleR dataset -> ${DEEPSCALER_DATASET_DIR}"
  PYTHONPATH="$ROOT" "$PYTHON" -u -m classifer_training.prepare_custom_deepscaler_dataset \
    --output_dir "$DEEPSCALER_DATASET_DIR" \
    --dataset_name "$DATASET_SLUG" \
    --train_prompts "$TRAIN_PROMPTS" \
    --validation_prompts "$VALIDATION_PROMPTS" \
    --train_generation_shard_size "$TRAIN_GENERATION_SHARD_SIZE" \
    --validation_generation_shard_size "$VALIDATION_GENERATION_SHARD_SIZE" \
    --sample_seed "$SEED" \
    "${reuse_train_flag[@]}" \
    "${OVERWRITE_FLAG[@]}"
}

prepare_prompt_shards() {
  local input_dir="$1"
  local output_dir="$2"
  if [[ "$OVERWRITE" != "1" && -f "${output_dir}/summary.json" ]]; then
    local ok=1
    local shard
    for ((shard=0; shard<NUM_SHARDS; shard++)); do
      [[ -f "${output_dir}/shard${shard}.jsonl" ]] || ok=0
    done
    if [[ "$ok" == "1" ]]; then
      log "[prompt-shards] already exist: ${output_dir}"
      return 0
    fi
  fi
  log "[prompt-shards] preparing ${NUM_SHARDS} shards under ${output_dir}"
  INPUT_DIR="$input_dir" OUTPUT_DIR="$output_dir" DATASET_NAME="$DATASET_SLUG" NUM_SHARDS="$NUM_SHARDS" "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

input_dir = Path(os.environ["INPUT_DIR"]).expanduser().resolve()
output_dir = Path(os.environ["OUTPUT_DIR"]).expanduser().resolve()
dataset_name = os.environ["DATASET_NAME"]
num_shards = int(os.environ["NUM_SHARDS"])

rows = []
split_counts = {}
for split_name in ("train", "validation", "test"):
    path = input_dir / f"{split_name}.jsonl"
    if not path.exists():
        continue
    split_rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                split_rows.append(json.loads(line))
    split_counts[split_name] = len(split_rows)
    rows.extend(split_rows)
if not rows:
    raise FileNotFoundError(f"No train/validation/test JSONL files found under {input_dir}")

output_dir.mkdir(parents=True, exist_ok=True)
def write_jsonl(path: Path, items: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

shards = [[] for _ in range(num_shards)]
for idx, row in enumerate(rows):
    shards[idx % num_shards].append(row)
write_jsonl(output_dir / "all.jsonl", rows)
for shard_idx, shard_rows in enumerate(shards):
    write_jsonl(output_dir / f"shard{shard_idx}.jsonl", shard_rows)

summary = {
    "dataset_name": dataset_name,
    "input_dir": str(input_dir),
    "output_dir": str(output_dir),
    "num_rows_total": len(rows),
    "split_counts": split_counts,
    "num_shards": num_shards,
    "shard_sizes": [len(shard) for shard in shards],
}
(output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2), flush=True)
PY
}

run_generation_shard() {
  local split_label="$1"
  local input_path="$2"
  local run_dir="$3"
  local num_samples="$4"
  local log_path="$5"
  local visible_gpu_csv="${6:-$GPU_IDS_CSV}"
  local shard_tp_size="${7:-$TP_SIZE}"

  if [[ "$SKIP_GENERATION" == "1" ]]; then
    log "[${split_label}] skipping generation"
    return 0
  fi
  if [[ "$OVERWRITE" != "1" && -f "${run_dir}/all_experiments.jsonl" && -f "${run_dir}/evaluation_results.jsonl" ]]; then
    log "[${split_label}] generation shard already exists: ${run_dir}"
    return 0
  fi

  wait_for_gpu_group "$visible_gpu_csv"
  mkdir -p "$(dirname "$log_path")"
  log "[${split_label}][gpus=${visible_gpu_csv}][tp=${shard_tp_size}] generation -> ${log_path}"
  CUDA_VISIBLE_DEVICES="$visible_gpu_csv" PYTHONPATH="$ROOT" "$PYTHON" -u -m classifer_training.sample \
    --model_name_or_path "$MODEL_LOAD_NAME_OR_PATH" \
    --input_path "$input_path" \
    --dataset_name "$DATASET_SLUG" \
    --output_dir "$run_dir" \
    --backend vllm \
    --grader math_verify \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --top_k "$TOP_K" \
    --max_new_tokens "$MATH_MAX_NEW_TOKENS" \
    --batch_size "$GEN_BATCH_SIZE" \
    --seed "$SEED" \
    --num_samples "$num_samples" \
    --tensor_parallel_size "$shard_tp_size" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    "${TRUST_FLAG[@]}" \
    "${ENFORCE_EAGER_FLAG[@]}" \
    "${CUSTOM_ALL_REDUCE_FLAG[@]}" \
    "${OVERWRITE_FLAG[@]}" \
    > "$log_path" 2>&1
}

collect_run_dirs() {
  local run_root="$1"
  local -n out_ref="$2"
  out_ref=()
  if [[ ! -d "$run_root" ]]; then
    return 0
  fi
  while IFS= read -r dir; do
    [[ -n "$dir" ]] && out_ref+=("$dir")
  done < <(find "$run_root" -mindepth 1 -maxdepth 1 -type d | sort)
}

verify_reused_train_run_root() {
  if [[ -z "$REUSE_TRAIN_RUN_ROOT" ]]; then
    return 0
  fi

  require_dir "$TRAIN_RUN_ROOT"
  local train_run_dirs=()
  collect_run_dirs "$TRAIN_RUN_ROOT" train_run_dirs
  if [[ "${#train_run_dirs[@]}" -eq 0 ]]; then
    echo "No train generation run dirs found under ${TRAIN_RUN_ROOT}" >&2
    exit 1
  fi

  local run_dir
  for run_dir in "${train_run_dirs[@]}"; do
    if [[ ! -f "${run_dir}/all_experiments.jsonl" || ! -f "${run_dir}/evaluation_results.jsonl" ]]; then
      echo "Incomplete reused train run dir: ${run_dir}" >&2
      exit 1
    fi
  done
  log "[train] reusing generation run dirs from ${TRAIN_RUN_ROOT}"
}

run_generation_split_tp() {
  local split_name="$1"
  local shard_dir="$2"
  local run_root="$3"
  local num_samples="$4"

  local shard_path shard_base run_dir log_path split_label
  for shard_path in "$shard_dir"/shard*.jsonl; do
    [[ -f "$shard_path" ]] || continue
    shard_base="$(basename "${shard_path%.jsonl}")"
    run_dir="${run_root}/${split_name}_${shard_base}"
    log_path="${LOG_ROOT}/${split_name}_generation.${shard_base}.log"
    split_label="${split_name}:${shard_base}"
    run_generation_shard "$split_label" "$shard_path" "$run_dir" "$num_samples" "$log_path" "$GPU_IDS_CSV" "$TP_SIZE"
  done
}

run_generation_split_shard() {
  local split_name="$1"
  local shard_dir="$2"
  local run_root="$3"
  local num_samples="$4"

  local shard_paths=()
  local shard_path
  for shard_path in "$shard_dir"/shard*.jsonl; do
    [[ -f "$shard_path" ]] || continue
    shard_paths+=("$shard_path")
  done
  if [[ "${#shard_paths[@]}" -eq 0 ]]; then
    return 0
  fi

  local start_index=0
  local total_shards="${#shard_paths[@]}"
  while [[ "$start_index" -lt "$total_shards" ]]; do
    local pids=()
    local launched=0
    local gpu shard_index shard_base run_dir log_path split_label
    for gpu in "${GPU_IDS[@]}"; do
      shard_index=$((start_index + launched))
      if [[ "$shard_index" -ge "$total_shards" ]]; then
        break
      fi
      shard_path="${shard_paths[$shard_index]}"
      shard_base="$(basename "${shard_path%.jsonl}")"
      run_dir="${run_root}/${split_name}_${shard_base}"
      log_path="${LOG_ROOT}/${split_name}_generation.${shard_base}.log"
      split_label="${split_name}:${shard_base}"
      run_generation_shard "$split_label" "$shard_path" "$run_dir" "$num_samples" "$log_path" "$gpu" "1" &
      pids+=("$!")
      launched=$((launched + 1))
    done
    wait_for_pids "${split_name} generation" "${pids[@]}"
    start_index=$((start_index + launched))
  done
}

run_generation_split() {
  local split_name="$1"
  local shard_dir="$2"
  local run_root="$3"
  local num_samples="$4"

  if [[ "$split_name" == "train" && -n "$REUSE_TRAIN_RUN_ROOT" ]]; then
    verify_reused_train_run_root
    return 0
  fi

  if [[ "$GENERATION_PARALLELISM" == "shard" ]]; then
    run_generation_split_shard "$split_name" "$shard_dir" "$run_root" "$num_samples"
  else
    run_generation_split_tp "$split_name" "$shard_dir" "$run_root" "$num_samples"
  fi
}

run_labels() {
  local train_names=("$@")
  if [[ "$SKIP_LABELS" == "1" ]]; then
    log "[labels] skipping"
    return 0
  fi
  if [[ "$OVERWRITE" != "1" && -f "$LABELS_PATH" && -f "$LABELS_SUMMARY" ]]; then
    log "[labels] already exist: ${LABELS_PATH}"
    return 0
  fi
  mkdir -p "$(dirname "$LABELS_PATH")"

  local train_run_dirs=()
  local validation_run_dirs=()
  collect_run_dirs "$TRAIN_RUN_ROOT" train_run_dirs
  collect_run_dirs "$VALIDATION_RUN_ROOT" validation_run_dirs
  if [[ "${#train_run_dirs[@]}" -eq 0 || "${#validation_run_dirs[@]}" -eq 0 ]]; then
    echo "Missing train/validation run dirs for label aggregation." >&2
    exit 1
  fi

  local args=(--run_dirs "${train_run_dirs[@]}" "${validation_run_dirs[@]}"
              --prompt_dataset_dir "$LABELS_SCRATCH"
              --labels_path "$LABELS_PATH"
              --summary_path "$LABELS_SUMMARY")
  local run_dir
  for run_dir in "${train_run_dirs[@]}"; do
    args+=(--train_run_dir_names "$(basename "$run_dir")")
  done
  for run_dir in "${validation_run_dirs[@]}"; do
    args+=(--validation_run_dir_names "$(basename "$run_dir")")
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
    --input_path "${DEEPSCALER_PROMPT_SHARD_DIR}/shard${shard}.jsonl" \
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
    log "[prompt] skipping prompt hidden"
    return 0
  fi
  local pids=()
  local shard gpu
  for ((shard=0; shard<NUM_SHARDS; shard++)); do
    gpu="${GPU_IDS[$((shard % ${#GPU_IDS[@]}))]}"
    run_prompt_shard "$gpu" "$shard" &
    pids+=("$!")
  done
  wait_for_pids "prompt hidden" "${pids[@]}"
}

run_rollout_shard() {
  local gpu="$1"
  local shard="$2"
  shift 2
  local run_dirs=("$@")
  local suffix
  suffix="$(printf 'shard%02dof%02d' "$shard" "$NUM_SHARDS")"
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
    log "[rollout] skipping rollout hidden"
    return 0
  fi
  local train_run_dirs=()
  local validation_run_dirs=()
  local all_run_dirs=()
  collect_run_dirs "$TRAIN_RUN_ROOT" train_run_dirs
  collect_run_dirs "$VALIDATION_RUN_ROOT" validation_run_dirs
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
  wait_for_pids "rollout hidden" "${pids[@]}"
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

write_manifest() {
  local train_run_dirs=()
  local validation_run_dirs=()
  collect_run_dirs "$TRAIN_RUN_ROOT" train_run_dirs
  collect_run_dirs "$VALIDATION_RUN_ROOT" validation_run_dirs
  MANIFEST_PATH="${LOG_ROOT}/qwen3_4b_base_deepscaler_custom_manifest.json" \
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
  DEEPSCALER_DATASET_DIR="$DEEPSCALER_DATASET_DIR" \
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
prompt_slug = os.environ["PROMPT_MODEL_SLUG"]
rollout_slug = os.environ["ROLLOUT_MODEL_SLUG"]
dataset_slug = os.environ["DATASET_SLUG"]

def prompt_paths() -> dict[str, list[str]]:
    hidden, index = [], []
    for shard in range(num_shards):
        dataset_shard = f"{dataset_slug}_shard{shard}"
        hidden.append(str((root / "classifer_training/artifacts/hidden" / dataset_shard / prompt_slug / "hidden_states.pt").resolve()))
        index.append(str((root / "classifer_training/artifacts/index" / dataset_shard / prompt_slug / "index.jsonl").resolve()))
    return {"hidden": hidden, "index": index}

def rollout_paths(dataset_name: str) -> dict[str, list[str]]:
    hidden, index = [], []
    for shard in range(num_shards):
        suffix = f"shard{shard:02d}of{num_shards:02d}"
        hidden.append(str((root / "classifer_training/artifacts/rollout_hidden" / dataset_name / rollout_slug / f"rollout_hidden_states.{suffix}.pt").resolve()))
        index.append(str((root / "classifer_training/artifacts/rollout_index" / dataset_name / rollout_slug / f"rollout_index.{suffix}.jsonl").resolve()))
    return {"hidden": hidden, "index": index}

manifest = {
    "model_name_or_path": os.environ["MODEL_NAME"],
    "load_model_name_or_path": os.environ["MODEL_LOAD_NAME_OR_PATH"],
    "model_slug": os.environ["MODEL_SLUG"],
    "prompt_model_slug": prompt_slug,
    "rollout_model_slug": rollout_slug,
    "dataset_slug": dataset_slug,
    "dataset_dir": os.environ["DEEPSCALER_DATASET_DIR"],
    "labels_path": os.environ["LABELS_PATH"],
    "num_shards": num_shards,
    "selected_layers": os.environ["LAYERS"],
    "prompt_last_n_values": os.environ["PROMPT_LAST_N_VALUES"],
    "rollout_components": os.environ["ROLLOUT_COMPONENTS"].split(),
    "train_run_dirs": [line for line in os.environ.get("TRAIN_RUN_DIRS", "").splitlines() if line],
    "validation_run_dirs": [line for line in os.environ.get("VALIDATION_RUN_DIRS", "").splitlines() if line],
    "prompt": prompt_paths(),
    "rollout_dataset_name": os.environ["RESPONSE_DATASET_NAME"],
    "rollout": rollout_paths(os.environ["RESPONSE_DATASET_NAME"]),
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
log "GPU_IDS=${GPU_IDS_CSV}"
log "NUM_SHARDS=${NUM_SHARDS}"
log "GENERATION_PARALLELISM=${GENERATION_PARALLELISM}"
log "TP_SIZE=${TP_SIZE}"
log "TRAIN_PROMPTS=${TRAIN_PROMPTS}"
log "VALIDATION_PROMPTS=${VALIDATION_PROMPTS}"
log "TRAIN_NUM_SAMPLES=${TRAIN_NUM_SAMPLES}"
log "VALIDATION_NUM_SAMPLES=${VALIDATION_NUM_SAMPLES}"
log "TRAIN_GENERATION_SHARD_SIZE=${TRAIN_GENERATION_SHARD_SIZE}"
log "VALIDATION_GENERATION_SHARD_SIZE=${VALIDATION_GENERATION_SHARD_SIZE}"
log "REUSE_TRAIN_FROM_VALIDATION_PROMPTS=${REUSE_TRAIN_FROM_VALIDATION_PROMPTS:-}"
log "REUSE_TRAIN_DATASET_DIR=${REUSE_TRAIN_DATASET_DIR:-}"
log "REUSE_TRAIN_RUN_ROOT=${REUSE_TRAIN_RUN_ROOT:-}"
log "LAYERS=${LAYERS}"
log "PROMPT_LAST_N_VALUES=${PROMPT_LAST_N_VALUES_CSV}"
log "ROLLOUT_COMPONENTS=${ROLLOUT_COMPONENTS}"
log "LOG_ROOT=${LOG_ROOT}"

prepare_custom_dataset
require_dir "$DEEPSCALER_DATASET_DIR"
prepare_prompt_shards "$DEEPSCALER_DATASET_DIR" "$DEEPSCALER_PROMPT_SHARD_DIR"
run_generation_split "train" "${DEEPSCALER_DATASET_DIR}/train_generation_shards" "$TRAIN_RUN_ROOT" "$TRAIN_NUM_SAMPLES"
run_generation_split "validation" "${DEEPSCALER_DATASET_DIR}/validation_generation_shards" "$VALIDATION_RUN_ROOT" "$VALIDATION_NUM_SAMPLES"
run_labels
run_parallel_prompt
run_parallel_rollout
write_manifest
log "Done. Manifest: ${LOG_ROOT}/qwen3_4b_base_deepscaler_custom_manifest.json"
