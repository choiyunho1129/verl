#!/usr/bin/env bash
# Generate Qwen3-4B base rollouts and extract prompt/rollout hidden states for
# the existing DeepScaleR val500/test500 and IFBench test300 datasets.
#
# Defaults:
#   - model identity: Qwen/Qwen3-4B
#   - local load path when --local-files-only:
#       classifer_training/artifacts/models/Qwen_Qwen3-4B_merged_snapshot
#   - hidden layers: 18:35
#   - prompt pools: last 5 / 10 / 15 mean
#   - rollout pools: response last 5 / 10 / 15 mean
#
# Typical use:
#   bash classifer_training/run_deepscaler_ifbench_qwen3_4b_base_4gpu.sh \
#     --gpu-ids 0,1,2,3 \
#     --local-files-only \
#     --model-cache-dir /data2/sangjunsong/.cache/transformers
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON="${PYTHON:-/home/jongwonlim/anaconda3/envs/CB/bin/python}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B}"
MODEL_LOAD_NAME_OR_PATH="${MODEL_LOAD_NAME_OR_PATH:-}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3}"

NUM_SAMPLES="${NUM_SAMPLES:-4}"
SEED="${SEED:-1}"
TEMPERATURE="${TEMPERATURE:-1}"
TOP_P="${TOP_P:-1}"
TOP_K="${TOP_K:--1}"
MATH_MAX_NEW_TOKENS="${MATH_MAX_NEW_TOKENS:-8192}"
IFBENCH_MAX_NEW_TOKENS="${IFBENCH_MAX_NEW_TOKENS:-8192}"
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

SKIP_DEEPSCALER="${SKIP_DEEPSCALER:-0}"
SKIP_IFBENCH="${SKIP_IFBENCH:-0}"
SKIP_GENERATION="${SKIP_GENERATION:-0}"
SKIP_LABELS="${SKIP_LABELS:-0}"
SKIP_PROMPT="${SKIP_PROMPT:-0}"
SKIP_ROLLOUT="${SKIP_ROLLOUT:-0}"

DEEPSCALER_DATASET_DIR_ENV_PROVIDED="${DEEPSCALER_DATASET_DIR+x}"
DEEPSCALER_SHARD_DIR_ENV_PROVIDED="${DEEPSCALER_SHARD_DIR+x}"
IFBENCH_INPUT_PATH_ENV_PROVIDED="${IFBENCH_INPUT_PATH+x}"
IFBENCH_DATASET_DIR_ENV_PROVIDED="${IFBENCH_DATASET_DIR+x}"
IFBENCH_SHARD_DIR_ENV_PROVIDED="${IFBENCH_SHARD_DIR+x}"
LOG_ROOT_ENV_PROVIDED="${LOG_ROOT+x}"

export PYTHONPATH="${ROOT}"
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

usage() {
  cat <<'EOF'
Usage:
  bash classifer_training/run_deepscaler_ifbench_qwen3_4b_base_4gpu.sh \
    --gpu-ids 0,1,2,3 \
    --local-files-only \
    --model-cache-dir /data2/sangjunsong/.cache/transformers

Key options:
  --skip-deepscaler        Run IFBench only.
  --skip-ifbench           Run DeepScaleR only.
  --skip-generation        Reuse existing rollout run dirs.
  --skip-labels            Skip label/rescore stages.
  --skip-prompt            Skip prompt hidden extraction.
  --skip-rollout           Skip rollout hidden extraction.
  --overwrite              Replace existing outputs.
  --layers 18:35           Hidden layers to save.
  --num-samples 4          Rollouts per prompt.
  --enable-custom-all-reduce
                           Re-enable vLLM custom all-reduce kernels.
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
    --num-samples) shift; NUM_SAMPLES="$1" ;;
    --seed) shift; SEED="$1" ;;
    --temperature) shift; TEMPERATURE="$1" ;;
    --top-p) shift; TOP_P="$1" ;;
    --top-k) shift; TOP_K="$1" ;;
    --math-max-new-tokens) shift; MATH_MAX_NEW_TOKENS="$1" ;;
    --ifbench-max-new-tokens) shift; IFBENCH_MAX_NEW_TOKENS="$1" ;;
    --gen-batch-size) shift; GEN_BATCH_SIZE="$1" ;;
    --prompt-batch-size) shift; PROMPT_BATCH_SIZE="$1" ;;
    --rollout-batch-size) shift; ROLLOUT_BATCH_SIZE="$1" ;;
    --rollout-max-batch-tokens) shift; ROLLOUT_MAX_BATCH_TOKENS="$1" ;;
    --gpu-memory-utilization) shift; GPU_MEMORY_UTILIZATION="$1" ;;
    --layers) shift; LAYERS="$1" ;;
    --prompt-last-n-values) shift; PROMPT_LAST_N_VALUES_CSV="$1" ;;
    --rollout-components) shift; ROLLOUT_COMPONENTS="$1" ;;
    --deepscaler-dataset-dir) shift; DEEPSCALER_DATASET_DIR="$1"; DEEPSCALER_DATASET_DIR_ENV_PROVIDED=1 ;;
    --deepscaler-shard-dir) shift; DEEPSCALER_SHARD_DIR="$1"; DEEPSCALER_SHARD_DIR_ENV_PROVIDED=1 ;;
    --ifbench-input-path) shift; IFBENCH_INPUT_PATH="$1"; IFBENCH_INPUT_PATH_ENV_PROVIDED=1 ;;
    --ifbench-dataset-dir) shift; IFBENCH_DATASET_DIR="$1"; IFBENCH_DATASET_DIR_ENV_PROVIDED=1 ;;
    --ifbench-shard-dir) shift; IFBENCH_SHARD_DIR="$1"; IFBENCH_SHARD_DIR_ENV_PROVIDED=1 ;;
    --log-root) shift; LOG_ROOT="$1"; LOG_ROOT_ENV_PROVIDED=1 ;;
    --overwrite) OVERWRITE=1 ;;
    --local-files-only) LOCAL_FILES_ONLY=1 ;;
    --trust-remote-code) TRUST_REMOTE_CODE=1 ;;
    --enforce-eager) ENFORCE_EAGER=1 ;;
    --enable-custom-all-reduce) DISABLE_CUSTOM_ALL_REDUCE=0 ;;
    --skip-wait) SKIP_WAIT=1 ;;
    --skip-deepscaler) SKIP_DEEPSCALER=1 ;;
    --skip-ifbench) SKIP_IFBENCH=1 ;;
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

IFS=',' read -r -a GPU_IDS <<< "$GPU_IDS_CSV"
NUM_SHARDS="${NUM_SHARDS:-${#GPU_IDS[@]}}"
TP_SIZE="${TP_SIZE:-${#GPU_IDS[@]}}"
if [[ "${#GPU_IDS[@]}" -lt 1 ]]; then
  echo "At least one GPU id is required." >&2
  exit 2
fi
if [[ "$NUM_SHARDS" -lt 1 ]]; then
  echo "NUM_SHARDS must be at least 1." >&2
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

if [[ -z "$DEEPSCALER_DATASET_DIR_ENV_PROVIDED" ]]; then
  DEEPSCALER_DATASET_DIR="${ROOT}/classifer_training/artifacts/datasets/deepscaler"
fi
if [[ -z "$DEEPSCALER_SHARD_DIR_ENV_PROVIDED" ]]; then
  DEEPSCALER_SHARD_DIR="${ROOT}/classifer_training/artifacts/datasets/deepscaler_val500_test500_shards${NUM_SHARDS}"
fi
if [[ -z "$IFBENCH_INPUT_PATH_ENV_PROVIDED" ]]; then
  IFBENCH_INPUT_PATH="${ROOT}/classifer_training/external/IFBench/data/IFBench_test.jsonl"
fi
if [[ -z "$IFBENCH_DATASET_DIR_ENV_PROVIDED" ]]; then
  IFBENCH_DATASET_DIR="${ROOT}/classifer_training/artifacts/datasets/ifbench_test"
fi
if [[ -z "$IFBENCH_SHARD_DIR_ENV_PROVIDED" ]]; then
  IFBENCH_SHARD_DIR="${ROOT}/classifer_training/artifacts/datasets/ifbench_test_shards${NUM_SHARDS}"
fi

py_sanitize_model_slug() {
  PYTHONPATH="$ROOT" "$PYTHON" - "$MODEL_NAME" <<'PY'
import sys
from classifer_training.utils import sanitize_name
print(sanitize_name(sys.argv[1]))
PY
}

MODEL_SLUG="${MODEL_SLUG:-$(py_sanitize_model_slug)}"
PROMPT_MODEL_SLUG="${PROMPT_MODEL_SLUG:-qwen3_4b_base_l18_35_last5_10_15mean}"
ROLLOUT_MODEL_SLUG="${ROLLOUT_MODEL_SLUG:-${MODEL_SLUG}_l18_35_last5_10_15mean}"

DEEPSCALER_RUN_SUFFIX="${DEEPSCALER_RUN_SUFFIX:-temp${TEMPERATURE}_topp${TOP_P}_topk${TOP_K}_multisample${NUM_SAMPLES}_val500_test500_vllm_tp${TP_SIZE}_seed${SEED}}"
IFBENCH_RUN_SUFFIX="${IFBENCH_RUN_SUFFIX:-temp${TEMPERATURE}_topp${TOP_P}_topk${TOP_K}_multisample${NUM_SAMPLES}_test300_vllm_tp${TP_SIZE}_seed${SEED}}"

if [[ -z "$LOG_ROOT_ENV_PROVIDED" ]]; then
  LOG_ROOT="${ROOT}/classifer_training/artifacts/logs/qwen3_4b_base_deepscaler_ifbench_existing_${DEEPSCALER_RUN_SUFFIX}"
fi
mkdir -p "$LOG_ROOT"
PIPELINE_LOG="${LOG_ROOT}/pipeline.log"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

DEEPSCALER_RUN_DIR="${ROOT}/classifer_training/artifacts/runs/deepscaler/${MODEL_SLUG}/${DEEPSCALER_RUN_SUFFIX}"
DEEPSCALER_LABELS_PATH="${ROOT}/classifer_training/artifacts/labels/deepscaler/${MODEL_SLUG}/deepscaler_${DEEPSCALER_RUN_SUFFIX}_labels.jsonl"
DEEPSCALER_LABELS_SUMMARY="${ROOT}/classifer_training/artifacts/labels/deepscaler/${MODEL_SLUG}/deepscaler_${DEEPSCALER_RUN_SUFFIX}_summary.json"
DEEPSCALER_LABELS_SCRATCH="${ROOT}/classifer_training/artifacts/datasets/deepscaler_${DEEPSCALER_RUN_SUFFIX}_${MODEL_SLUG}_labels_scratch"
DEEPSCALER_RESPONSE_DATASET_NAME="deepscaler_${DEEPSCALER_RUN_SUFFIX}_response_l18_35_last5_10_15mean"

IFBENCH_RUN_DIR="${ROOT}/classifer_training/artifacts/runs/ifbench/${MODEL_SLUG}/${IFBENCH_RUN_SUFFIX}"
IFBENCH_LABELS_PATH="${ROOT}/classifer_training/artifacts/labels/ifbench/${MODEL_SLUG}/ifbench_${IFBENCH_RUN_SUFFIX}_labels.jsonl"
IFBENCH_LABELS_SUMMARY="${ROOT}/classifer_training/artifacts/labels/ifbench/${MODEL_SLUG}/ifbench_${IFBENCH_RUN_SUFFIX}_summary.json"
IFBENCH_LABELS_SCRATCH="${ROOT}/classifer_training/artifacts/datasets/ifbench_${IFBENCH_RUN_SUFFIX}_${MODEL_SLUG}_labels_scratch"
IFBENCH_RESPONSE_DATASET_NAME="ifbench_${IFBENCH_RUN_SUFFIX}_response_l18_35_last5_10_15mean"

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

all_exist() {
  local path
  for path in "$@"; do
    if [[ ! -e "$path" ]]; then
      return 1
    fi
  done
  return 0
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

prepare_ifbench_dataset_if_needed() {
  if [[ -f "${IFBENCH_DATASET_DIR}/test.jsonl" ]]; then
    return 0
  fi
  log "Preparing IFBench normalized dataset under ${IFBENCH_DATASET_DIR}"
  PYTHONPATH="$ROOT" "$PYTHON" -u -m classifer_training.prepare_ifbench_dataset \
    --input_path "$IFBENCH_INPUT_PATH" \
    --output_dir "$IFBENCH_DATASET_DIR" \
    --shard_dir "$IFBENCH_SHARD_DIR" \
    --dataset_name ifbench_test \
    --num_shards "$NUM_SHARDS"
}

prepare_prompt_shards() {
  local input_dir="$1"
  local output_dir="$2"
  local dataset_name="$3"
  if [[ "$OVERWRITE" != "1" && -f "${output_dir}/summary.json" ]]; then
    local ok=1
    local shard
    for ((shard=0; shard<NUM_SHARDS; shard++)); do
      [[ -f "${output_dir}/shard${shard}.jsonl" ]] || ok=0
    done
    if [[ "$ok" == "1" ]]; then
      log "Prompt shards already exist: ${output_dir}"
      return 0
    fi
  fi

  log "Preparing ${NUM_SHARDS} prompt shards under ${output_dir}"
  INPUT_DIR="$input_dir" OUTPUT_DIR="$output_dir" DATASET_NAME="$dataset_name" NUM_SHARDS="$NUM_SHARDS" "$PYTHON" - <<'PY'
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

shard_suffix() {
  local shard="$1"
  printf "shard%02dof%02d" "$shard" "$NUM_SHARDS"
}

run_generation() {
  local dataset_label="$1"
  local input_path="$2"
  local dataset_name="$3"
  local run_dir="$4"
  local grader="$5"
  local max_new_tokens="$6"
  local log_path="$7"

  if [[ "$SKIP_GENERATION" == "1" ]]; then
    log "[${dataset_label}] skipping generation"
    return 0
  fi
  if [[ "$OVERWRITE" != "1" && -f "${run_dir}/all_experiments.jsonl" && -f "${run_dir}/evaluation_results.jsonl" ]]; then
    log "[${dataset_label}] generation already exists: ${run_dir}"
    return 0
  fi

  wait_for_all_gpus
  mkdir -p "$(dirname "$log_path")"
  log "[${dataset_label}] generation -> ${log_path}"
  CUDA_VISIBLE_DEVICES="$GPU_IDS_CSV" PYTHONPATH="$ROOT" "$PYTHON" -u -m classifer_training.sample \
    --model_name_or_path "$MODEL_LOAD_NAME_OR_PATH" \
    --input_path "$input_path" \
    --dataset_name "$dataset_name" \
    --output_dir "$run_dir" \
    --backend vllm \
    --grader "$grader" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --top_k "$TOP_K" \
    --max_new_tokens "$max_new_tokens" \
    --batch_size "$GEN_BATCH_SIZE" \
    --seed "$SEED" \
    --num_samples "$NUM_SAMPLES" \
    --tensor_parallel_size "$TP_SIZE" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    "${TRUST_FLAG[@]}" \
    "${ENFORCE_EAGER_FLAG[@]}" \
    "${CUSTOM_ALL_REDUCE_FLAG[@]}" \
    "${OVERWRITE_FLAG[@]}" \
    > "$log_path" 2>&1
}

run_labels() {
  local dataset_label="$1"
  local run_dir="$2"
  local labels_scratch="$3"
  local labels_path="$4"
  local summary_path="$5"
  local log_path="$6"

  if [[ "$SKIP_LABELS" == "1" ]]; then
    log "[${dataset_label}] skipping labels"
    return 0
  fi
  if [[ "$OVERWRITE" != "1" && -f "$labels_path" && -f "$summary_path" ]]; then
    log "[${dataset_label}] labels already exist: ${labels_path}"
    return 0
  fi

  mkdir -p "$(dirname "$log_path")" "$(dirname "$labels_path")"
  log "[${dataset_label}] labels -> ${log_path}"
  PYTHONPATH="$ROOT" "$PYTHON" -u -m classifer_training.build_weak_prompt_dataset_and_labels \
    --run_dirs "$run_dir" \
    --prompt_dataset_dir "$labels_scratch" \
    --labels_path "$labels_path" \
    --summary_path "$summary_path" \
    > "$log_path" 2>&1
}

run_ifbench_rescore_and_labels() {
  if [[ "$SKIP_LABELS" == "1" ]]; then
    log "[ifbench] skipping rescore + labels"
    return 0
  fi
  if [[ "$OVERWRITE" != "1" && -f "$IFBENCH_LABELS_PATH" && -f "$IFBENCH_LABELS_SUMMARY" && -f "${IFBENCH_RUN_DIR}/ifbench_loose_summary.json" ]]; then
    log "[ifbench] rescore + labels already exist: ${IFBENCH_LABELS_PATH}"
    return 0
  fi

  log "[ifbench] rescore + labels"
  PYTHONPATH="$ROOT" "$PYTHON" -u -m classifer_training.rescore_ifbench_run \
    --run_dir "$IFBENCH_RUN_DIR" \
    --ifbench_input_path "$IFBENCH_INPUT_PATH" \
    --mode loose \
    --overwrite \
    > "${LOG_ROOT}/ifbench_rescore.log" 2>&1
  run_labels "ifbench" "$IFBENCH_RUN_DIR" "$IFBENCH_LABELS_SCRATCH" "$IFBENCH_LABELS_PATH" "$IFBENCH_LABELS_SUMMARY" "${LOG_ROOT}/ifbench_labels.log"
}

run_prompt_shard() {
  local gpu="$1"
  local shard="$2"
  local shard_dir="$3"
  local dataset_prefix="$4"
  local log_prefix="$5"
  local dataset_shard="${dataset_prefix}_shard${shard}"
  local hidden_path="${ROOT}/classifer_training/artifacts/hidden/${dataset_shard}/${PROMPT_MODEL_SLUG}/hidden_states.pt"
  local index_path="${ROOT}/classifer_training/artifacts/index/${dataset_shard}/${PROMPT_MODEL_SLUG}/index.jsonl"
  local log_path="${LOG_ROOT}/${log_prefix}_prompt_hidden.shard${shard}.gpu${gpu}.log"

  if [[ "$OVERWRITE" != "1" && -f "$hidden_path" && -f "$index_path" ]]; then
    log "[${log_prefix}][prompt][shard${shard}][gpu${gpu}] already exists; skipping"
    return 0
  fi

  wait_for_gpu "$gpu"
  log "[${log_prefix}][prompt][shard${shard}][gpu${gpu}] extracting -> ${log_path}"
  PYTHONPATH="$ROOT" "$PYTHON" -u -m classifer_training.extract_hidden_states \
    --input_path "${shard_dir}/shard${shard}.jsonl" \
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

run_rollout_shard() {
  local gpu="$1"
  local shard="$2"
  local run_dir="$3"
  local response_dataset_name="$4"
  local log_prefix="$5"
  local suffix
  suffix="$(shard_suffix "$shard")"
  local hidden_path="${ROOT}/classifer_training/artifacts/rollout_hidden/${response_dataset_name}/${ROLLOUT_MODEL_SLUG}/rollout_hidden_states.${suffix}.pt"
  local index_path="${ROOT}/classifer_training/artifacts/rollout_index/${response_dataset_name}/${ROLLOUT_MODEL_SLUG}/rollout_index.${suffix}.jsonl"
  local log_path="${LOG_ROOT}/${log_prefix}_rollout_hidden.shard${shard}.gpu${gpu}.log"

  if [[ "$OVERWRITE" != "1" && -f "$hidden_path" && -f "$index_path" ]]; then
    log "[${log_prefix}][rollout][shard${shard}][gpu${gpu}] already exists; skipping"
    return 0
  fi

  wait_for_gpu "$gpu"
  log "[${log_prefix}][rollout][shard${shard}][gpu${gpu}] extracting -> ${log_path}"
  PYTHONPATH="$ROOT" "$PYTHON" -u -m classifer_training.extract_rollout_hidden_states \
    --model_name_or_path "$MODEL_NAME" \
    --load_model_name_or_path "$MODEL_LOAD_NAME_OR_PATH" \
    --model_slug "$ROLLOUT_MODEL_SLUG" \
    --run_dirs "$run_dir" \
    --dataset_name "$response_dataset_name" \
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

run_parallel_prompt() {
  local shard_dir="$1"
  local dataset_prefix="$2"
  local log_prefix="$3"
  if [[ "$SKIP_PROMPT" == "1" ]]; then
    log "[${log_prefix}] skipping prompt hidden"
    return 0
  fi
  local pids=()
  local shard gpu
  for ((shard=0; shard<NUM_SHARDS; shard++)); do
    gpu="${GPU_IDS[$((shard % ${#GPU_IDS[@]}))]}"
    run_prompt_shard "$gpu" "$shard" "$shard_dir" "$dataset_prefix" "$log_prefix" &
    pids+=("$!")
  done
  wait_for_pids "${log_prefix} prompt hidden" "${pids[@]}"
}

run_parallel_rollout() {
  local run_dir="$1"
  local response_dataset_name="$2"
  local log_prefix="$3"
  if [[ "$SKIP_ROLLOUT" == "1" ]]; then
    log "[${log_prefix}] skipping rollout hidden"
    return 0
  fi
  local pids=()
  local shard gpu
  for ((shard=0; shard<NUM_SHARDS; shard++)); do
    gpu="${GPU_IDS[$((shard % ${#GPU_IDS[@]}))]}"
    run_rollout_shard "$gpu" "$shard" "$run_dir" "$response_dataset_name" "$log_prefix" &
    pids+=("$!")
  done
  wait_for_pids "${log_prefix} rollout hidden" "${pids[@]}"
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
  MANIFEST_PATH="${LOG_ROOT}/qwen3_4b_base_deepscaler_ifbench_manifest.json" \
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
  DEEPSCALER_RUN_DIR="$DEEPSCALER_RUN_DIR" \
  DEEPSCALER_LABELS_PATH="$DEEPSCALER_LABELS_PATH" \
  DEEPSCALER_RESPONSE_DATASET_NAME="$DEEPSCALER_RESPONSE_DATASET_NAME" \
  IFBENCH_RUN_DIR="$IFBENCH_RUN_DIR" \
  IFBENCH_LABELS_PATH="$IFBENCH_LABELS_PATH" \
  IFBENCH_RESPONSE_DATASET_NAME="$IFBENCH_RESPONSE_DATASET_NAME" \
  "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["ROOT"])
num_shards = int(os.environ["NUM_SHARDS"])
prompt_slug = os.environ["PROMPT_MODEL_SLUG"]
rollout_slug = os.environ["ROLLOUT_MODEL_SLUG"]

def prompt_paths(prefix: str) -> dict[str, list[str]]:
    hidden, index = [], []
    for shard in range(num_shards):
        dataset_shard = f"{prefix}_shard{shard}"
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
    "num_shards": num_shards,
    "selected_layers": os.environ["LAYERS"],
    "prompt_last_n_values": os.environ["PROMPT_LAST_N_VALUES"],
    "rollout_components": os.environ["ROLLOUT_COMPONENTS"].split(),
    "datasets": {
        "deepscaler": {
            "run_dir": os.environ["DEEPSCALER_RUN_DIR"],
            "labels_path": os.environ["DEEPSCALER_LABELS_PATH"],
            "prompt": prompt_paths("deepscaler_val500_test500"),
            "rollout_dataset_name": os.environ["DEEPSCALER_RESPONSE_DATASET_NAME"],
            "rollout": rollout_paths(os.environ["DEEPSCALER_RESPONSE_DATASET_NAME"]),
        },
        "ifbench": {
            "run_dir": os.environ["IFBENCH_RUN_DIR"],
            "labels_path": os.environ["IFBENCH_LABELS_PATH"],
            "prompt": prompt_paths("ifbench_test"),
            "rollout_dataset_name": os.environ["IFBENCH_RESPONSE_DATASET_NAME"],
            "rollout": rollout_paths(os.environ["IFBENCH_RESPONSE_DATASET_NAME"]),
        },
    },
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
log "TP_SIZE=${TP_SIZE}"
log "NUM_SAMPLES=${NUM_SAMPLES}"
log "DISABLE_CUSTOM_ALL_REDUCE=${DISABLE_CUSTOM_ALL_REDUCE}"
log "LAYERS=${LAYERS}"
log "PROMPT_LAST_N_VALUES=${PROMPT_LAST_N_VALUES_CSV}"
log "ROLLOUT_COMPONENTS=${ROLLOUT_COMPONENTS}"
log "LOG_ROOT=${LOG_ROOT}"

if [[ "$SKIP_DEEPSCALER" != "1" ]]; then
  require_dir "$DEEPSCALER_DATASET_DIR"
  prepare_prompt_shards "$DEEPSCALER_DATASET_DIR" "$DEEPSCALER_SHARD_DIR" "deepscaler"
  run_generation "deepscaler" "$DEEPSCALER_DATASET_DIR" "deepscaler" "$DEEPSCALER_RUN_DIR" "math_verify" "$MATH_MAX_NEW_TOKENS" "${LOG_ROOT}/deepscaler_generation.log"
  run_labels "deepscaler" "$DEEPSCALER_RUN_DIR" "$DEEPSCALER_LABELS_SCRATCH" "$DEEPSCALER_LABELS_PATH" "$DEEPSCALER_LABELS_SUMMARY" "${LOG_ROOT}/deepscaler_labels.log"
  run_parallel_prompt "$DEEPSCALER_SHARD_DIR" "deepscaler_val500_test500" "deepscaler"
  run_parallel_rollout "$DEEPSCALER_RUN_DIR" "$DEEPSCALER_RESPONSE_DATASET_NAME" "deepscaler"
else
  log "Skipping DeepScaleR"
fi

if [[ "$SKIP_IFBENCH" != "1" ]]; then
  require_file "$IFBENCH_INPUT_PATH"
  prepare_ifbench_dataset_if_needed
  prepare_prompt_shards "$IFBENCH_DATASET_DIR" "$IFBENCH_SHARD_DIR" "ifbench_test"
  run_generation "ifbench" "$IFBENCH_DATASET_DIR" "ifbench_test" "$IFBENCH_RUN_DIR" "exact" "$IFBENCH_MAX_NEW_TOKENS" "${LOG_ROOT}/ifbench_generation.log"
  run_ifbench_rescore_and_labels
  run_parallel_prompt "$IFBENCH_SHARD_DIR" "ifbench_test" "ifbench"
  run_parallel_rollout "$IFBENCH_RUN_DIR" "$IFBENCH_RESPONSE_DATASET_NAME" "ifbench"
else
  log "Skipping IFBench"
fi

write_manifest
log "Done. Manifest: ${LOG_ROOT}/qwen3_4b_base_deepscaler_ifbench_manifest.json"
