#!/usr/bin/env bash
# Sample SPO train subsets first, extract train hidden states, then sample eval
# subsets and extract eval hidden states. Hidden components are think-end last-10
# states for layers 18..22.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON="${PYTHON:-python3}"

SOURCE_SPO_ROOT="${SOURCE_SPO_ROOT:-/NHNHOME/WORKSPACE/26msit006_A/kisti/snu/yunhochoi/crrl/spo_verl_pr/spo}"
DATASET_NAME="${DATASET_NAME:-spo_temp1_subset0to3_s0_1_4096x2_s2_3_1024x16_l18_22_last10}"
DATASET_DIR="${DATASET_DIR:-${ROOT}/classifer_training/artifacts/datasets/${DATASET_NAME}}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/classifer_training/artifacts/runs/${DATASET_NAME}/sampled_runs}"
LOG_DIR="${LOG_DIR:-${ROOT}/classifer_training/artifacts/logs/${DATASET_NAME}}"

MODEL_NAME="${MODEL_NAME:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
MODEL_LOAD_NAME_OR_PATH="${MODEL_LOAD_NAME_OR_PATH:-$MODEL_NAME}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-${ROOT}/classifer_training/artifacts/hf_cache}"
MODEL_SLUG="${MODEL_SLUG:-deepseek_r1_distill_qwen_1_5b_l18_22_last10}"

GPU_IDS_CSV="${GPU_IDS:-0,1,2,3}"
BACKEND="${BACKEND:-vllm}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
BATCH_SIZE="${BATCH_SIZE:-128}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-4096}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"

LAYERS="${LAYERS:-18:22}"
PROMPT_BATCH_SIZE="${PROMPT_BATCH_SIZE:-32}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
ROLLOUT_MAX_BATCH_TOKENS="${ROLLOUT_MAX_BATCH_TOKENS:-24000}"
ROLLOUT_COMPONENTS="${ROLLOUT_COMPONENTS:-think_end_last10_hidden}"

SEED="${SEED:-1}"
SHUFFLE_PROMPTS="${SHUFFLE_PROMPTS:-0}"
OVERWRITE="${OVERWRITE:-0}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
SKIP_SAMPLING="${SKIP_SAMPLING:-0}"
SKIP_PROMPT_HIDDEN="${SKIP_PROMPT_HIDDEN:-0}"
SKIP_ROLLOUT_HIDDEN="${SKIP_ROLLOUT_HIDDEN:-0}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"

mkdir -p "$DATASET_DIR" "$RUN_ROOT" "$LOG_DIR" "$MODEL_CACHE_DIR"

IFS=',' read -r -a GPU_IDS <<< "$GPU_IDS_CSV"
if [[ "${#GPU_IDS[@]}" -lt 1 ]]; then
  echo "At least one GPU id is required." >&2
  exit 2
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

OVERWRITE_FLAG=()
if [[ "$OVERWRITE" == "1" ]]; then
  OVERWRITE_FLAG=(--overwrite)
fi

TRUST_FLAG=()
if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  TRUST_FLAG=(--trust_remote_code)
fi

LOCAL_ONLY_FLAG=()
if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
  LOCAL_ONLY_FLAG=(--local_files_only)
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_OFFLINE=1
fi

CACHE_DIR_FLAG=(--cache_dir "$MODEL_CACHE_DIR")

prepare_datasets() {
  if [[ "$SKIP_PREPARE" == "1" ]]; then
    log "Skipping dataset preparation"
    return 0
  fi

  log "Preparing prompt datasets under ${DATASET_DIR}"
  SOURCE_SPO_ROOT="$SOURCE_SPO_ROOT" \
  DATASET_DIR="$DATASET_DIR" \
  DATASET_NAME="$DATASET_NAME" \
  SEED="$SEED" \
  SHUFFLE_PROMPTS="$SHUFFLE_PROMPTS" \
  "$PYTHON" - <<'PY'
import hashlib
import json
import os
import random
from pathlib import Path

source_root = Path(os.environ["SOURCE_SPO_ROOT"])
dataset_dir = Path(os.environ["DATASET_DIR"])
dataset_name = os.environ["DATASET_NAME"]
seed = int(os.environ["SEED"])
shuffle_prompts = os.environ["SHUFFLE_PROMPTS"] == "1"

plans = {
    0: {"max_prompts": 4096, "num_samples": 2, "phase": "train", "split": "train"},
    1: {"max_prompts": 4096, "num_samples": 2, "phase": "train", "split": "train"},
    2: {"max_prompts": 1024, "num_samples": 16, "phase": "eval", "split": "validation"},
    3: {"max_prompts": 1024, "num_samples": 16, "phase": "eval", "split": "validation"},
}

def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def clean_prompt(raw: str) -> str:
    text = str(raw or "").strip()
    if text.startswith("user\n") and text.endswith("assistant"):
        text = text[len("user\n") : -len("assistant")].strip()
    elif text.startswith("user\n") and text.endswith("assistant\n"):
        text = text[len("user\n") : -len("assistant\n")].strip()
    return text

def prompt_hash(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:16]

manifest = {
    "dataset_name": dataset_name,
    "source_spo_root": str(source_root),
    "shuffle_prompts": shuffle_prompts,
    "seed": seed,
    "subsets": [],
}
all_rows: list[dict] = []
phase_rows: dict[str, list[dict]] = {"train": [], "eval": []}

for subset_id, plan in plans.items():
    source_path = source_root / f"offline_value_estimation_subset_{subset_id}" / "validation_data" / "0.jsonl"
    source_rows = load_jsonl(source_path)

    by_prompt: dict[str, dict] = {}
    ordered_prompts: list[str] = []
    for source_row in source_rows:
        raw_prompt = str(source_row.get("input", ""))
        if raw_prompt in by_prompt:
            continue
        by_prompt[raw_prompt] = source_row
        ordered_prompts.append(raw_prompt)

    if shuffle_prompts:
        rng = random.Random(seed + subset_id)
        rng.shuffle(ordered_prompts)

    selected_prompts = ordered_prompts[: int(plan["max_prompts"])]
    rows: list[dict] = []
    for row_idx, raw_prompt in enumerate(selected_prompts):
        source_row = by_prompt[raw_prompt]
        prompt = clean_prompt(raw_prompt)
        task_id = f"subset{subset_id}_{prompt_hash(raw_prompt)}"
        row = {
            "dataset_name": dataset_name,
            "task_id": task_id,
            "split": str(plan["split"]),
            "user_input": prompt,
            "messages": [{"role": "user", "content": prompt}],
            "ground_truth": str(source_row.get("gts", source_row.get("ground_truth", ""))),
            "source_subset_id": subset_id,
            "source_row_index": row_idx,
            "source_validation_data": str(source_path),
        }
        rows.append(row)
        all_rows.append(row)
        phase_rows[str(plan["phase"])].append(row)

    subset_path = dataset_dir / f"subset_{subset_id}.jsonl"
    write_jsonl(subset_path, rows)
    manifest["subsets"].append(
        {
            "subset_id": subset_id,
            "input_path": str(subset_path),
            "num_prompts": len(rows),
            "num_samples": int(plan["num_samples"]),
            "phase": str(plan["phase"]),
            "split": str(plan["split"]),
            "source_unique_prompts": len(ordered_prompts),
            "source_path": str(source_path),
        }
    )

write_jsonl(dataset_dir / "train.jsonl", phase_rows["train"])
write_jsonl(dataset_dir / "eval.jsonl", phase_rows["eval"])
write_jsonl(dataset_dir / "all.jsonl", all_rows)
(dataset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(dataset_dir / "manifest.json")
PY
}

run_sampling_subset() {
  local subset_id="$1"
  local num_samples="$2"
  local gpu="$3"
  local phase="$4"
  local input_path="${DATASET_DIR}/subset_${subset_id}.jsonl"
  local output_dir="${RUN_ROOT}/offline_value_estimation_subset_${subset_id}"
  local log_path="${LOG_DIR}/sample_${phase}_subset${subset_id}_gpu${gpu}.log"

  require_file "$input_path"
  if [[ "$OVERWRITE" != "1" && -f "${output_dir}/all_experiments.jsonl" && -f "${output_dir}/evaluation_results.jsonl" ]]; then
    log "[sample][${phase}][subset${subset_id}] already exists; skipping"
    return 0
  fi

  log "[sample][${phase}][subset${subset_id}] prompts=$(wc -l < "$input_path") num_samples=${num_samples} gpu=${gpu} -> ${log_path}"
  CUDA_VISIBLE_DEVICES="$gpu" \
  PYTHONPATH="$ROOT" \
  "$PYTHON" -u -m classifer_training.sample \
    --model_name_or_path "$MODEL_NAME" \
    --input_path "$input_path" \
    --dataset_name "${DATASET_NAME}_${phase}" \
    --output_dir "$output_dir" \
    --backend "$BACKEND" \
    --grader math_verify \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --max_model_len "$MAX_MODEL_LEN" \
    --max_num_batched_tokens "$VLLM_MAX_NUM_BATCHED_TOKENS" \
    --max_num_seqs "$VLLM_MAX_NUM_SEQS" \
    --batch_size "$BATCH_SIZE" \
    --seed "$SEED" \
    --num_samples "$num_samples" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    "${TRUST_FLAG[@]}" \
    "${OVERWRITE_FLAG[@]}" \
    > "$log_path" 2>&1
}

run_sampling_train() {
  if [[ "$SKIP_SAMPLING" == "1" ]]; then
    log "Skipping train sampling"
    return 0
  fi

  run_sampling_subset 0 2 "${GPU_IDS[0]}" train
  run_sampling_subset 1 2 "${GPU_IDS[$((1 % ${#GPU_IDS[@]}))]}" train
}

run_sampling_eval() {
  if [[ "$SKIP_SAMPLING" == "1" ]]; then
    log "Skipping eval sampling"
    return 0
  fi

  run_sampling_subset 2 16 "${GPU_IDS[0]}" eval
  run_sampling_subset 3 16 "${GPU_IDS[$((1 % ${#GPU_IDS[@]}))]}" eval
}

run_prompt_hidden_phase() {
  local phase="$1"
  local input_file="$2"
  local output_dataset_name="${DATASET_NAME}_${phase}"
  if [[ "$SKIP_PROMPT_HIDDEN" == "1" ]]; then
    log "Skipping ${phase} prompt hidden extraction"
    return 0
  fi

  local hidden_path="${ROOT}/classifer_training/artifacts/hidden/${output_dataset_name}/${MODEL_SLUG}/hidden_states.pt"
  local index_path="${ROOT}/classifer_training/artifacts/index/${output_dataset_name}/${MODEL_SLUG}/index.jsonl"
  local log_path="${LOG_DIR}/prompt_hidden_${phase}_gpu${GPU_IDS[0]}.log"
  if [[ "$OVERWRITE" != "1" && -f "$hidden_path" && -f "$index_path" ]]; then
    log "[prompt-hidden][${phase}] already exists; skipping"
    return 0
  fi

  log "[prompt-hidden][${phase}] layers=${LAYERS} last10 gpu=${GPU_IDS[0]} -> ${log_path}"
  CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" \
  PYTHONPATH="$ROOT" \
  "$PYTHON" -u -m classifer_training.extract_hidden_states \
    --input_path "$input_file" \
    --model_name_or_path "$MODEL_NAME" \
    --load_model_name_or_path "$MODEL_LOAD_NAME_OR_PATH" \
    --model_slug "$MODEL_SLUG" \
    --dataset_name "$output_dataset_name" \
    --components hidden \
    --layers "$LAYERS" \
    --token_pooling lastn_mean \
    --last_n_values 10 \
    --batch_size "$PROMPT_BATCH_SIZE" \
    --cuda_device 0 \
    --hidden_root "${ROOT}/classifer_training/artifacts/hidden" \
    --index_root "${ROOT}/classifer_training/artifacts/index" \
    "${TRUST_FLAG[@]}" \
    "${LOCAL_ONLY_FLAG[@]}" \
    "${CACHE_DIR_FLAG[@]}" \
    "${OVERWRITE_FLAG[@]}" \
    > "$log_path" 2>&1
}

run_rollout_hidden_phase() {
  local phase="$1"
  local output_dataset_name="${DATASET_NAME}_${phase}"
  if [[ "$SKIP_ROLLOUT_HIDDEN" == "1" ]]; then
    log "Skipping ${phase} rollout hidden extraction"
    return 0
  fi

  local run_dirs=()
  if [[ "$phase" == "train" ]]; then
    run_dirs=(
      "${RUN_ROOT}/offline_value_estimation_subset_0"
      "${RUN_ROOT}/offline_value_estimation_subset_1"
    )
  elif [[ "$phase" == "eval" ]]; then
    run_dirs=(
      "${RUN_ROOT}/offline_value_estimation_subset_2"
      "${RUN_ROOT}/offline_value_estimation_subset_3"
    )
  else
    echo "Unknown rollout hidden phase: ${phase}" >&2
    exit 2
  fi
  local num_shards="${#GPU_IDS[@]}"
  local pids=()

  for shard in $(seq 0 $((num_shards - 1))); do
    local gpu="${GPU_IDS[$shard]}"
    local suffix
    suffix="$(printf 'shard%02dof%02d' "$shard" "$num_shards")"
    local log_path="${LOG_DIR}/rollout_hidden_${phase}_${suffix}_gpu${gpu}.log"

    log "[rollout-hidden][${phase}][${suffix}] layers=${LAYERS} components=${ROLLOUT_COMPONENTS} gpu=${gpu} -> ${log_path}"
    (
      CUDA_VISIBLE_DEVICES="$gpu" \
      PYTHONPATH="$ROOT" \
      "$PYTHON" -u -m classifer_training.extract_rollout_hidden_states \
        --model_name_or_path "$MODEL_NAME" \
        --load_model_name_or_path "$MODEL_LOAD_NAME_OR_PATH" \
        --model_slug "$MODEL_SLUG" \
        --run_dirs "${run_dirs[@]}" \
        --dataset_name "$output_dataset_name" \
        --components $ROLLOUT_COMPONENTS \
        --layers "$LAYERS" \
        --num_shards "$num_shards" \
        --shard_index "$shard" \
        --cuda_device 0 \
        --hidden_root "${ROOT}/classifer_training/artifacts/rollout_hidden" \
        --index_root "${ROOT}/classifer_training/artifacts/rollout_index" \
        --batch_size "$ROLLOUT_BATCH_SIZE" \
        --max_batch_tokens "$ROLLOUT_MAX_BATCH_TOKENS" \
        "${TRUST_FLAG[@]}" \
        "${LOCAL_ONLY_FLAG[@]}" \
        "${CACHE_DIR_FLAG[@]}" \
        "${OVERWRITE_FLAG[@]}" \
        > "$log_path" 2>&1
    ) &
    pids+=("$!")
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" != "0" ]]; then
    echo "Rollout hidden extraction failed. Check logs under ${LOG_DIR}" >&2
    exit 1
  fi
}

log "ROOT=${ROOT}"
log "SOURCE_SPO_ROOT=${SOURCE_SPO_ROOT}"
log "DATASET_NAME=${DATASET_NAME}"
log "MODEL_NAME=${MODEL_NAME}"
log "MODEL_LOAD_NAME_OR_PATH=${MODEL_LOAD_NAME_OR_PATH}"
log "GPU_IDS=${GPU_IDS_CSV}"
log "sampling: temp=${TEMPERATURE} top_p=${TOP_P} max_new_tokens=${MAX_NEW_TOKENS} max_model_len=${MAX_MODEL_LEN} batch_size=${BATCH_SIZE} max_num_batched_tokens=${VLLM_MAX_NUM_BATCHED_TOKENS} max_num_seqs=${VLLM_MAX_NUM_SEQS}"
log "hidden: layers=${LAYERS} prompt_last_n=10 rollout_components=${ROLLOUT_COMPONENTS}"

prepare_datasets
run_sampling_train
run_prompt_hidden_phase train "${DATASET_DIR}/train.jsonl"
run_rollout_hidden_phase train
run_sampling_eval
run_prompt_hidden_phase eval "${DATASET_DIR}/eval.jsonl"
run_rollout_hidden_phase eval

log "Done. Logs: ${LOG_DIR}"
