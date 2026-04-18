#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/jongwonlim/anaconda3/envs/CB/bin/python}"
TRAINED_MODEL_NAME="${TRAINED_MODEL_NAME:-jaygala24/Qwen3-4B-GRPO-math-reasoning}"
TRAINED_MODEL_SLUG="${TRAINED_MODEL_SLUG:-jaygala24_Qwen3-4B-GRPO-math-reasoning}"
GPU_IDS="${GPU_IDS:-0 1 2 3}"
NUM_SAMPLES="${NUM_SAMPLES:-4}"
SEED="${SEED:-1}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-64}"
EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE:-4}"
MAX_BATCH_TOKENS="${MAX_BATCH_TOKENS:-16000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"
RUN_SUFFIX="${RUN_SUFFIX:-temp${TEMPERATURE}_topp${TOP_P}_topk${TOP_K}_multisample${NUM_SAMPLES}_vllm_tp4_seed${SEED}}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/classifer_training/artifacts/logs/dapo_weak4_trained_transfer_gpu_chain_${RUN_SUFFIX}}"

WEAK4_RUNS=(
  "${REPO_ROOT}/classifer_training/artifacts/runs/dapo_math_17k/qwen3_4b_instruct_2507/weak4_runs/0"
  "${REPO_ROOT}/classifer_training/artifacts/runs/dapo_math_17k/qwen3_4b_instruct_2507/weak4_runs/1"
)
WEAK4_GT_DATASET_DIR="${WEAK4_GT_DATASET_DIR:-${REPO_ROOT}/classifer_training/artifacts/datasets/dapo_math_17k_weak4_gt}"
WEAK4_VAL_SHARDS="${WEAK4_VAL_SHARDS:-${REPO_ROOT}/classifer_training/artifacts/datasets/dapo_math_17k_weak4_validation_shards4}"
WEAK4_TRAIN_SHARDS="${WEAK4_TRAIN_SHARDS:-${REPO_ROOT}/classifer_training/artifacts/datasets/dapo_math_17k_weak4_train_shards4}"

export PYTHONPATH="${REPO_ROOT}"
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p "${LOG_ROOT}"
PIPELINE_LOG="${LOG_ROOT}/pipeline.log"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

read -r GPU0 GPU1 GPU2 GPU3 <<<"${GPU_IDS}"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

all_exist() {
  local path
  for path in "$@"; do
    if [[ ! -e "${path}" ]]; then
      return 1
    fi
  done
  return 0
}

find_one() {
  local root="$1"
  local pattern="$2"
  find "${root}" -name "${pattern}" 2>/dev/null | head -n 1
}

prepare_weak4_groundtruth_dataset() {
  mkdir -p "${WEAK4_GT_DATASET_DIR}" "${WEAK4_VAL_SHARDS}" "${WEAK4_TRAIN_SHARDS}"
  WEAK4_RUNS_0="${WEAK4_RUNS[0]}" WEAK4_RUNS_1="${WEAK4_RUNS[1]}" WEAK4_GT_DATASET_DIR="${WEAK4_GT_DATASET_DIR}" WEAK4_VAL_SHARDS="${WEAK4_VAL_SHARDS}" WEAK4_TRAIN_SHARDS="${WEAK4_TRAIN_SHARDS}" \
  "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

run_dirs = [Path(os.environ["WEAK4_RUNS_0"]), Path(os.environ["WEAK4_RUNS_1"])]
dataset_dir = Path(os.environ["WEAK4_GT_DATASET_DIR"])
val_shards_dir = Path(os.environ["WEAK4_VAL_SHARDS"])
train_shards_dir = Path(os.environ["WEAK4_TRAIN_SHARDS"])

records = {}
for run_dir in run_dirs:
    with (run_dir / "all_experiments.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            task_id = str(row["task_id"])
            if task_id in records:
                continue
            records[task_id] = {
                "dataset_name": "dapo_math_17k_weak4",
                "task_id": task_id,
                "split": str(row["split"]),
                "user_input": row["user_input"],
                "ground_truth": row.get("ground_truth", ""),
                "messages": row["messages"],
                "source_file": str(run_dir.name),
            }

train_rows = sorted((r for r in records.values() if r["split"] == "train"), key=lambda x: x["task_id"])
val_rows = sorted((r for r in records.values() if r["split"] == "validation"), key=lambda x: x["task_id"])

def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def write_shards(root: Path, rows):
    root.mkdir(parents=True, exist_ok=True)
    shards = [[] for _ in range(4)]
    for idx, row in enumerate(rows):
        shards[idx % 4].append(row)
    for shard_idx, shard_rows in enumerate(shards):
        write_jsonl(root / f"shard{shard_idx}.jsonl", shard_rows)
    write_jsonl(root / "all.jsonl", rows)
    (root / "summary.json").write_text(json.dumps({
        "num_rows_total": len(rows),
        "num_shards": 4,
        "shard_sizes": [len(s) for s in shards],
    }, indent=2), encoding="utf-8")

write_jsonl(dataset_dir / "train.jsonl", train_rows)
write_jsonl(dataset_dir / "validation.jsonl", val_rows)
(dataset_dir / "summary.json").write_text(json.dumps({
    "num_train": len(train_rows),
    "num_validation": len(val_rows),
    "num_total": len(train_rows) + len(val_rows),
    "source_run_dirs": [str(p) for p in run_dirs],
}, indent=2), encoding="utf-8")

write_shards(train_shards_dir, train_rows)
write_shards(val_shards_dir, val_rows)
PY
}

run_prompt_hidden_shards() {
  local model_name="$1"
  local shard_dir="$2"
  local dataset_prefix="$3"
  local model_slug="$4"
  local log_prefix="$5"
  local pids=()
  for shard_idx in 0 1 2 3; do
    local gpu_var="GPU${shard_idx}"
    local gpu_id="${!gpu_var}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" -u -m classifer_training.extract_hidden_states \
      --model_name_or_path "${model_name}" \
      --input_path "${shard_dir}/shard${shard_idx}.jsonl" \
      --dataset_name "${dataset_prefix}_shard${shard_idx}" \
      --model_slug "${model_slug}" \
      --token_pooling lastn_mean \
      --last_n 6 \
      --batch_size 16 \
      --trust_remote_code \
      --overwrite \
      > "${LOG_ROOT}/${log_prefix}.shard${shard_idx}.log" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
}

run_response_hidden_shards() {
  local model_name="$1"
  local run_dir="$2"
  local dataset_name="$3"
  local log_prefix="$4"
  local pids=()
  for shard_idx in 0 1 2 3; do
    local gpu_var="GPU${shard_idx}"
    local gpu_id="${!gpu_var}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" -u -m classifer_training.extract_rollout_hidden_states \
      --model_name_or_path "${model_name}" \
      --run_dirs "${run_dir}" \
      --dataset_name "${dataset_name}" \
      --components think_end_hidden \
      --layers 26 \
      --response_anchor reasoning_or_answer \
      --hidden_root "${REPO_ROOT}/classifer_training/artifacts/rollout_hidden" \
      --index_root "${REPO_ROOT}/classifer_training/artifacts/rollout_index" \
      --num_shards 4 \
      --shard_index "${shard_idx}" \
      --batch_size "${EXTRACT_BATCH_SIZE}" \
      --max_batch_tokens "${MAX_BATCH_TOKENS}" \
      --trust_remote_code \
      --overwrite \
      > "${LOG_ROOT}/${log_prefix}.shard${shard_idx}.log" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
}

run_split_pipeline() {
  local split_name="$1"
  local input_path="$2"
  local shard_dir="$3"
  local dataset_name="dapo_math_17k_weak4_${split_name}"
  local run_dir="${REPO_ROOT}/classifer_training/artifacts/runs/dapo_math_17k/${TRAINED_MODEL_SLUG}/${RUN_SUFFIX}_weak4_${split_name}"
  local labels_path="${REPO_ROOT}/classifer_training/artifacts/labels/dapo_math_17k/${TRAINED_MODEL_SLUG}/${dataset_name}_${RUN_SUFFIX}_labels.jsonl"
  local labels_summary="${REPO_ROOT}/classifer_training/artifacts/labels/dapo_math_17k/${TRAINED_MODEL_SLUG}/${dataset_name}_${RUN_SUFFIX}_summary.json"
  local prompt_dataset_scratch="${REPO_ROOT}/classifer_training/artifacts/datasets/${dataset_name}_${RUN_SUFFIX}_${TRAINED_MODEL_SLUG}_labels_scratch"
  local response_dataset_name="${dataset_name}_${TRAINED_MODEL_SLUG}_${RUN_SUFFIX}_response_l26"

  if all_exist "${run_dir}/all_experiments.jsonl" "${run_dir}/evaluation_results.jsonl"; then
    echo "[skip] ${dataset_name} generation already exists $(timestamp)"
  else
    echo "[stage] ${dataset_name} generation $(timestamp)"
    "${PYTHON_BIN}" -u -m classifer_training.sample \
      --model_name_or_path "${TRAINED_MODEL_NAME}" \
      --input_path "${input_path}" \
      --dataset_name "${dataset_name}" \
      --output_dir "${run_dir}" \
      --backend vllm \
      --grader math_verify \
      --temperature "${TEMPERATURE}" \
      --top_p "${TOP_P}" \
      --top_k "${TOP_K}" \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      --batch_size "${GEN_BATCH_SIZE}" \
      --seed "${SEED}" \
      --num_samples "${NUM_SAMPLES}" \
      --tensor_parallel_size 4 \
      --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
      --trust_remote_code \
      --overwrite
  fi

  if all_exist "${labels_path}" "${labels_summary}"; then
    echo "[skip] ${dataset_name} labels already exist $(timestamp)"
  else
    echo "[stage] ${dataset_name} labels $(timestamp)"
    "${PYTHON_BIN}" -u -m classifer_training.build_weak_prompt_dataset_and_labels \
      --run_dirs "${run_dir}" \
      --prompt_dataset_dir "${prompt_dataset_scratch}" \
      --labels_path "${labels_path}" \
      --summary_path "${labels_summary}"
  fi

  if all_exist \
    "${REPO_ROOT}/classifer_training/artifacts/hidden/${dataset_name}_shard0/${TRAINED_MODEL_SLUG}/hidden_states.pt" \
    "${REPO_ROOT}/classifer_training/artifacts/hidden/${dataset_name}_shard1/${TRAINED_MODEL_SLUG}/hidden_states.pt" \
    "${REPO_ROOT}/classifer_training/artifacts/hidden/${dataset_name}_shard2/${TRAINED_MODEL_SLUG}/hidden_states.pt" \
    "${REPO_ROOT}/classifer_training/artifacts/hidden/${dataset_name}_shard3/${TRAINED_MODEL_SLUG}/hidden_states.pt"; then
    echo "[skip] ${dataset_name} prompt hidden already exists $(timestamp)"
  else
    echo "[stage] ${dataset_name} prompt hidden extraction $(timestamp)"
    run_prompt_hidden_shards "${TRAINED_MODEL_NAME}" "${shard_dir}" "${dataset_name}" "${TRAINED_MODEL_SLUG}" "${dataset_name}_prompt_hidden"
  fi

  if all_exist \
    "$(find_one "${REPO_ROOT}/classifer_training/artifacts/rollout_hidden/${response_dataset_name}" 'rollout_hidden_states.shard00of04.pt')" \
    "$(find_one "${REPO_ROOT}/classifer_training/artifacts/rollout_hidden/${response_dataset_name}" 'rollout_hidden_states.shard03of04.pt')" \
    "$(find_one "${REPO_ROOT}/classifer_training/artifacts/rollout_index/${response_dataset_name}" 'rollout_index.shard00of04.jsonl')" \
    "$(find_one "${REPO_ROOT}/classifer_training/artifacts/rollout_index/${response_dataset_name}" 'rollout_index.shard03of04.jsonl')"; then
    echo "[skip] ${dataset_name} response hidden already exists $(timestamp)"
  else
    echo "[stage] ${dataset_name} response hidden extraction $(timestamp)"
    run_response_hidden_shards "${TRAINED_MODEL_NAME}" "${run_dir}" "${response_dataset_name}" "${dataset_name}_response_hidden"
  fi
}

echo "[start] $(timestamp) dapo weak4 trained transfer gpu chain"
if all_exist "${WEAK4_GT_DATASET_DIR}/train.jsonl" "${WEAK4_GT_DATASET_DIR}/validation.jsonl" "${WEAK4_VAL_SHARDS}/shard0.jsonl" "${WEAK4_TRAIN_SHARDS}/shard3.jsonl"; then
  echo "[skip] weak4 ground-truth dataset already prepared $(timestamp)"
else
  echo "[stage] weak4 ground-truth dataset prep $(timestamp)"
  prepare_weak4_groundtruth_dataset
fi

# weak4 has validation + train only. Use validation first as the held-out test-like split.
run_split_pipeline "validation" "${WEAK4_GT_DATASET_DIR}/validation.jsonl" "${WEAK4_VAL_SHARDS}"
run_split_pipeline "train" "${WEAK4_GT_DATASET_DIR}/train.jsonl" "${WEAK4_TRAIN_SHARDS}"
echo "[done] $(timestamp) dapo weak4 trained transfer gpu chain"
