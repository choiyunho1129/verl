#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jongwonlim/verl/yoonho/verl"
MODEL_NAME="${MODEL_NAME:-/data2/sangjunsong/.cache/hf_hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}"
DATASET_NAME="${DATASET_NAME:-dapo_math_17k}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
PID0="${PID0:?PID0 is required}"
PID1="${PID1:?PID1 is required}"
SKIP_WAIT="${SKIP_WAIT:-0}"

SAMPLE_DIR0="$ROOT/classifer_training/artifacts/runs/dapo_math_17k/qwen3_4b_instruct_2507/temp0.7_multisample4_extra2000_v2_shard0_len12288_bs32_seed1"
SAMPLE_DIR1="$ROOT/classifer_training/artifacts/runs/dapo_math_17k/qwen3_4b_instruct_2507/temp0.7_multisample4_extra2000_v2_shard1_len12288_bs32_seed1"

LOG_ROOT="$ROOT/classifer_training/artifacts/logs/think_end_after_sampling"
mkdir -p "$LOG_ROOT"

wait_for_pid() {
  local pid="$1"
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
  done
}

if [[ "$SKIP_WAIT" == "1" ]]; then
  echo "Skipping wait for sampling jobs."
else
  echo "Waiting for sampling jobs PID0=$PID0 PID1=$PID1"
  wait_for_pid "$PID0"
  wait_for_pid "$PID1"
fi

for dir in "$SAMPLE_DIR0" "$SAMPLE_DIR1"; do
  if [[ ! -f "$dir/all_experiments.jsonl" ]]; then
    echo "Expected sampled outputs missing in $dir" >&2
    exit 1
  fi
done

RUN_DIRS=()
for seed in $(seq 1 16); do
  RUN_DIRS+=("$ROOT/classifer_training/artifacts/runs/dapo_math_17k/qwen3_4b_instruct_2507/temp0.7_seed${seed}")
done
RUN_DIRS+=("$SAMPLE_DIR0" "$SAMPLE_DIR1")

cd "$ROOT"

CUDA_VISIBLE_DEVICES="$GPU0" TOKENIZERS_PARALLELISM=false PYTHONPATH=. \
python -u -m classifer_training.extract_rollout_hidden_states \
  --model_name_or_path "$MODEL_NAME" \
  --run_dirs "${RUN_DIRS[@]}" \
  --dataset_name "$DATASET_NAME" \
  --components think_end_hidden think_end_last10_hidden \
  --layers 26 \
  --response_anchor reasoning \
  --hidden_root "$ROOT/classifer_training/artifacts/rollout_hidden" \
  --index_root "$ROOT/classifer_training/artifacts/rollout_index" \
  --hidden_filename finished16_plus_extra2000v2_think_end_l26.pt \
  --index_filename finished16_plus_extra2000v2_think_end_l26.jsonl \
  --num_shards 2 \
  --shard_index 0 \
  --batch_size 4 \
  --max_batch_tokens 16000 \
  --trust_remote_code \
  --overwrite \
  > "$LOG_ROOT/gpu${GPU0}_shard0.log" 2>&1 &
PID_EXTRACT0=$!

CUDA_VISIBLE_DEVICES="$GPU1" TOKENIZERS_PARALLELISM=false PYTHONPATH=. \
python -u -m classifer_training.extract_rollout_hidden_states \
  --model_name_or_path "$MODEL_NAME" \
  --run_dirs "${RUN_DIRS[@]}" \
  --dataset_name "$DATASET_NAME" \
  --components think_end_hidden think_end_last10_hidden \
  --layers 26 \
  --response_anchor reasoning \
  --hidden_root "$ROOT/classifer_training/artifacts/rollout_hidden" \
  --index_root "$ROOT/classifer_training/artifacts/rollout_index" \
  --hidden_filename finished16_plus_extra2000v2_think_end_l26.pt \
  --index_filename finished16_plus_extra2000v2_think_end_l26.jsonl \
  --num_shards 2 \
  --shard_index 1 \
  --batch_size 4 \
  --max_batch_tokens 16000 \
  --trust_remote_code \
  --overwrite \
  > "$LOG_ROOT/gpu${GPU1}_shard1.log" 2>&1 &
PID_EXTRACT1=$!

echo "Launched think-end extraction on GPUs $GPU0,$GPU1: $PID_EXTRACT0 $PID_EXTRACT1"
wait "$PID_EXTRACT0" "$PID_EXTRACT1"
