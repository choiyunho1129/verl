#!/usr/bin/env bash

set -euo pipefail

ROOT="/home/jongwonlim/verl/yoonho/verl"
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
RUN_ROOT="$ROOT/classifer_training/artifacts/runs/dapo_math_17k/deepseek-ai_DeepSeek-R1-Distill-Qwen-1_5B"
HIDDEN_ROOT="$ROOT/classifer_training/artifacts/rollout_hidden"
INDEX_ROOT="$ROOT/classifer_training/artifacts/rollout_index"
LOG_ROOT="/tmp/dapo_rollout_extract_all16"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_BATCH_TOKENS="${MAX_BATCH_TOKENS:-24000}"
GPU_IDS="${GPU_IDS:-0 1 2 3}"
COMPONENTS="${COMPONENTS:-prompt_hidden response_hidden}"
HIDDEN_FILENAME="${HIDDEN_FILENAME:-all16_rollout_hidden_states.pt}"
INDEX_FILENAME="${INDEX_FILENAME:-all16_rollout_index.jsonl}"

mkdir -p "$LOG_ROOT"

read -r -a GPU_ID_ARRAY <<<"$GPU_IDS"
NUM_SHARDS="${#GPU_ID_ARRAY[@]}"
if [[ "$NUM_SHARDS" -lt 1 ]]; then
  echo "GPU_IDS must contain at least one GPU id." >&2
  exit 1
fi

declare -a RUN_DIRS=()
for seed in $(seq 1 16); do
  RUN_DIRS+=("$RUN_ROOT/temp0.7_seed${seed}")
done

launch_shard() {
  local gpu_id="$1"
  local shard_index="$2"
  local log_path="$LOG_ROOT/shard${shard_index}.log"
  nohup env CUDA_VISIBLE_DEVICES="$gpu_id" python -u -m classifer_training.extract_rollout_hidden_states \
    --model_name_or_path "$MODEL_NAME" \
    --run_dirs "${RUN_DIRS[@]}" \
    --dataset_name dapo_math_17k \
    --components ${COMPONENTS} \
    --layers 27 \
    --response_anchor reasoning_or_answer \
    --batch_size "$BATCH_SIZE" \
    --max_batch_tokens "$MAX_BATCH_TOKENS" \
    --num_shards "$NUM_SHARDS" \
    --shard_index "$shard_index" \
    --hidden_root "$HIDDEN_ROOT" \
    --index_root "$INDEX_ROOT" \
    --hidden_filename "$HIDDEN_FILENAME" \
    --index_filename "$INDEX_FILENAME" \
    --trust_remote_code \
    --local_files_only \
    --overwrite \
    >"$log_path" 2>&1 &
  echo "$!" >"$LOG_ROOT/shard${shard_index}.pid"
  echo "launched shard $((shard_index + 1))/${NUM_SHARDS} on GPU ${gpu_id}: $log_path"
}

for shard_index in "${!GPU_ID_ARRAY[@]}"; do
  launch_shard "${GPU_ID_ARRAY[$shard_index]}" "$shard_index"
done

echo
echo "Logs:"
for shard_index in "${!GPU_ID_ARRAY[@]}"; do
  echo "  tail -f $LOG_ROOT/shard${shard_index}.log"
done
echo
echo "Batching:"
echo "  BATCH_SIZE=$BATCH_SIZE"
echo "  MAX_BATCH_TOKENS=$MAX_BATCH_TOKENS"
echo "  GPU_IDS=$GPU_IDS"
echo "  COMPONENTS=$COMPONENTS"
echo "  HIDDEN_FILENAME=$HIDDEN_FILENAME"
echo "  INDEX_FILENAME=$INDEX_FILENAME"
echo
echo "PIDs:"
for shard_index in "${!GPU_ID_ARRAY[@]}"; do
  echo "  $(cat "$LOG_ROOT/shard${shard_index}.pid")"
done
