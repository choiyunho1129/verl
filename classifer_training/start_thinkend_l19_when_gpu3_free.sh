#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data2/jongwonlim/verl/yoonho/verl}"
PYTHON_BIN="${PYTHON_BIN:-/home/jongwonlim/anaconda3/envs/CB/bin/python}"
GPU_INDEX="${GPU_INDEX:-3}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B}"
MODEL_LOAD_NAME_OR_PATH="${MODEL_LOAD_NAME_OR_PATH:-${ROOT}/classifer_training/artifacts/models/Qwen_Qwen3-4B_merged_snapshot}"
MODEL_SLUG="${MODEL_SLUG:-Qwen_Qwen3-4B}"
DATASET_NAME="${DATASET_NAME:-spo_temp1_subset0to4_thinkendlast10_l19}"
CACHE_DIR="${CACHE_DIR:-/data2/sangjunsong/.cache/transformers}"
LOG_PATH="${LOG_PATH:-${ROOT}/classifer_training/artifacts/logs/spo_thinkendlast10_l19_extract/run.log}"
SESSION_NAME="${SESSION_NAME:-thinkend_l19_extract}"
WATCH_SESSION_NAME="${WATCH_SESSION_NAME:-thinkend_l19_watch}"
POLL_SEC="${POLL_SEC:-60}"

RUN_DIRS=(
  "${ROOT}/classifer_training/artifacts/runs/spo_temp1_subset0to4/imported_runs/offline_value_estimation_subset_0"
  "${ROOT}/classifer_training/artifacts/runs/spo_temp1_subset0to4/imported_runs/offline_value_estimation_subset_1"
  "${ROOT}/classifer_training/artifacts/runs/spo_temp1_subset0to4/imported_runs/offline_value_estimation_subset_2"
  "${ROOT}/classifer_training/artifacts/runs/spo_temp1_subset0to4/imported_runs/offline_value_estimation_subset_3"
  "${ROOT}/classifer_training/artifacts/runs/spo_temp1_subset0to4/imported_runs/offline_value_estimation_subset_4"
)

mkdir -p "$(dirname "${LOG_PATH}")"

gpu_uuid="$(nvidia-smi -i "${GPU_INDEX}" --query-gpu=gpu_uuid --format=csv,noheader | tr -d ' ')"

while true; do
  if pgrep -af "extract_rollout_hidden_states.*${DATASET_NAME}" >/dev/null 2>&1; then
    echo "Extraction for ${DATASET_NAME} is already running. Exiting watcher."
    exit 0
  fi

  busy_count="$(
    nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader \
      | awk -F',' -v uuid="${gpu_uuid}" '
          $1 ~ uuid { count += 1 }
          END { print count + 0 }
        '
  )"

  if [[ "${busy_count}" -eq 0 ]]; then
    break
  fi

  printf '[%s] GPU %s busy with %s compute app(s); waiting %ss\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "${GPU_INDEX}" "${busy_count}" "${POLL_SEC}" >> "${LOG_PATH}"
  sleep "${POLL_SEC}"
done

tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true
tmux new-session -d -s "${SESSION_NAME}" \
  "cd ${ROOT} && env PYTHONPATH=${ROOT} TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=${GPU_INDEX} ${PYTHON_BIN} -m classifer_training.extract_rollout_hidden_states --model_name_or_path ${MODEL_NAME} --load_model_name_or_path ${MODEL_LOAD_NAME_OR_PATH} --model_slug ${MODEL_SLUG} --run_dirs ${RUN_DIRS[*]} --dataset_name ${DATASET_NAME} --components think_end_last10_hidden --layers 19 --cuda_device 0 --hidden_root ${ROOT}/classifer_training/artifacts/rollout_hidden --index_root ${ROOT}/classifer_training/artifacts/rollout_index --batch_size 4 --max_batch_tokens 24000 --local_files_only --cache_dir ${CACHE_DIR} >> ${LOG_PATH} 2>&1"

tmux kill-session -t "${WATCH_SESSION_NAME}" 2>/dev/null || true

