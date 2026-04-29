#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data2/jongwonlim/verl/yoonho/verl}"
PYTHON_BIN="${PYTHON_BIN:-/home/jongwonlim/anaconda3/envs/CB/bin/python}"
POLL_SEC="${POLL_SEC:-60}"
SESSION_NAME="${SESSION_NAME:-thinkend_l19_train_after_extract}"
EXTRACT_DATASET_NAME="${EXTRACT_DATASET_NAME:-spo_temp1_subset0to4_thinkendlast10_l19}"
ROLLOUT_DIR="${ROLLOUT_DIR:-${ROOT}/classifer_training/artifacts/rollout_hidden/${EXTRACT_DATASET_NAME}/Qwen_Qwen3-4B}"
INDEX_DIR="${INDEX_DIR:-${ROOT}/classifer_training/artifacts/rollout_index/${EXTRACT_DATASET_NAME}/Qwen_Qwen3-4B}"
HIDDEN_PATH="${HIDDEN_PATH:-${ROLLOUT_DIR}/rollout_hidden_states.pt}"
INDEX_PATH="${INDEX_PATH:-${INDEX_DIR}/rollout_index.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/classifer_training/artifacts/probe/spo_temp1_subset0to4_qwen3_4b_base_rowr2_single_L19_thinkendlast10_hidden_scalar6}"
LOG_PATH="${LOG_PATH:-${ROOT}/classifer_training/artifacts/logs/spo_rowr2_pca_tied_single_L19_thinkendlast10_hidden_scalar6/run.log}"

mkdir -p "$(dirname "${LOG_PATH}")"

while true; do
  if [[ -f "${HIDDEN_PATH}" && -f "${INDEX_PATH}" ]] && ! pgrep -af "extract_rollout_hidden_states.*${EXTRACT_DATASET_NAME}" >/dev/null 2>&1; then
    break
  fi
  printf '[%s] waiting for think_end extraction finalize\n' "$(date '+%Y-%m-%d %H:%M:%S')" >> "${LOG_PATH}"
  sleep "${POLL_SEC}"
done

tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true
tmux new-session -d -s "${SESSION_NAME}" \
  "cd ${ROOT} && env PYTHONPATH=${ROOT} BASE_OUTPUT=${OUTPUT_DIR} SINGLE_TIED_CONFIG=1 TIED_LAYER=19 TIED_N_VALUE=10 SINGLE_ROLLOUT_COMPONENT=think_end_last10_hidden SINGLE_NAME_SUFFIX=_thinkend_hidden_scalar6 INCLUDE_PROMPT_HIDDEN=1 INCLUDE_ROLLOUT_HIDDEN=1 ROLLOUT_HIDDEN_DIR=${ROLLOUT_DIR} ROLLOUT_INDEX_DIR=${INDEX_DIR} ROLLOUT_SCALAR_KEYS_JSON='[\"output_mean_token_entropy\",\"reasoning_mean_token_entropy\",\"answer_mean_token_entropy\",\"output_length\",\"think_tokens\",\"answer_tokens\"]' ${PYTHON_BIN} classifer_training/sweep_spo_base_rowr2_axis.py >> ${LOG_PATH} 2>&1"

