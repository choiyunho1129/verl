#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON="${PYTHON:-python3}"
INPUT_INDEX_DIR="${ROOT}/classifer_training/artifacts/rollout_index/spo_temp1_subset0to4/Qwen_Qwen3-4B"
RESCORED_INDEX_DIR="${ROOT}/classifer_training/artifacts/rollout_index/spo_temp1_subset0to4/Qwen_Qwen3-4B_dapo_score"
BASE_OUTPUT="${ROOT}/classifer_training/artifacts/probe/spo_temp1_subset0to4_qwen3_4b_base_rowr2_single_L19_last10_hidden_entropy3_dapo_labels"
SOLUTION_FIELD="generated_text_before_think"
GROUND_TRUTH_FIELD="ground_truth"
STRICT_BOX_VERIFY=1
FORCE_RESCORE=0
SKIP_RESCORE=0
RESCORE_ONLY=0
MAX_ROWS_PER_FILE=""

usage() {
  cat <<'EOF'
Usage:
  bash classifer_training/run_spo_l19_last10_hidden_entropy3_dapo_labels.sh [options]

Options:
  --python PATH                 Python executable. Default: python3 or $PYTHON.
  --input-index-dir PATH        Existing rollout index dir with old labels.
  --rescored-index-dir PATH     Output rollout index dir with math_dapo labels.
  --base-output PATH            Probe output root.
  --solution-field FIELD        Response text field for math_dapo. Default: generated_text_before_think.
  --ground-truth-field FIELD    Ground-truth field. Default: ground_truth.
  --strict-box-verify           Use math_dapo strict_box_verify=True. Default for this wrapper.
  --minerva-answer-pattern      Use math_dapo strict_box_verify=False.
  --force-rescore               Overwrite existing rescored index dir contents.
  --skip-rescore                Train from --rescored-index-dir without rescoring.
  --rescore-only                Stop after writing rescored rollout index files.
  --max-rows-per-file N         Debug/smoke: rescore only first N rows per shard.
  -h, --help                    Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON="$2"
      shift 2
      ;;
    --input-index-dir)
      INPUT_INDEX_DIR="$2"
      shift 2
      ;;
    --rescored-index-dir)
      RESCORED_INDEX_DIR="$2"
      shift 2
      ;;
    --base-output)
      BASE_OUTPUT="$2"
      shift 2
      ;;
    --solution-field)
      SOLUTION_FIELD="$2"
      shift 2
      ;;
    --ground-truth-field)
      GROUND_TRUTH_FIELD="$2"
      shift 2
      ;;
    --strict-box-verify)
      STRICT_BOX_VERIFY=1
      shift
      ;;
    --minerva-answer-pattern)
      STRICT_BOX_VERIFY=0
      shift
      ;;
    --force-rescore)
      FORCE_RESCORE=1
      shift
      ;;
    --skip-rescore)
      SKIP_RESCORE=1
      shift
      ;;
    --rescore-only)
      RESCORE_ONLY=1
      shift
      ;;
    --max-rows-per-file)
      MAX_ROWS_PER_FILE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
}

log "ROOT=${ROOT}"
log "PYTHON=${PYTHON}"
log "INPUT_INDEX_DIR=${INPUT_INDEX_DIR}"
log "RESCORED_INDEX_DIR=${RESCORED_INDEX_DIR}"
log "BASE_OUTPUT=${BASE_OUTPUT}"
log "LABEL_SOURCE=math_dapo.compute_score"
log "SOLUTION_FIELD=${SOLUTION_FIELD}"
log "STRICT_BOX_VERIFY=${STRICT_BOX_VERIFY}"
log "SETTING=L19 last10 prompt hidden + L19 last10 rollout hidden + entropy3 scalars"

if [[ "${SKIP_RESCORE}" -eq 0 ]]; then
  if [[ -s "${RESCORED_INDEX_DIR}/rescore_manifest.json" && "${FORCE_RESCORE}" -eq 0 ]]; then
    log "Rescored rollout index already exists; skipping rescore. Use --force-rescore to rebuild."
  else
    rescore_cmd=(
      "${PYTHON}" -m classifer_training.rescore_spo_rollout_index_dapo
      --input-dir "${INPUT_INDEX_DIR}"
      --output-dir "${RESCORED_INDEX_DIR}"
      --solution-field "${SOLUTION_FIELD}"
      --ground-truth-field "${GROUND_TRUTH_FIELD}"
    )
    if [[ "${STRICT_BOX_VERIFY}" -eq 1 ]]; then
      rescore_cmd+=(--strict-box-verify)
    fi
    if [[ "${FORCE_RESCORE}" -eq 1 ]]; then
      rescore_cmd+=(--overwrite)
    fi
    if [[ -n "${MAX_ROWS_PER_FILE}" ]]; then
      rescore_cmd+=(--max-rows-per-file "${MAX_ROWS_PER_FILE}")
    fi
    log "Starting DAPO rescoring"
    "${rescore_cmd[@]}"
    log "Finished DAPO rescoring"
  fi
fi

if [[ "${RESCORE_ONLY}" -eq 1 ]]; then
  exit 0
fi

export BASE_OUTPUT
export ROLLOUT_INDEX_DIR="${RESCORED_INDEX_DIR}"
export LABEL_SOURCE="math_dapo.compute_score generated from ${RESCORED_INDEX_DIR}"
export SINGLE_TIED_CONFIG=1
export TIED_N_VALUE=10
export TIED_LAYER=19
export SINGLE_NAME_SUFFIX="_hidden_entropy3_dapo_labels"
export ROLLOUT_SCALAR_KEYS_JSON='["output_mean_token_entropy","reasoning_mean_token_entropy","answer_mean_token_entropy"]'
export INCLUDE_PROMPT_HIDDEN=1
export INCLUDE_ROLLOUT_HIDDEN=1

log "Starting probe training"
"${PYTHON}" -m classifer_training.sweep_spo_base_rowr2_axis
log "Finished probe training"
