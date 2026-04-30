#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON="${PYTHON:-/home/jongwonlim/anaconda3/envs/vllm311/bin/python}"
ROLLOUT_INDEX_DIR="${ROOT}/classifer_training/artifacts/rollout_index/spo_temp1_subset0to4/Qwen_Qwen3-4B_dapo_score"
OUTPUT_DIR="${ROOT}/classifer_training/artifacts/probe/spo_temp1_subset0to4_qwen3_4b_base_rowr2_L19_last10_hidden_entropy3_dapo_ablation"
SEED=1
SIZE_FRACTIONS="1,0.5,0.25,0.125"
BALANCE_TOTAL_PROMPTS=1200
BALANCE_HARD_TO_MID_RATIOS="0.5,1,2,4,8"
BALANCE_USE_MAX_FEASIBLE=0

usage() {
  cat <<'EOF'
Usage:
  bash classifer_training/run_spo_l19_last10_hidden_entropy3_dapo_ablation.sh [options]

Options:
  --python PATH                    Python executable. Default: sklearn 1.8.0 vllm311 env.
  --rollout-index-dir PATH         DAPO-rescored rollout index dir.
  --output-dir PATH                Ablation output dir.
  --seed N                         Sampling seed. Default: 1.
  --size-fractions CSV             Count ablation fractions. Default: 1,0.5,0.25,0.125.
  --balance-total-prompts N        Fixed prompt count for label-balance ablation. Default: 1200.
  --balance-hard-to-mid-ratios CSV Hard-label(0+1) : mid-label(0.5) ratios. Default: 0.5,1,2,4,8.
  --balance-use-max-feasible       Use max feasible prompt count separately for each ratio.
  --skip-count-ablation            Do not run data-count ablation.
  --skip-balance-ablation          Do not run label-balance ablation.
  -h, --help                       Show this help.
EOF
}

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON="$2"
      shift 2
      ;;
    --rollout-index-dir)
      ROLLOUT_INDEX_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --size-fractions)
      SIZE_FRACTIONS="$2"
      shift 2
      ;;
    --balance-total-prompts)
      BALANCE_TOTAL_PROMPTS="$2"
      shift 2
      ;;
    --balance-hard-to-mid-ratios)
      BALANCE_HARD_TO_MID_RATIOS="$2"
      shift 2
      ;;
    --balance-use-max-feasible)
      BALANCE_USE_MAX_FEASIBLE=1
      shift
      ;;
    --skip-count-ablation|--skip-balance-ablation)
      EXTRA_ARGS+=("$1")
      shift
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
log "ROLLOUT_INDEX_DIR=${ROLLOUT_INDEX_DIR}"
log "OUTPUT_DIR=${OUTPUT_DIR}"
log "SIZE_FRACTIONS=${SIZE_FRACTIONS}"
log "BALANCE_TOTAL_PROMPTS=${BALANCE_TOTAL_PROMPTS}"
log "BALANCE_HARD_TO_MID_RATIOS=${BALANCE_HARD_TO_MID_RATIOS}"
log "BALANCE_USE_MAX_FEASIBLE=${BALANCE_USE_MAX_FEASIBLE}"

if [[ "${BALANCE_USE_MAX_FEASIBLE}" == "1" ]]; then
  EXTRA_ARGS+=(--balance-use-max-feasible)
fi

"${PYTHON}" -m classifer_training.sweep_spo_l19_last10_dapo_ablation \
  --rollout-index-dir "${ROLLOUT_INDEX_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --seed "${SEED}" \
  --size-fractions "${SIZE_FRACTIONS}" \
  --balance-total-prompts "${BALANCE_TOTAL_PROMPTS}" \
  --balance-hard-to-mid-ratios "${BALANCE_HARD_TO_MID_RATIOS}" \
  "${EXTRA_ARGS[@]}"
