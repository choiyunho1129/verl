#!/usr/bin/env bash
# Thin wrapper around the combined DeepScaleR/IFBench pipeline that runs
# DeepScaleR only.
#
# This keeps the existing generation / labeling / prompt hidden /
# rollout hidden code paths in one place while giving us a stable
# DeepScaleR-specific entrypoint.
#
# Typical use:
#   bash classifer_training/run_deepscaler_qwen3_4b_base_4gpu.sh \
#     --gpu-ids 0,1 \
#     --local-files-only \
#     --model-cache-dir /data2/sangjunsong/.cache/transformers
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
TARGET_SCRIPT="${SCRIPT_DIR}/run_deepscaler_ifbench_qwen3_4b_base_4gpu.sh"

if [[ ! -f "${TARGET_SCRIPT}" ]]; then
  echo "Missing target script: ${TARGET_SCRIPT}" >&2
  exit 1
fi

has_skip_ifbench=0
for arg in "$@"; do
  if [[ "${arg}" == "--skip-ifbench" ]]; then
    has_skip_ifbench=1
    break
  fi
done

args=("$@")
if [[ "${has_skip_ifbench}" != "1" ]]; then
  args+=(--skip-ifbench)
fi

exec bash "${TARGET_SCRIPT}" "${args[@]}"
