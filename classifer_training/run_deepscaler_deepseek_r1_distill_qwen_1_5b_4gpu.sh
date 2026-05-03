#!/usr/bin/env bash
# Alias for the custom DeepScaleR DeepSeek-R1-Distill-Qwen-1.5B pipeline.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_deepscaler_custom_deepseek_r1_distill_qwen_1_5b_4gpu.sh" "$@"
