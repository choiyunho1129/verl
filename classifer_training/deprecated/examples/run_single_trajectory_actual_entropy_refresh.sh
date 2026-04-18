#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data2/jongwonlim/verl/yoonho/verl}"
exec "${ROOT}/classifer_training/examples/run_weak_single_trajectory_e2e.sh" "$@"
