#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jongwonlim/verl/yoonho/verl"
MANIFEST="${ROOT}/classifer_training/artifacts/manifests/dapo_math_17k_qwen3_4b_instruct_2507_promptonly_finished6_rollout_enriched.json"
OUT_DIR="${ROOT}/classifer_training/artifacts/models/dapo_math_17k_qwen3_4b_instruct_2507_two_random_traj_et_trainplusval"

cd "${ROOT}"

python -m classifer_training.train_prompt_two_trajectory \
  --manifest "${MANIFEST}" \
  --output_dir "${OUT_DIR}" \
  --train_splits train validation \
  --test_splits test \
  --model et \
  --n_estimators 2000 \
  --min_samples_leaf 5 \
  --max_features 0.5 \
  --train_pairs_per_prompt 4 \
  --test_pairs_per_prompt 1 \
  --random_seed 42
