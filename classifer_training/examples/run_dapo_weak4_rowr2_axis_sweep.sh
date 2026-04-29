#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/jongwonlim/anaconda3/envs/CB/bin/python}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-dapo-rowr2}"

BASE_OUTPUT="${BASE_OUTPUT:-classifer_training/artifacts/probe/dapo_math_17k_weak4_simple_ridge_entropy_rowr2_axis_sweep}"
mkdir -p "$BASE_OUTPUT" "$MPLCONFIGDIR"

RUN_DIRS=(
  classifer_training/artifacts/runs/dapo_math_17k/qwen3_4b_instruct_2507/weak4_runs/0
  classifer_training/artifacts/runs/dapo_math_17k/qwen3_4b_instruct_2507/weak4_runs/1
)
DATASET_DIR="classifer_training/artifacts/datasets/dapo_math_17k_weak4"
LABELS_PATH="classifer_training/artifacts/labels/dapo_math_17k/qwen3_4b_instruct_2507/weak4_labels.jsonl"
ROLLOUT_MODEL_DIR="_data2_sangjunsong__cache_hf_hub_models--Qwen--Qwen3-4B-Instruct-2507_snapshots_cdbee75f17c01a7cc42f958dc650907174af0554"
ROLLOUT_HIDDEN=(classifer_training/artifacts/rollout_hidden/dapo_math_17k_weak4_think_end_l26/"$ROLLOUT_MODEL_DIR"/rollout_hidden_states.shard*.pt)
ROLLOUT_INDEX=(classifer_training/artifacts/rollout_index/dapo_math_17k_weak4_think_end_l26/"$ROLLOUT_MODEL_DIR"/rollout_index.shard*.jsonl)

SCALARS=(
  output_mean_token_entropy
  reasoning_mean_token_entropy
  output_last_token_entropy
  output_max_token_entropy
  output_min_token_entropy
  reasoning_last_token_entropy
  reasoning_max_token_entropy
  reasoning_min_token_entropy
  answer_last_token_entropy
  answer_max_token_entropy
  answer_mean_token_entropy
  answer_min_token_entropy
  entropy_gap_reasoning_answer
  answer_entropy_gap_vs_output
  rollout_features.answer_mean_token_entropy
)

run_probe() {
  local name="$1"
  local prompt_slug="$2"
  local prompt_layer="$3"
  local rollout_component="$4"
  local prompt_pca="$5"
  local rollout_pca="$6"

  local prompt_hidden=(classifer_training/artifacts/hidden/dapo_math_17k_weak4_shard*/"$prompt_slug"/hidden_states.pt)
  local prompt_index=(classifer_training/artifacts/index/dapo_math_17k_weak4_shard*/"$prompt_slug"/index.jsonl)
  local out="$BASE_OUTPUT/$name"

  echo "$(date '+%F %T') START $name prompt_slug=$prompt_slug prompt_layer=$prompt_layer rollout_component=$rollout_component prompt_pca=$prompt_pca rollout_pca=$rollout_pca"
  "$PYTHON_BIN" classifer_training/train_weak_only_single_rollout_hidden.py \
    --weak_run_dirs "${RUN_DIRS[@]}" \
    --weak_prompt_dataset_dir "$DATASET_DIR" \
    --weak_labels_path "$LABELS_PATH" \
    --weak_prompt_hidden_paths "${prompt_hidden[@]}" \
    --weak_prompt_index_paths "${prompt_index[@]}" \
    --weak_rollout_hidden_paths "${ROLLOUT_HIDDEN[@]}" \
    --weak_rollout_index_paths "${ROLLOUT_INDEX[@]}" \
    --output_dir "$out" \
    --prompt_hidden_component hidden \
    --prompt_layer_index "$prompt_layer" \
    --rollout_component "$rollout_component" \
    --rollout_layer_index 26 \
    --rollout_pool_mode mean \
    --feature_mode prompt_plus_rollout \
    --prompt_hidden_pca_dim "$prompt_pca" \
    --rollout_hidden_pca_dim "$rollout_pca" \
    --single_rollout_strategy all \
    --model_family ridge \
    --train_target_mode other_rollout_correctness \
    --selection_metric row_r2 \
    --alphas 0.01 \
    --rollout_scalar_keys "${SCALARS[@]}" \
    --allow_missing_entropy_scalars
  echo "$(date '+%F %T') DONE $name"
}

# Center setting matches the SPO row-R2-optimized probe as closely as current
# DAPO weak4 artifacts allow: prompt last6mean L26 + think_end_hidden L26.
run_probe "center_prompt_last6_L26_thinkend_L26_p32_r256" "qwen3_4b_instruct_2507_last6mean" 26 "think_end_hidden" 32 256

# Move one axis at a time. The rollout think-end hidden is always present.
for layer in 18 20 22 24 26 28 30 32 34 35; do
  run_probe "prompt_layer_sweep_last6_L${layer}_thinkend_L26_p32_r256" "qwen3_4b_instruct_2507_last6mean" "$layer" "think_end_hidden" 32 256
done

run_probe "prompt_pool_last_L26_thinkend_L26_p32_r256" "qwen3_4b_instruct_2507" 26 "think_end_hidden" 32 256
run_probe "rollout_component_thinkend_last10_prompt_last6_L26_p32_r256" "qwen3_4b_instruct_2507_last6mean" 26 "think_end_last10_hidden" 32 256

for prompt_pca in 16 32 64 128; do
  run_probe "prompt_pca_sweep_p${prompt_pca}_r256_last6_L26_thinkend_L26" "qwen3_4b_instruct_2507_last6mean" 26 "think_end_hidden" "$prompt_pca" 256
done

for rollout_pca in 64 128 256 512; do
  run_probe "rollout_pca_sweep_p32_r${rollout_pca}_last6_L26_thinkend_L26" "qwen3_4b_instruct_2507_last6mean" 26 "think_end_hidden" 32 "$rollout_pca"
done

"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

base = Path("classifer_training/artifacts/probe/dapo_math_17k_weak4_simple_ridge_entropy_rowr2_axis_sweep")
rows = []
for summary_path in sorted(base.glob("*/summary.json")):
    data = json.loads(summary_path.read_text())
    rows.append({
        "name": summary_path.parent.name,
        "row_r2": data["weak_val_row_metrics"]["r2"],
        "row_mae": data["weak_val_row_metrics"]["mae"],
        "row_rmse": data["weak_val_row_metrics"]["rmse"],
        "prompt_mean_r2": data["weak_val_prompt_mean_metrics"]["r2"],
        "prompt_mean_mae": data["weak_val_prompt_mean_metrics"]["mae"],
        "feature_dim": data["feature_dim"],
        "num_train_rows": data["num_train_rows"],
        "num_weak_val_rows": data["num_weak_val_rows"],
    })
rows.sort(key=lambda row: row["row_r2"], reverse=True)
(base / "axis_sweep_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
with (base / "axis_sweep_summary.md").open("w") as f:
    f.write("| rank | name | row_r2 | prompt_mean_r2 | row_mae | prompt_mean_mae | dim |\n")
    f.write("|---:|---|---:|---:|---:|---:|---:|\n")
    for idx, row in enumerate(rows, 1):
        f.write(
            f"| {idx} | {row['name']} | {row['row_r2']:.4f} | "
            f"{row['prompt_mean_r2']:.4f} | {row['row_mae']:.4f} | "
            f"{row['prompt_mean_mae']:.4f} | {row['feature_dim']} |\n"
        )
print(json.dumps(rows[:5], indent=2))
PY
