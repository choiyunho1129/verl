#!/usr/bin/env bash
# Pipeline: SPO temp1 subset0-4 → import → build dataset → extract hidden states → train probe
# Usage: bash run_temp1_pipeline.sh [--gpu 3] [--skip-import] [--skip-build] [--skip-extract-prompt] [--skip-extract-rollout] [--skip-train]
set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────────────
REPO=/NHNHOME/WORKSPACE/26msit006_A/kisti/snu/jongwon/verl
PYTHON=/NHNHOME/WORKSPACE/26msit006_A/kisti/snu/yunhochoi/envs/spo_jongwon/bin/python3
GPU=3
DATASET_NAME=spo_temp1_subset0to4
MODEL=Qwen/Qwen3-4B-Instruct-2507
MODEL_SLUG=qwen3_4b_instruct_2507_last6mean
TRAIN_RUN_DIR_NAMES=(offline_value_estimation_subset_0 offline_value_estimation_subset_1)
VALIDATION_RUN_DIR_NAMES=(offline_value_estimation_subset_2 offline_value_estimation_subset_3 offline_value_estimation_subset_4)
TRAIN_RUN_DIR_NAMES_CSV=offline_value_estimation_subset_0,offline_value_estimation_subset_1
VALIDATION_RUN_DIR_NAMES_CSV=offline_value_estimation_subset_2,offline_value_estimation_subset_3,offline_value_estimation_subset_4

SPO_GLOB="/NHNHOME/WORKSPACE/26msit006_A/kisti/snu/yunhochoi/crrl/spo_verl_pr_temp1/spo/offline_value_estimation_subset_*/validation_data/0.jsonl"

IMPORTED_ROOT=$REPO/classifer_training/artifacts/runs/$DATASET_NAME/imported_runs
DATASET_DIR=$REPO/classifer_training/artifacts/datasets/$DATASET_NAME
LABELS_PATH=$REPO/classifer_training/artifacts/labels/$DATASET_NAME/labels.jsonl
LABELS_SUMMARY=$REPO/classifer_training/artifacts/labels/$DATASET_NAME/labels_summary.json
SHARDS_DIR=$REPO/classifer_training/artifacts/datasets/${DATASET_NAME}_shards

HIDDEN_DIR=$REPO/classifer_training/artifacts/hidden/${DATASET_NAME}_shard0/$MODEL_SLUG
INDEX_DIR=$REPO/classifer_training/artifacts/index/${DATASET_NAME}_shard0/$MODEL_SLUG
ROLLOUT_MODEL_SLUG=Qwen_Qwen3-4B-Instruct-2507
ROLLOUT_HIDDEN_DIR=$REPO/classifer_training/artifacts/rollout_hidden/$DATASET_NAME/$ROLLOUT_MODEL_SLUG
ROLLOUT_INDEX_DIR=$REPO/classifer_training/artifacts/rollout_index/$DATASET_NAME/$ROLLOUT_MODEL_SLUG

VALIDATION_ROLLOUTS_PER_PROMPT=2
VALIDATION_ROLLOUT_SEED=42
TRAIN_TARGET_MODE=other_rollout_correctness
MODEL_FAMILY=ridge
SELECTION_METRIC=row_r2
RIDGE_ALPHAS=(10000 30000 100000 300000 1000000 3000000)
OUTPUT_DIR=$REPO/classifer_training/artifacts/probe/${DATASET_NAME}_val${VALIDATION_ROLLOUTS_PER_PROMPT}_seed${VALIDATION_ROLLOUT_SEED}_${TRAIN_TARGET_MODE}_${MODEL_FAMILY}_high_alpha_${SELECTION_METRIC}

LOG_DIR=$REPO/classifer_training/artifacts/logs/$DATASET_NAME
STATE_FILE=$LOG_DIR/pipeline_state.json

# ── Flags ────────────────────────────────────────────────────────────────────
SKIP_IMPORT=0; SKIP_BUILD=0; SKIP_EXTRACT_PROMPT=0; SKIP_EXTRACT_ROLLOUT=0; SKIP_TRAIN=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpu=*) GPU="${1#--gpu=}" ;;
    --gpu)   shift; GPU="$1" ;;
    --skip-import)          SKIP_IMPORT=1 ;;
    --skip-build)           SKIP_BUILD=1 ;;
    --skip-extract-prompt)  SKIP_EXTRACT_PROMPT=1 ;;
    --skip-extract-rollout) SKIP_EXTRACT_ROLLOUT=1 ;;
    --skip-train)           SKIP_TRAIN=1 ;;
  esac
  shift
done

# ── Helpers ───────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

save_state() {
  mkdir -p "$LOG_DIR"
  echo "{\"last_completed_step\": \"$1\", \"timestamp\": \"$(date -Iseconds)\"}" > "$STATE_FILE"
}

show_state() {
  if [[ -f "$STATE_FILE" ]]; then
    log "Resuming. Last completed step: $(python3 -c "import json; print(json.load(open('$STATE_FILE'))['last_completed_step'])")"
  else
    log "Starting fresh pipeline."
  fi
}

dataset_artifacts_valid() {
  "$PYTHON" - "$DATASET_DIR" "$LABELS_PATH" "$LABELS_SUMMARY" "$SHARDS_DIR" "$TRAIN_RUN_DIR_NAMES_CSV" "$VALIDATION_RUN_DIR_NAMES_CSV" <<'PY'
import json
import sys
from pathlib import Path

dataset_dir, labels_path, summary_path, shards_dir = map(Path, sys.argv[1:5])
expected_train_run_dir_names = sorted(name for name in sys.argv[5].split(",") if name)
expected_validation_run_dir_names = sorted(name for name in sys.argv[6].split(",") if name)
train_path = dataset_dir / "train.jsonl"
validation_path = dataset_dir / "validation.jsonl"
shard_path = shards_dir / "shard0.jsonl"

for path in (train_path, validation_path, labels_path, summary_path, shard_path):
    if not path.exists():
        raise SystemExit(f"missing {path}")

def count_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())

train_count = count_rows(train_path)
validation_count = count_rows(validation_path)
label_count = count_rows(labels_path)
shard_count = count_rows(shard_path)

if train_count <= 0:
    raise SystemExit("train split is empty")
if validation_count <= 0:
    raise SystemExit("validation split is empty")
if label_count != train_count + validation_count:
    raise SystemExit(
        f"labels count {label_count} does not match train+validation {train_count + validation_count}"
    )
if shard_count != label_count:
    raise SystemExit(f"shard0 count {shard_count} does not match labels count {label_count}")

summary = json.loads(summary_path.read_text(encoding="utf-8"))
if sorted(summary.get("train_run_dir_names", [])) != expected_train_run_dir_names:
    raise SystemExit("dataset train run-dir split policy is stale")
if sorted(summary.get("validation_run_dir_names", [])) != expected_validation_run_dir_names:
    raise SystemExit("dataset validation run-dir split policy is stale")
if int(summary.get("num_prompts_train", -1)) != train_count:
    raise SystemExit("summary train count does not match train.jsonl")
if int(summary.get("num_prompts_validation", -1)) != validation_count:
    raise SystemExit("summary validation count does not match validation.jsonl")

print(f"valid dataset artifacts: train={train_count} validation={validation_count}")
PY
}

# Run a timed step with ETA tracking via log tail
run_with_eta() {
  local label="$1"; shift
  local logfile="$LOG_DIR/${label}.log"
  mkdir -p "$LOG_DIR"
  log "Starting: $label → $logfile"
  local start=$SECONDS
  "$@" 2>&1 | tee "$logfile" | awk -v start="$SECONDS" '
    /^Processed [0-9]+\/[0-9]+/ {
      split($2, parts, "/")
      done = parts[1]+0; total = parts[2]+0
      elapsed = systime() - start
      if (done > 0 && elapsed > 0) {
        rate = done / elapsed
        remaining = (total - done) / rate
        printf "[ETA] %d/%d done  elapsed=%ds  eta=~%ds remaining\n", done, total, elapsed, remaining
      }
    }
    !/^Processed / { print }
  '
  local rc=${PIPESTATUS[0]}
  if [[ $rc -ne 0 ]]; then
    log "FAILED: $label (exit $rc). See $logfile"
    exit $rc
  fi
  log "Done: $label in $(( SECONDS - start ))s"
}

mkdir -p "$LOG_DIR"
show_state

RUN_DIRS=(
  $IMPORTED_ROOT/offline_value_estimation_subset_0
  $IMPORTED_ROOT/offline_value_estimation_subset_1
  $IMPORTED_ROOT/offline_value_estimation_subset_2
  $IMPORTED_ROOT/offline_value_estimation_subset_3
  $IMPORTED_ROOT/offline_value_estimation_subset_4
)

# ── Step 1: Import ────────────────────────────────────────────────────────────
if [[ $SKIP_IMPORT -eq 0 ]]; then
  if [[ -f "$IMPORTED_ROOT/summary.json" ]]; then
    log "Step 1 (import): already done, skipping. Use --skip-import to suppress this check."
  else
    run_with_eta "01_import" env PYTHONPATH=$REPO $PYTHON -m classifer_training.import_spo_rollouts \
      --input_glob "$SPO_GLOB" \
      --output_root "$IMPORTED_ROOT" \
      --dataset_name "$DATASET_NAME"
    save_state "import"
  fi
else
  log "Step 1 (import): skipped."
fi

# ── Step 2: Build dataset + labels ───────────────────────────────────────────
if [[ $SKIP_BUILD -eq 0 ]]; then
  if DATASET_CHECK="$(dataset_artifacts_valid 2>&1)"; then
    log "Step 2 (build dataset): already done, skipping. $DATASET_CHECK"
  else
    log "Step 2 (build dataset): rebuilding because $DATASET_CHECK"
    mkdir -p "$(dirname $LABELS_PATH)"
    run_with_eta "02_build_dataset" env PYTHONPATH=$REPO $PYTHON -m classifer_training.build_weak_prompt_dataset_and_labels \
      --run_dirs "${RUN_DIRS[@]}" \
      --prompt_dataset_dir "$DATASET_DIR" \
      --labels_path "$LABELS_PATH" \
      --summary_path "$LABELS_SUMMARY" \
      --train_run_dir_names "${TRAIN_RUN_DIR_NAMES[@]}" \
      --validation_run_dir_names "${VALIDATION_RUN_DIR_NAMES[@]}"
    save_state "build_dataset"

    log "Step 2b: preparing shards..."
    run_with_eta "02b_shards" env PYTHONPATH=$REPO $PYTHON -m classifer_training.prepare_weak4_shards \
      --input_dir "$DATASET_DIR" \
      --output_dir "$SHARDS_DIR" \
      --num_shards 1 \
      --overwrite
    save_state "prepare_shards"
  fi
else
  if DATASET_CHECK="$(dataset_artifacts_valid 2>&1)"; then
    log "Step 2 (build dataset): skipped. $DATASET_CHECK"
  else
    log "Step 2 (build dataset): skipped, but artifacts are invalid: $DATASET_CHECK"
    log "Re-run without --skip-build to rebuild them."
    exit 1
  fi
fi

# ── Step 3: Extract prompt hidden states ─────────────────────────────────────
if [[ $SKIP_EXTRACT_PROMPT -eq 0 ]]; then
  if [[ -f "$HIDDEN_DIR/hidden_states.pt" ]]; then
    log "Step 3 (prompt hidden): already done, skipping."
  else
    run_with_eta "03_prompt_hidden" env PYTHONPATH=$REPO $PYTHON \
      -m classifer_training.extract_hidden_states \
      --input_path "$SHARDS_DIR/shard0.jsonl" \
      --model_name_or_path "$MODEL" \
      --model_slug "$MODEL_SLUG" \
      --dataset_name "${DATASET_NAME}_shard0" \
      --token_pooling lastn_mean \
      --last_n 6 \
      --batch_size 64 \
      --cuda_device $GPU \
      --hidden_root "$REPO/classifer_training/artifacts/hidden" \
      --index_root "$REPO/classifer_training/artifacts/index"
    save_state "extract_prompt_hidden"
  fi
else
  log "Step 3 (prompt hidden): skipped."
fi

# ── Step 4: Extract rollout hidden states ────────────────────────────────────
if [[ $SKIP_EXTRACT_ROLLOUT -eq 0 ]]; then
  if [[ -f "$ROLLOUT_HIDDEN_DIR/rollout_hidden_states.pt" ]]; then
    log "Step 4 (rollout hidden): already done, skipping."
  else
    run_with_eta "04_rollout_hidden" env PYTHONPATH=$REPO $PYTHON \
      -m classifer_training.extract_rollout_hidden_states \
      --model_name_or_path "$MODEL" \
      --run_dirs "${RUN_DIRS[@]}" \
      --dataset_name "$DATASET_NAME" \
      --components response_hidden \
      --layers 26 \
      --cuda_device $GPU \
      --hidden_root "$REPO/classifer_training/artifacts/rollout_hidden" \
      --index_root "$REPO/classifer_training/artifacts/rollout_index" \
      --batch_size 64
    save_state "extract_rollout_hidden"
  fi
else
  log "Step 4 (rollout hidden): skipped."
fi

# ── Step 5: Train probe ───────────────────────────────────────────────────────
if [[ $SKIP_TRAIN -eq 0 ]]; then
  run_with_eta "05_train_probe" env PYTHONPATH=$REPO $PYTHON \
    -m classifer_training.train_weak_only_single_rollout_hidden \
    --weak_run_dirs "${RUN_DIRS[@]}" \
    --weak_prompt_dataset_dir "$DATASET_DIR" \
    --weak_labels_path "$LABELS_PATH" \
    --weak_prompt_hidden_paths "$HIDDEN_DIR/hidden_states.pt" \
    --weak_prompt_index_paths "$INDEX_DIR/index.jsonl" \
    --weak_rollout_hidden_paths "$ROLLOUT_HIDDEN_DIR/rollout_hidden_states.pt" \
    --weak_rollout_index_paths "$ROLLOUT_INDEX_DIR/rollout_index.jsonl" \
    --output_dir "$OUTPUT_DIR" \
    --feature_mode prompt_plus_rollout \
    --rollout_component response_hidden \
    --prompt_layer_index 26 \
    --single_rollout_strategy all \
    --validation_rollouts_per_prompt "$VALIDATION_ROLLOUTS_PER_PROMPT" \
    --validation_rollout_seed "$VALIDATION_ROLLOUT_SEED" \
    --train_target_mode "$TRAIN_TARGET_MODE" \
    --model_family "$MODEL_FAMILY" \
    --selection_metric "$SELECTION_METRIC" \
    --alphas "${RIDGE_ALPHAS[@]}"
  save_state "train_probe"
else
  log "Step 5 (train probe): skipped."
fi

log "Pipeline complete. Outputs:"
log "  Dataset:         $DATASET_DIR"
log "  Labels:          $LABELS_PATH"
log "  Prompt hidden:   $HIDDEN_DIR/hidden_states.pt"
log "  Rollout hidden:  $ROLLOUT_HIDDEN_DIR/rollout_hidden_states.pt"
log "  Probe:           $OUTPUT_DIR"
