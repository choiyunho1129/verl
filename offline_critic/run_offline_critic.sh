#!/usr/bin/env bash
set -euo pipefail

# Replays CRRL prompt_reward_logs into an SPPO-style prompt-value critic,
# saving one checkpoint per global_step so we can compare critic predictions
# against CRRL's adaptive estimator at every step.

CRRL_RUN_DIR="/NHNHOME/WORKSPACE/26msit006_A/kisti/snu/yunhochoi/crrl/crrl_verl_pr/Qwen3-4B_CRRL_batch_1024_B200_dynamicsampling"
PROMPT_LOGS="${CRRL_RUN_DIR}/checkpoints/prompt_reward_logs"
VALIDATION="${CRRL_RUN_DIR}/validation_data"

OUTPUT_DIR="${OUTPUT_DIR:-/NHNHOME/WORKSPACE/26msit006_A/kisti/snu/jongwon/offline_critic/runs/qwen3-4b-replay}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B}"

# Saving the full 4B base every step costs ~8GB/ckpt. With 91 steps that's
# ~700GB. By default save the base every 10 steps (so warm-startable HF dirs
# exist at strategic milestones); v_head + optim still saved every step.
SAVE_EVERY="${SAVE_EVERY:-10}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python3 /NHNHOME/WORKSPACE/26msit006_A/kisti/snu/jongwon/offline_critic/train_offline_critic.py \
    --prompt_reward_logs "${PROMPT_LOGS}" \
    --validation_data "${VALIDATION}" \
    --base_model "${BASE_MODEL}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_length 2048 \
    --batch_size 8 \
    --eval_batch_size 16 \
    --lr 1e-5 \
    --v_head_lr_mult 10 \
    --weight_decay 0.01 \
    --grad_clip 1.0 \
    --epochs_per_step 1 \
    --save_every "${SAVE_EVERY}" \
    --num_workers 2 \
    "$@"
