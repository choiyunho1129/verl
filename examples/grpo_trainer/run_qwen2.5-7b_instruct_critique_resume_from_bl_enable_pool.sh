#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

train_file="${REPO_ROOT}/data/train_critique_3.1_4.parquet"
test_file="${REPO_ROOT}/data/MATH-500/test.parquet"
reward_fn_path="${REPO_ROOT}/verl/trainer/ppo/custom_rewards/critique_reward_new.py"
resume_ckpt="${REPO_ROOT}/checkpoints/verl_grpo_critique/qwen2.5_7b_instruct_train_MATH500_baseline/global_step_80"

export DEBUG_REWARD="${DEBUG_REWARD:-False}"
export REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
export ENABLE_RM_POOL="${ENABLE_RM_POOL:-True}"           # standalone RM pool if true
export RM_NGPUS_PER_NODE="${RM_NGPUS_PER_NODE:-1}"
export RM_NNODES="${RM_NNODES:-1}"
export RM_TP_SIZE="${RM_TP_SIZE:-1}"
export RM_GPU_UTIL="${RM_GPU_UTIL:-0.80}"
export RM_PROMPT_LEN="${RM_PROMPT_LEN:-6144}"
export RM_RESPONSE_LEN="${RM_RESPONSE_LEN:-2048}"

# Drop stale dataloader state from the MATH-500 run so the loader restarts on the new dataset
RESET_DATALOADER_STATE=${RESET_DATALOADER_STATE:-1}
if [[ "${RESET_DATALOADER_STATE}" == "1" ]]; then
    rm -f "${resume_ckpt}/data.pt"
fi
# Reset global_steps to 0 when resuming from baseline so total_steps is recomputed for the new dataset
export RESET_GLOBAL_STEPS_ON_RESUME="${RESET_GLOBAL_STEPS_ON_RESUME:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
echo "Resuming PPO Training from baseline checkpoint at ${resume_ckpt}..."

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_file" \
    data.val_files="$test_file" \
    data.train_batch_size=64 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path="$reward_fn_path" \
    custom_reward_function.name=compute_score \
    reward_model.enable=True \
    reward_model.use_reward_loop=True \
    reward_model.reward_manager=naive \
    reward_model.model.path="$REWARD_MODEL_PATH" \
    reward_model.rollout.name=vllm \
    reward_model.rollout.tensor_model_parallel_size=${RM_TP_SIZE} \
    reward_model.rollout.gpu_memory_utilization=${RM_GPU_UTIL} \
    reward_model.rollout.prompt_length=${RM_PROMPT_LEN} \
    reward_model.rollout.response_length=${RM_RESPONSE_LEN} \
    reward_model.enable_resource_pool=${ENABLE_RM_POOL} \
    reward_model.n_gpus_per_node=${RM_NGPUS_PER_NODE} \
    reward_model.nnodes=${RM_NNODES} \
    reward_model.num_workers=1 \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_critique' \
    trainer.experiment_name='qwen2.5_7b_instruct_critique_resume_from_bl' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=24 \
    trainer.test_freq=3 \
    trainer.val_only=False \
    trainer.val_before_train=False \
    trainer.total_epochs=2 \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="$resume_ckpt" \
    "$@"
