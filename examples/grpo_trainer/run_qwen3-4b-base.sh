set -e  

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export CUDA_VISIBLE_DEVICES=0,1
export HF_HUB_CACHE="/data01/yunhochoi/.cache/huggingface"

# Qwen3-4B-Base should consume the raw MATH 4-shot prompt, so bypass any chat template.
NO_CHAT_TEMPLATE='{% for message in messages %}{{ message["content"] }}\n{% endfor %}{% if add_generation_prompt %}{% endif %}'
NO_CHAT_TEMPLATE_OVERRIDE="'${NO_CHAT_TEMPLATE}'"


train_file="${REPO_ROOT}/data/MATH-500/train_MATH3-5_4shot.parquet"
test_file="${REPO_ROOT}/data/MATH-500/test_4shot.parquet"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_file" \
    data.val_files="$test_file" \
    data.train_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B-Base \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.stop_tokens='["Problem:","\nProblem:","\n\nProblem:","Question:"]' \
    actor_rollout_ref.model.custom_chat_template="$NO_CHAT_TEMPLATE_OVERRIDE" \
    +data.apply_chat_template_kwargs.chat_template="$NO_CHAT_TEMPLATE_OVERRIDE" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_baseline' \
    trainer.experiment_name='qwen3_4b_base_MATH3-5' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=15 \
    trainer.test_freq=5 \
    trainer.val_before_train=True \
    trainer.total_training_steps=1500 \
    trainer.total_epochs=30 $@
