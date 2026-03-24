set -x

# 0. download the config
# only need to download the `configuration_deepseek.py`, `config.json`, `tokenizer_config.json`, `tokenizer.json` and `generation_config.json`
# remove the `quantization_config` in the `config.json`
# set `num_nextn_predict_layers=0` to disable MTP, which is not currently supported

# huggingface-cli download deepseek-ai/DeepSeek-V3-0324 configuration_deepseek.py config.json

# 1. download the dist_ckpt format model from https://huggingface.co/BearBiscuit05/dpsk-v3-671B-BF16-dist_ckpt/tree/main
# change the HF_MODEL_PATH to your own path
HF_MODEL_PATH="Qwen/Qwen3-1.7B-Base"
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export NVTE_FLASH_ATTN=1
export NVTE_DEBUG=1 
export NVTE_DEBUG_LEVEL=2
# W&B logging defaults.
WANDB_PROJECT=${WANDB_PROJECT:-verl_examples}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-qwen-onpolicydistillation_maxlen_8192}
WANDB_LOGGER=${WANDB_LOGGER:-'["console","wandb"]'}
PROJECT_NAME=${PROJECT_NAME:-${WANDB_PROJECT}}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-${WANDB_RUN_NAME}}
# Respect existing wandb login state (e.g. `wandb login`); only override
# when user explicitly sets WANDB_MODE.
export WANDB_MODE=${WANDB_MODE:-online}

# 512 H20(96GB)
NODES=1
PP=1
TP=1
EP=1
ETP=1
INFER_TP=1
# Set GPU ids manually (example: "0" or "0,1,2").
CUDA_VISIBLE_DEVICES="0,1,2"
# consider TP/ETP, and enable recompute if short of memory

# full recompute
# +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
# +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
# +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \

WORKING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${WORKING_DIR}/../.." && pwd)"
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/config/runtime_env.yaml"}
REWARD_FN_PATH="${REPO_ROOT}/verl/trainer/ppo/custom_rewards/critique_reward.py"
VAL_DATA_DIR=${VAL_DATA_DIR:-"${REPO_ROOT}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}/validation"}
# 2. run the script
train_files=${TRAIN_FILES:-"${REPO_ROOT}/data/MATH-500/train.parquet"}
test_files=${TEST_FILES:-"${REPO_ROOT}/data/MATH-500/test.parquet"}
mkdir -p "${VAL_DATA_DIR}"

# Run locally from repo root so package imports (recipe.gkd.*) work; main_gkd will ray.init() if needed.
cd "${REPO_ROOT}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}" \
python3 -m recipe.gkd.main_gkd --config-name on_policy_distill_trainer \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.prompt_key=prompt \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.use_chat_template=False \
    data.trust_remote_code=True \
    data.return_raw_chat=True \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.teacher.server_ip=127.0.0.1 \
    actor_rollout_ref.teacher.server_port=15555 \
    actor_rollout_ref.teacher.use_sampled_token_logprobs=True \
    actor_rollout_ref.model.path=$HF_MODEL_PATH \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.megatron.sequence_parallel=False \
    +actor_rollout_ref.actor.megatron.override_transformer_config.sequence_parallel=False \
    actor_rollout_ref.actor.router_replay.mode=disabled \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.actor.policy_loss.loss_mode=bypass_mode \
    +actor_rollout_ref.actor.policy_loss.rollout_correction.loss_type=reinforce \
    +actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is=token \
    +actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is_threshold=2.0 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$INFER_TP \
    actor_rollout_ref.rollout.load_format='auto' \
    custom_reward_function.path="${REWARD_FN_PATH}" \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    trainer.logger="${WANDB_LOGGER}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    +trainer.validation_data_dir="${VAL_DATA_DIR}" \
    trainer.n_gpus_per_node=3 \
    trainer.nnodes=$NODES \
    trainer.save_freq=40 \
    trainer.test_freq=10 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=$EP \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=$ETP \
    trainer.val_before_train=False \
    trainer.total_training_steps=500 \
    trainer.total_epochs=30 "$@"
    # +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_first_pipeline_stage=11 \
