#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Move Ray/tmp spill to /data1 to avoid /tmp filling up.
export RAY_TMPDIR="${RAY_TMPDIR:-/data1/ray_tmp}"
export TMPDIR="${TMPDIR:-/data1/tmp}"
mkdir -p "${RAY_TMPDIR}" "${TMPDIR}"

# Per-run log directory and run metadata.
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs}"
LOG_DIR="${LOG_ROOT}/grpo_critique_${RUN_ID}"
mkdir -p "${LOG_DIR}"
{
  echo "run_id=${RUN_ID}"
  echo "script=$0"
  echo "args=$*"
  echo "cwd=$(pwd)"
} > "${LOG_DIR}/run.meta"
env | sort > "${LOG_DIR}/env.txt"
exec > >(tee -a "${LOG_DIR}/train.log") 2>&1

if [ "${ENABLE_SYS_MONITOR:-1}" = "1" ]; then
  (
    while true; do
      date
      free -h
      ps -eo pid,ppid,cmd,rss,%mem --sort=-rss | head -n 30
      nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv || true
      sleep "${MONITOR_INTERVAL:-300}"
    done
  ) > "${LOG_DIR}/sys_monitor.log" &
  MONITOR_PID=$!
fi

cleanup() {
  local code=$?

  if [ -n "${MONITOR_PID:-}" ]; then
    kill "${MONITOR_PID}" >/dev/null 2>&1 || true
  fi

  free -h > "${LOG_DIR}/free.txt" || true
  ps -eo pid,ppid,cmd,rss,%mem --sort=-rss | head -n 50 > "${LOG_DIR}/ps_top_mem.txt" || true
  nvidia-smi > "${LOG_DIR}/nvidia-smi.txt" 2>&1 || true
  nvidia-smi -q -d MEMORY > "${LOG_DIR}/nvidia-smi.mem.txt" 2>&1 || true
  dmesg -T | tail -n 200 > "${LOG_DIR}/dmesg_tail.txt" 2>&1 || true

  if command -v ray >/dev/null 2>&1; then
    ray status > "${LOG_DIR}/ray_status.txt" 2>&1 || true
    ray memory --stats-only > "${LOG_DIR}/ray_memory.txt" 2>&1 || true
  fi

  if [ -d "${RAY_TMPDIR}/ray/session_latest/logs" ]; then
    mkdir -p "${LOG_DIR}/ray_logs"
    cp -a "${RAY_TMPDIR}/ray/session_latest/logs/." "${LOG_DIR}/ray_logs/" || true
  fi

  # Optional: stop Ray to clean up sessions after run finishes.
  if [ "${AUTO_RAY_STOP:-1}" = "1" ] && command -v ray >/dev/null 2>&1; then
    ray stop -f >/dev/null 2>&1 || true
  fi

  exit "${code}"
}
trap cleanup EXIT

train_file="${REPO_ROOT}/data/math_variant_criitique_prompt/prompt_llama_3.2_4.parquet"
test_file="${REPO_ROOT}/data/snapshots_variants_test/prompt_llama_3b_instruct_trajectories_1.parquet"
reward_fn_path="${REPO_ROOT}/verl/trainer/ppo/custom_rewards/critique_reward.py"

PROJECT_NAME="${PROJECT_NAME:-verl_grpo_critique}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen2.5_7b_instruct_critique_llama3b_val_only}"
VAL_DATA_DIR="${REPO_ROOT}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}/validation"

# Reward Loop / GenRM configuration
export DEBUG_REWARD="${DEBUG_REWARD:-False}"
export REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-meta-llama/Llama-3.2-3B-Instruct}"
export ENABLE_RM_POOL="${ENABLE_RM_POOL:-False}"           # standalone RM pool if true
export RM_NGPUS_PER_NODE="${RM_NGPUS_PER_NODE:-2}"
export RM_NNODES="${RM_NNODES:-1}"
export RM_TP_SIZE="${RM_TP_SIZE:-1}"
export RM_GPU_UTIL="${RM_GPU_UTIL:-0.70}"
export RM_PROMPT_LEN="${RM_PROMPT_LEN:-6144}"
export RM_RESPONSE_LEN="${RM_RESPONSE_LEN:-2048}"

echo "Starting PPO Training with Reward Loop..."

# override when using a separate RM pool
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_file" \
    data.val_files="$test_file" \
    data.train_batch_size=128 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path="$reward_fn_path" \
    custom_reward_function.name=compute_score \
    reward_model.enable=True \
    reward_model.use_reward_loop=True \
    reward_model.reward_manager=naive \
    reward_model.model.path="$REWARD_MODEL_PATH" \
    +reward_model.rollout.engine_kwargs.vllm.served_model_name=${REWARD_MODEL_PATH} \
    reward_model.rollout.name=vllm \
    reward_model.rollout.tensor_model_parallel_size=${RM_TP_SIZE} \
    reward_model.rollout.gpu_memory_utilization=${RM_GPU_UTIL} \
    reward_model.rollout.prompt_length=${RM_PROMPT_LEN} \
    reward_model.rollout.response_length=${RM_RESPONSE_LEN} \
    reward_model.rollout.enable_prefix_caching=False \
    reward_model.enable_resource_pool=${ENABLE_RM_POOL} \
    reward_model.n_gpus_per_node=${RM_NGPUS_PER_NODE} \
    reward_model.nnodes=${RM_NNODES} \
    reward_model.num_workers=2 \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.validation_data_dir="${VAL_DATA_DIR}" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=15 \
    trainer.test_freq=5 \
    trainer.val_only=True \
    trainer.val_before_train=True  \
    trainer.total_epochs=4 \
    "$@"
