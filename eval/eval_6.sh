export CUDA_VISIBLE_DEVICES="2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TEMPLATE="${TEMPLATE:-qwen}"
ENABLE_THINKING="${ENABLE_THINKING:-}"

TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-1.0}"
MAX_TOKENS="${MAX_TOKENS:-2048}"

CRITIQUE_TEMPERATURE="${CRITIQUE_TEMPERATURE:-0.6}"
CRITIQUE_TOP_P="${CRITIQUE_TOP_P:-1.0}"
CRITIQUE_MAX_TOKENS="${CRITIQUE_MAX_TOKENS:-2048}"

REVISE_TEMPERATURE="${REVISE_TEMPERATURE:-0.6}"
REVISE_TOP_P="${REVISE_TOP_P:-1.0}"
REVISE_MAX_TOKENS="${REVISE_MAX_TOKENS:-2048}"

# CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/checkpoints/checkpoints/verl_grpo_critique/qwen2.5_7b_instruct_pure_critique_llama3b_math_variants/global_step_400}"
# ACTOR_DIR="${ACTOR_DIR:-${CKPT_DIR}/actor}"
# MERGED_DIR="${MERGED_DIR:-${ACTOR_DIR}/hf_merged}"

# Default to merged HF model path for vLLM
#MODEL_PATH="${MODEL_PATH:-${MERGED_DIR}}"
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
# HF_MODEL_REPO="${HF_MODEL_REPO:-yunhowhour/qwen2_5_7b_instruct_critique_step400}"
# HF_MODEL_REVISION="${HF_MODEL_REVISION:-9b2395cbd7719620dc275f3005edeca40fb705ac}"
# HF_MODEL_CACHE_DIR="${HF_MODEL_CACHE_DIR:-${REPO_ROOT}/.cache/huggingface}"
# HF_MODEL_TOKEN="${HF_MODEL_TOKEN:-${HF_TOKEN:-}}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results}"

DATA="${DATA:-${REPO_ROOT}/data/eval/test.id.parquet}"
MODEL_NAME="${MODEL_NAME:-qwen2.5_7b_instruct_self_improvement}"

mkdir -p "$OUTPUT_DIR"

ENABLE_THINKING_ARGS=()
if [ -n "${ENABLE_THINKING}" ]; then
  ENABLE_THINKING_ARGS=(--enable_thinking "${ENABLE_THINKING}")
fi

CRITIQUE_ARGS=()
if [ -n "${CRITIQUE_TEMPERATURE}" ]; then
  CRITIQUE_ARGS+=(--critique_temperature "${CRITIQUE_TEMPERATURE}")
fi
if [ -n "${CRITIQUE_TOP_P}" ]; then
  CRITIQUE_ARGS+=(--critique_top_p "${CRITIQUE_TOP_P}")
fi
if [ -n "${CRITIQUE_MAX_TOKENS}" ]; then
  CRITIQUE_ARGS+=(--critique_max_tokens "${CRITIQUE_MAX_TOKENS}")
fi

REVISE_ARGS=()
if [ -n "${REVISE_TEMPERATURE}" ]; then
  REVISE_ARGS+=(--revise_temperature "${REVISE_TEMPERATURE}")
fi
if [ -n "${REVISE_TOP_P}" ]; then
  REVISE_ARGS+=(--revise_top_p "${REVISE_TOP_P}")
fi
if [ -n "${REVISE_MAX_TOKENS}" ]; then
  REVISE_ARGS+=(--revise_max_tokens "${REVISE_MAX_TOKENS}")
fi

# If HF_MODEL_REPO is set, download snapshot locally and use it as MODEL_PATH.
if [ -n "${HF_MODEL_REPO}" ]; then
  DOWNLOAD_ARGS=(
    "${SCRIPT_DIR}/hf_checkpoint_hub.py"
    download
    --repo-id "${HF_MODEL_REPO}"
    --cache-dir "${HF_MODEL_CACHE_DIR}"
    --print-path-only
  )
  if [ -n "${HF_MODEL_REVISION}" ]; then
    DOWNLOAD_ARGS+=(--revision "${HF_MODEL_REVISION}")
  fi
  if [ -n "${HF_MODEL_TOKEN}" ]; then
    DOWNLOAD_ARGS+=(--token "${HF_MODEL_TOKEN}")
  fi

  echo "Downloading model from Hugging Face repo: ${HF_MODEL_REPO}"
  MODEL_PATH="$(python "${DOWNLOAD_ARGS[@]}")"
  echo "Resolved MODEL_PATH=${MODEL_PATH}"
fi

# If MODEL_PATH points to an FSDP checkpoint (or default), merge to HF for vLLM
if [ "$MODEL_PATH" = "$MERGED_DIR" ] || [ -f "${MODEL_PATH}/fsdp_config.json" ] || ls "${MODEL_PATH}"/model_world_size_* >/dev/null 2>&1; then
  has_merged_weights=false
  for pattern in \
    "${MERGED_DIR}"/model*.safetensors \
    "${MERGED_DIR}"/pytorch_model*.bin \
    "${MERGED_DIR}"/model*.index.json \
    "${MERGED_DIR}"/pytorch_model*.index.json; do
    if compgen -G "$pattern" > /dev/null; then
      has_merged_weights=true
      break
    fi
  done
  if [ "$has_merged_weights" != "true" ]; then
    echo "Merged HF weights not found. Running FSDP merge..."
    python -m verl.model_merger merge \
      --backend fsdp \
      --local_dir "${ACTOR_DIR}" \
      --target_dir "${MERGED_DIR}"
  fi
  MODEL_PATH="${MERGED_DIR}"
fi

python "${SCRIPT_DIR}/generate_vllm_self_improvement.py" \
  --model_path "$MODEL_PATH" \
  --input_file "$DATA" \
  --remove_system True \
  --add_oat_evaluate True \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --max_tokens "$MAX_TOKENS" \
  --output_file "$OUTPUT_DIR/$MODEL_NAME.jsonl" \
  --template "$TEMPLATE" \
  "${ENABLE_THINKING_ARGS[@]}" \
  "${CRITIQUE_ARGS[@]}" \
  "${REVISE_ARGS[@]}" > "$OUTPUT_DIR/$MODEL_NAME.log"
