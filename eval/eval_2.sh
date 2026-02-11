SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TEMPLATE="${TEMPLATE:-own}"

CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/checkpoints/verl_grpo_critique/qwen2.5_7b_instruct_MATH3-5_dapo/global_step_180}"
ACTOR_DIR="${ACTOR_DIR:-${CKPT_DIR}/actor}"
MERGED_DIR="${MERGED_DIR:-${ACTOR_DIR}/hf_merged}"

# Default to merged HF model path for vLLM
MODEL_PATH="${MODEL_PATH:-${MERGED_DIR}}"
#MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results}"

DATA="${DATA:-${REPO_ROOT}/data/eval/test.id.parquet}"
MODEL_NAME="${MODEL_NAME:-qwen2.5_7b_instruct_MATH3-5_dapo_step_180_instructprompt}"

mkdir -p "$OUTPUT_DIR"

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

python eval/generate_vllm.py \
  --model_path $MODEL_PATH \
  --input_file $DATA \
  --remove_system True \
  --add_oat_evaluate True \
  --output_file $OUTPUT_DIR/$MODEL_NAME.jsonl \
  --template $TEMPLATE > $OUTPUT_DIR/$MODEL_NAME.log

# DATA=$ROOT/data/valid.ood.parquet
# MODEL_NAME=exgrpo+testood

# mkdir -p $OUTPUT_DIR

# python generate_vllm.py \
#    --model_path $MODEL_PATH \
#    --input_file $DATA \
#    --remove_system True \
#    --add_oat_evaluate True \
#    --output_file $OUTPUT_DIR/$MODEL_NAME.jsonl \
#    --template $TEMPLATE > $OUTPUT_DIR/$MODEL_NAME.log
