#!/bin/bash
#SBATCH -J gkd_qwen
#SBATCH -p eme_h200nv_8
#SBATCH --comment python
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --mail-type=NONE

#SBATCH --ntasks=1

set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/test_qwen.sh"
TEACHER_DIR="${SCRIPT_DIR}/teacher"

CONDA_ENV="${CONDA_ENV:-d1}"
SCRATCH_BASE="${SCRATCH_BASE:-/scratch/${USER}}"
RUNSCR="${SCRATCH_BASE}/run/${SLURM_JOB_ID}"

PROXY_FRONTEND_PORT="${PROXY_FRONTEND_PORT:-15555}"
PROXY_BACKEND_PORT="${PROXY_BACKEND_PORT:-15556}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-Qwen/Qwen3-8B}"
TEACHER_BACKEND="${TEACHER_BACKEND:-vllm}"
TEACHER_TP_SIZE="${TEACHER_TP_SIZE:-1}"
TEACHER_N_LOGPROBS="${TEACHER_N_LOGPROBS:-256}"
TEACHER_STARTUP_TIMEOUT="${TEACHER_STARTUP_TIMEOUT:-1800}"
TRAIN_GPU_COUNT="${TRAIN_GPU_COUNT:-1}"

mkdir -p "${RUNSCR}"

wait_for_port() {
    local host="$1"
    local port="$2"
    local timeout_s="${3:-120}"
    python - "$host" "$port" "$timeout_s" <<'PY'
import socket
import sys
import time

host = sys.argv[1]
port = int(sys.argv[2])
timeout_s = int(sys.argv[3])
deadline = time.time() + timeout_s

while time.time() < deadline:
    try:
        with socket.create_connection((host, port), timeout=2):
            sys.exit(0)
    except OSError:
        time.sleep(1)

print(f"Timed out waiting for {host}:{port}", file=sys.stderr)
sys.exit(1)
PY
}

wait_for_log_line() {
    local logfile="$1"
    local pattern="$2"
    local timeout_s="${3:-300}"
    local start_ts
    start_ts="$(date +%s)"

    while true; do
        if [[ -f "${logfile}" ]] && grep -q "${pattern}" "${logfile}"; then
            return 0
        fi
        if (( "$(date +%s)" - start_ts >= timeout_s )); then
            echo "Timed out waiting for '${pattern}' in ${logfile}" >&2
            return 1
        fi
        sleep 2
    done
}

cleanup() {
    set +e
    if [[ -n "${TEACHER_WORKER_PID:-}" ]]; then
        kill "${TEACHER_WORKER_PID}" 2>/dev/null || true
    fi
    if [[ -n "${TEACHER_PROXY_PID:-}" ]]; then
        kill "${TEACHER_PROXY_PID}" 2>/dev/null || true
    fi
    wait "${TEACHER_WORKER_PID:-}" 2>/dev/null || true
    wait "${TEACHER_PROXY_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT

if [[ -f /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh ]]; then
    # shellcheck disable=SC1091
    source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
    conda activate "${CONDA_ENV}"
else
    echo "ERROR: conda.sh not found." >&2
    exit 1
fi

export HF_HOME="${RUNSCR}/hf"
export TORCH_HOME="${RUNSCR}/torch"
export WANDB_DIR="${RUNSCR}/wandb"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
export XDG_CACHE_HOME="${RUNSCR}/.cache"
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${WANDB_DIR}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${XDG_CACHE_HOME}"

echo "=========================================="
echo "SLURM_JOB_ID      = ${SLURM_JOB_ID}"
echo "SLURM_NODELIST    = ${SLURM_NODELIST:-N/A}"
echo "HOSTNAME          = $(hostname)"
echo "SLURM_SUBMIT_DIR  = ${SLURM_SUBMIT_DIR:-N/A}"
echo "REPO_ROOT         = ${REPO_ROOT}"
echo "RUNSCR            = ${RUNSCR}"
echo "CONDA_ENV         = ${CONDA_ENV}"
echo "=========================================="
nvidia-smi || true

python - <<'PY'
import torch
import transformers
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
PY

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a ALLOCATED_GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
else
    mapfile -t ALLOCATED_GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi

if (( ${#ALLOCATED_GPU_IDS[@]} < TRAIN_GPU_COUNT + 1 )); then
    echo "ERROR: expected at least $((TRAIN_GPU_COUNT + 1)) visible GPUs, got ${#ALLOCATED_GPU_IDS[@]}." >&2
    echo "Visible GPUs: ${CUDA_VISIBLE_DEVICES:-<unset>}" >&2
    exit 1
fi

TEACHER_GPU_ID="${TEACHER_GPU_ID:-${ALLOCATED_GPU_IDS[0]}}"
TRAIN_GPU_IDS_DEFAULT=("${ALLOCATED_GPU_IDS[@]:1:${TRAIN_GPU_COUNT}}")
if [[ -n "${TRAIN_GPU_IDS:-}" ]]; then
    TRAIN_GPU_IDS_STR="${TRAIN_GPU_IDS}"
else
    TRAIN_GPU_IDS_STR="$(IFS=,; echo "${TRAIN_GPU_IDS_DEFAULT[*]}")"
fi

echo "Teacher GPU        = ${TEACHER_GPU_ID}"
echo "Training GPUs      = ${TRAIN_GPU_IDS_STR}"

export PROXY_FRONTEND_PORT
export PROXY_BACKEND_PORT

TEACHER_PROXY_LOG="${RUNSCR}/teacher_proxy.log"
TEACHER_WORKER_LOG="${RUNSCR}/teacher_worker.log"

cd "${TEACHER_DIR}"
CUDA_VISIBLE_DEVICES="${TEACHER_GPU_ID}" PYTHONUNBUFFERED=1 python proxy.py >"${TEACHER_PROXY_LOG}" 2>&1 &
TEACHER_PROXY_PID=$!

wait_for_port 127.0.0.1 "${PROXY_BACKEND_PORT}" 120

CUDA_VISIBLE_DEVICES="${TEACHER_GPU_ID}" PYTHONUNBUFFERED=1 python worker.py \
    --backend "${TEACHER_BACKEND}" \
    --proxy-addr "127.0.0.1:${PROXY_BACKEND_PORT}" \
    --tp-size "${TEACHER_TP_SIZE}" \
    --n-logprobs "${TEACHER_N_LOGPROBS}" \
    --ckpt-path "${TEACHER_MODEL_PATH}" >"${TEACHER_WORKER_LOG}" 2>&1 &
TEACHER_WORKER_PID=$!

wait_for_log_line "${TEACHER_WORKER_LOG}" "worker started..." "${TEACHER_STARTUP_TIMEOUT}"

echo "Teacher server is ready."

PATCHED_TRAIN_SCRIPT="${RUNSCR}/test_qwen.slurm.sh"
sed \
    -e "s|^CUDA_VISIBLE_DEVICES=.*|CUDA_VISIBLE_DEVICES=\"${TRAIN_GPU_IDS_STR}\"|" \
    -e "s|^WORKING_DIR=.*|WORKING_DIR=\"${SCRIPT_DIR}\"|" \
    -e "s|^REPO_ROOT=.*|REPO_ROOT=\"${REPO_ROOT}\"|" \
    -e "s|trainer.n_gpus_per_node=2 \\\\|trainer.n_gpus_per_node=${TRAIN_GPU_COUNT} \\\\|" \
    "${TRAIN_SCRIPT}" > "${PATCHED_TRAIN_SCRIPT}"
chmod +x "${PATCHED_TRAIN_SCRIPT}"

cd "${REPO_ROOT}"
bash "${PATCHED_TRAIN_SCRIPT}" "$@"
