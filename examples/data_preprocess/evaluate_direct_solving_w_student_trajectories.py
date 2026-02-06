import argparse
import gc
import json
import multiprocessing as mp
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOLVER_CKPT = (
    REPO_ROOT
    / "checkpoints/verl_grpo_critique/qwen2.5_7b_instruct_MATH3-5_w_student_trajectories/global_step_120"
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score import math_verify as math_verify_metric

QUESTION_KEYS = ("question", "problem", "instruction", "input")
ANSWER_KEYS = ("answer", "solution", "ground_truth")

HF_WEIGHT_PATTERNS = (
    "model*.safetensors",
    "pytorch_model*.bin",
    "model*.index.json",
    "pytorch_model*.index.json",
)


def load_samples(path: Path) -> List[dict]:
    samples: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return samples


def get_question(sample: dict) -> str:
    prompt = sample.get("prompt") or sample.get("messages")
    if isinstance(prompt, list):
        for msg in prompt:
            if msg.get("role") == "user" and msg.get("content"):
                return msg["content"]

    extra = sample.get("extra_info") or {}
    if isinstance(extra, dict):
        q = extra.get("question")
        if q:
            return q

    for key in QUESTION_KEYS:
        val = sample.get(key)
        if val:
            return val

    raise KeyError(f"Question not found for sample keys: {list(sample.keys())}")


def get_ground_truth(sample: dict) -> str:
    reward = sample.get("reward_model") or {}
    if isinstance(reward, dict):
        gt = reward.get("ground_truth")
        if gt:
            return gt

    extra = sample.get("extra_info") or {}
    if isinstance(extra, dict):
        ans = extra.get("answer")
        if ans:
            return ans

    for key in ANSWER_KEYS:
        val = sample.get(key)
        if val:
            return val

    raise KeyError(f"Ground truth not found for sample keys: {list(sample.keys())}")


def build_student_prompt(tokenizer, sample: dict) -> str:
    messages = sample.get("prompt") or sample.get("messages")
    if not isinstance(messages, list):
        question = get_question(sample)
        system_prompt = (
            sample.get("system")
            or sample.get("system_prompt")
            or (sample.get("extra_info") or {}).get("system")
        )
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})

    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )


def build_direct_prompt(tokenizer, question: str, trajectory: str) -> str:
    system_prompt = (
        "Solve the problem step by step and put your final answer within \\boxed{}. "
        "You may use the student's solution as a hint."
    )
    user_prompt = f"Problem: {question}\n\nStudent Solution:\n{trajectory}\n\n"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )


def batched_generate(llm, prompts: List[str], sampling_params, batch_size: int) -> List[str]:
    texts: List[str] = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        outputs = llm.generate(batch_prompts, sampling_params)
        for out in outputs:
            texts.append(out.outputs[0].text if out.outputs else "")
    return texts


def write_jsonl(path: Path, records: List[dict]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def has_hf_weights(path: Path) -> bool:
    for pattern in HF_WEIGHT_PATTERNS:
        if any(path.glob(pattern)):
            return True
    return False


def is_fsdp_dir(path: Path) -> bool:
    return (path / "fsdp_config.json").exists() or any(
        path.glob("model_world_size_*")
    )


def ensure_hf_model(path_str: str, role: str) -> str:
    """
    If path points to an FSDP checkpoint, merge to HF and return merged path.
    Otherwise, return the original path.
    """
    path = Path(path_str).expanduser()
    if not path.exists():
        # Assume HF Hub id or remote model; leave untouched
        print(f"[{role}] Treating '{path_str}' as remote/HF id (not found locally).")
        return path_str

    actor_dir = path
    if (path / "actor").exists():
        actor_dir = path / "actor"

    # If already HF weights, return directly
    if has_hf_weights(actor_dir):
        return str(actor_dir)

    # If an inner huggingface folder exists with weights, use it
    hf_inner = actor_dir / "huggingface"
    if hf_inner.exists() and has_hf_weights(hf_inner):
        return str(hf_inner)

    # FSDP? then merge
    if is_fsdp_dir(actor_dir):
        merged_dir = actor_dir / "hf_merged"
        if not has_hf_weights(merged_dir):
            merged_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                "-m",
                "verl.model_merger",
                "merge",
                "--backend",
                "fsdp",
                "--local_dir",
                str(actor_dir),
                "--target_dir",
                str(merged_dir),
            ]
            print(f"[{role}] Merging FSDP checkpoint -> HF at {merged_dir}")
            subprocess.run(cmd, check=True)
        return str(merged_dir)

    # Otherwise treat as HF model root
    return str(actor_dir)


def evaluate_shard(args) -> dict:
    data_path = Path(args.data_path).expanduser()
    samples = load_samples(data_path)
    shard_samples = samples[args.shard_id :: args.num_shards]
    print(f"Shard {args.shard_id}/{args.num_shards} - samples: {len(shard_samples)}")
    if not shard_samples:
        return {"correct": 0.0, "total": 0}

    student_model = args.student_model_path or args.model_path
    solver_model = args.solver_model_path or args.model_path
    same_model = student_model == solver_model

    # Stage 1: student generations
    student_tokenizer = AutoTokenizer.from_pretrained(
        student_model, trust_remote_code=True
    )
    student_llm = LLM(
        model=student_model,
        trust_remote_code=True,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    student_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.student_temperature,
        top_p=args.student_top_p,
    )
    solver_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.solver_temperature,
        top_p=args.solver_top_p,
    )

    student_prompts = [build_student_prompt(student_tokenizer, s) for s in shard_samples]
    student_outputs = batched_generate(
        student_llm, student_prompts, student_params, args.batch_size
    )

    # Free student model if using different solver model to fit in memory
    if not same_model:
        del student_llm
        del student_tokenizer
        gc.collect()

    # Stage 2: solver/teacher generations
    if same_model:
        solver_tokenizer = student_tokenizer
        solver_llm = student_llm
    else:
        solver_tokenizer = AutoTokenizer.from_pretrained(
            solver_model, trust_remote_code=True
        )
        solver_llm = LLM(
            model=solver_model,
            trust_remote_code=True,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
        )

    direct_prompts = []
    for sample, trajectory in zip(shard_samples, student_outputs):
        question = get_question(sample)
        direct_prompts.append(build_direct_prompt(solver_tokenizer, question, trajectory))

    final_outputs = batched_generate(
        solver_llm, direct_prompts, solver_params, args.batch_size
    )

    records = []
    student_records = []
    correct = 0.0
    total = len(shard_samples)

    for sample, student_sol, final_sol in zip(
        shard_samples, student_outputs, final_outputs
    ):
        question = get_question(sample)
        ground_truth = get_ground_truth(sample)

        if args.metric == "math_verify":
            score = math_verify_metric.compute_score(final_sol, ground_truth)
        else:
            score = default_compute_score(
                data_source="HuggingFaceH4/MATH-500",
                solution_str=final_sol,
                ground_truth=ground_truth,
            )

        correct += float(score)
        student_records.append(
            {
                "uid": sample.get("uid"),
                "question": question,
                "student_solution": student_sol,
            }
        )
        records.append(
            {
                "uid": sample.get("uid"),
                "question": question,
                "ground_truth": ground_truth,
                "student_solution": student_sol,
                "final_solution": final_sol,
                "score": float(score),
            }
        )

    if args.student_output:
        write_jsonl(Path(args.student_output), student_records)
        print(f"Saved student solutions to {args.student_output}")
    if args.final_output:
        write_jsonl(Path(args.final_output), records)
        print(f"Saved final solutions to {args.final_output}")

    # Explicitly free solver model when it's distinct
    if not same_model:
        del solver_llm
        del solver_tokenizer
        gc.collect()

    accuracy = correct / total if total else 0.0
    print(
        f"Shard {args.shard_id} accuracy: {correct}/{total} = {accuracy:.3%}"
    )

    return {"correct": correct, "total": total}


def run_worker(worker_args, device_id, queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    try:
        stats = evaluate_shard(worker_args)
        queue.put(stats)
    except Exception as exc:  # surface worker errors
        queue.put({"correct": 0.0, "total": 0, "error": str(exc)})
        raise


def merge_jsonl(shard_paths: List[Path], final_path: Path) -> None:
    if not shard_paths:
        return
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with final_path.open("w", encoding="utf-8") as fout:
        for shard in shard_paths:
            if not shard.exists():
                continue
            with shard.open("r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)


def main():
    parser = argparse.ArgumentParser(
        description="Two-stage Math500 evaluation with student + direct solving prompts."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/MATH-500/test_mathsystemprompt.jsonl",
        help="Path to MATH-500 test split (jsonl).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="(deprecated fallback) Model path if student/solver not provided.",
    )
    parser.add_argument(
        "--student-model-path",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Model path or HF id for student (first-pass) generation.",
    )
    parser.add_argument(
        "--solver-model-path",
        type=str,
        default=str(DEFAULT_SOLVER_CKPT),
        help="Model path or HF id for solver/teacher (second-pass) generation. "
        "Defaults to in-repo GRPO checkpoint (global_step_120) if present.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--student-temperature", type=float, default=0.6)
    parser.add_argument("--solver-temperature", type=float, default=0.6)
    parser.add_argument("--student-top-p", type=float, default=1.0)
    parser.add_argument("--solver-top-p", type=float, default=1.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=6144)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument(
        "--metric",
        type=str,
        default="math_verify",
        choices=["math_verify", "default"],
        help="Scoring function for final answers.",
    )
    parser.add_argument(
        "--student-output",
        type=str,
        default="outputs/math500/llama_student.jsonl",
        help="Where to save first-pass student solutions (jsonl).",
    )
    parser.add_argument(
        "--final-output",
        type=str,
        default="outputs/math500/qwen_rl_teacher_llama_student.jsonl",
        help="Where to save second-pass final solutions (jsonl).",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="2,3",
        help="Comma-separated GPU ids for data-parallel workers. Leave empty to run single process.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total shards when not using --gpu-ids (advanced).",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="Shard id when not using --gpu-ids (advanced).",
    )

    args = parser.parse_args()

    # Resolve model paths and merge FSDP checkpoints if needed
    args.student_model_path = ensure_hf_model(
        args.student_model_path or args.model_path, role="student"
    )
    args.solver_model_path = ensure_hf_model(
        args.solver_model_path or args.model_path, role="solver"
    )

    if args.gpu_ids:
        mp.set_start_method("spawn", force=True)
        devices = [d.strip() for d in args.gpu_ids.split(",") if d.strip()]
        if not devices:
            raise ValueError("No valid GPU ids provided in --gpu-ids.")

        processes = []
        shard_student_paths = []
        shard_final_paths = []
        queue: mp.Queue = mp.Queue()

        for shard_id, dev in enumerate(devices):
            worker_args = SimpleNamespace(**vars(args))
            worker_args.tensor_parallel_size = 1  # data-parallel: one GPU per worker
            worker_args.shard_id = shard_id
            worker_args.num_shards = len(devices)
            if args.student_output:
                worker_args.student_output = f"{args.student_output}.shard{shard_id}"
                shard_student_paths.append(Path(worker_args.student_output))
            if args.final_output:
                worker_args.final_output = f"{args.final_output}.shard{shard_id}"
                shard_final_paths.append(Path(worker_args.final_output))

            p = mp.Process(target=run_worker, args=(worker_args, dev, queue))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        stats = []
        while not queue.empty():
            stats.append(queue.get())

        if len(stats) != len(devices):
            raise RuntimeError(
                f"Expected {len(devices)} worker reports, but received {len(stats)}"
            )

        if any("error" in s for s in stats):
            errors = [s.get("error") for s in stats if s.get("error")]
            raise RuntimeError(f"Worker errors: {errors}")

        total_correct = sum(s["correct"] for s in stats)
        total_samples = sum(s["total"] for s in stats)

        if args.student_output and shard_student_paths:
            merge_jsonl(shard_student_paths, Path(args.student_output))
        if args.final_output and shard_final_paths:
            merge_jsonl(shard_final_paths, Path(args.final_output))
    else:
        results = evaluate_shard(args)
        total_correct = results["correct"]
        total_samples = results["total"]

    final_acc = total_correct / total_samples if total_samples else 0.0
    print(f"Final accuracy: {total_correct}/{total_samples} = {final_acc:.3%}")


if __name__ == "__main__":
    main()
