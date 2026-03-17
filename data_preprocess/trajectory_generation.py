import argparse
import json
import os
import multiprocessing as mp
import sys
from types import SimpleNamespace
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime, timezone

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Ensure repo root is on PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Local evaluation
from verl.utils.reward_score import math_verify as math_verify_metric

DEFAULT_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}"


def _str_to_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}. Use true/false.")


def _extract_question(sample: dict) -> Optional[str]:
    for key in ("question", "problem", "instruction", "input"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _extract_messages(sample: dict) -> Optional[List[dict]]:
    for key in ("prompt", "messages"):
        candidate = sample.get(key)
        if not isinstance(candidate, list) or not candidate:
            continue
        normalized = []
        is_valid = True
        for msg in candidate:
            if not isinstance(msg, dict):
                is_valid = False
                break
            role = msg.get("role")
            content = msg.get("content")
            if role is None or content is None:
                is_valid = False
                break
            normalized.append({"role": role, "content": content})
        if is_valid and normalized:
            return normalized
    return None


def _set_system_prompt(messages: List[dict], system_prompt: Optional[str]) -> List[dict]:
    if system_prompt is None:
        return [dict(msg) for msg in messages]

    normalized = [dict(msg) for msg in messages]
    if normalized and normalized[0].get("role") == "system":
        normalized[0]["content"] = system_prompt
    else:
        normalized = [{"role": "system", "content": system_prompt}] + normalized
    return normalized


def _apply_chat_template(tokenizer, messages: List[dict], enable_thinking: bool) -> Tuple[str, bool]:
    try:
        return (
            tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=enable_thinking,
            ),
            False,
        )
    except TypeError as e:
        if "enable_thinking" not in str(e):
            raise
        return (
            tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            ),
            True,
        )


def _build_chat_prompt(
    sample: dict,
    tokenizer,
    system_prompt: Optional[str],
    enable_thinking: bool,
) -> Tuple[Optional[str], bool]:
    messages = _extract_messages(sample)
    if messages is None:
        question = _extract_question(sample)
        if question is None:
            return None, False
        messages = [{"role": "user", "content": question}]

    messages = _set_system_prompt(messages, system_prompt)
    if any(msg.get("content") is None for msg in messages):
        return None, False

    return _apply_chat_template(
        tokenizer=tokenizer,
        messages=messages,
        enable_thinking=enable_thinking,
    )


def _build_plain_prompt(sample: dict) -> Optional[str]:
    question = _extract_question(sample)
    if question is None:
        return None
    return f"User: {question}\nAssistant:"


def load_data(path: Path) -> List[dict]:
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return samples


def compute_accuracy_from_file(path: Path):
    """Compute math_verify accuracy from a generated trajectory file.

    This keeps the dataset untouched and only reads the JSONL file to
    report trajectory-level and best-of-question accuracies.
    """

    trajectory_correct = 0.0
    trajectory_total = 0
    per_question_scores = {}

    if not path.exists():
        print(f"[accuracy] File not found: {path}")
        return None

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            gt = item.get("answer")
            pred = item.get("trajectory")
            if gt is None or pred is None:
                continue

            score = float(math_verify_metric.compute_score(str(pred), str(gt)))
            trajectory_correct += score
            trajectory_total += 1

            qkey = (
                item.get("unique_id"),
                item.get("question"),
                item.get("answer"),
            )
            per_question_scores.setdefault(qkey, []).append(score)

    if trajectory_total == 0:
        print("[accuracy] No trajectories with answers found; accuracy not computed.")
        return None

    best_per_question = sum(max(scores) for scores in per_question_scores.values())
    question_total = len(per_question_scores)

    return {
        "trajectory_correct": trajectory_correct,
        "trajectory_total": trajectory_total,
        "trajectory_accuracy": trajectory_correct / trajectory_total,
        "question_correct": best_per_question,
        "question_total": question_total,
        "question_accuracy": best_per_question / question_total if question_total else 0.0,
    }


def print_accuracy_report(metrics: dict, label: str):
    if not metrics:
        return
    print(
        f"[accuracy] {label} | per-trajectory: {metrics['trajectory_accuracy']:.4f} "
        f"({metrics['trajectory_correct']:.1f}/{metrics['trajectory_total']}), "
        f"best-of-question: {metrics['question_accuracy']:.4f} "
        f"({metrics['question_correct']:.1f}/{metrics['question_total']})"
    )


def _default_accuracy_report_path(output_path: Path) -> Path:
    if output_path.suffix:
        return output_path.with_suffix(".accuracy.json")
    return output_path.with_name(f"{output_path.name}.accuracy.json")


def save_accuracy_report(metrics: dict, source_path: Path, report_path: Path, label: str):
    payload = {
        "label": label,
        "source_path": str(source_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if metrics:
        payload.update(metrics)
        payload["status"] = "ok"
    else:
        payload["status"] = "not_computed"
        payload["message"] = "Accuracy was not computed because no valid trajectories were found."
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"[accuracy] Saved report to {report_path}")


def report_and_save_accuracy(target_path: Path, label: str, report_path: Path):
    metrics = compute_accuracy_from_file(target_path)
    print_accuracy_report(metrics, label)
    save_accuracy_report(metrics, source_path=target_path, report_path=report_path, label=label)


def generate(args):
    data_path = Path(args.data_path).expanduser()
    samples = load_data(data_path)
    print(f"Loaded {len(samples)} samples.")
    if args.num_shards > 1:
        if args.shard_id < 0 or args.shard_id >= args.num_shards:
            raise ValueError(f"Invalid shard_id {args.shard_id} for num_shards {args.num_shards}")
        samples = samples[args.shard_id::args.num_shards]
        print(f"Shard {args.shard_id}/{args.num_shards}: {len(samples)} samples.")

    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    sampling_params = SamplingParams(
        n=args.num_trajectories,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    raw_prompts = []
    selected_samples = []  # Keep samples aligned with raw_prompts in case we skip any
    skipped = 0
    thinking_arg_unsupported_logged = False
    system_prompt = args.system_prompt.strip() if args.system_prompt is not None else None
    if system_prompt == "":
        system_prompt = None

    for s in samples:
        if args.use_chat_template:
            formatted_prompt, thinking_arg_unsupported = _build_chat_prompt(
                sample=s,
                tokenizer=tokenizer,
                system_prompt=system_prompt,
                enable_thinking=args.enable_thinking,
            )
            if (
                args.enable_thinking
                and thinking_arg_unsupported
                and not thinking_arg_unsupported_logged
            ):
                print(
                    "[prompt] Tokenizer chat template does not support enable_thinking; "
                    "falling back to default chat template behavior."
                )
                thinking_arg_unsupported_logged = True
        else:
            formatted_prompt = _build_plain_prompt(s)

        if formatted_prompt is None:
            skipped += 1
            continue

        raw_prompts.append(formatted_prompt)
        selected_samples.append(s)

    if skipped:
        print(f"Skipped {skipped} samples missing prompt content.")

    print(f"Generating {args.num_trajectories} trajectories per question...")
    outputs = llm.generate(raw_prompts, sampling_params)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for i, output in enumerate(outputs):
            base_item = selected_samples[i]
            if not output.outputs:
                continue
            for gen in output.outputs:
                item = dict(base_item)
                item["trajectory"] = gen.text
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Done. Saved to {output_path}")

def run_worker(worker_args, device_id):
    # Restrict each worker to a single GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    generate(worker_args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data-path", type=str, default="/data1/home/yunhochoi/verl/data/DeepMath-103K/train_1k.jsonl", help="Input JSONL path")
    parser.add_argument("--output-path", type=str, default="/data1/home/yunhochoi/verl/data/DeepMath-103K/train_1k_Qwen3_8B_trajectories_nothink_4.jsonl", help="Output JSONL path")
    
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-8B")
    # Keep one full copy of the model per GPU (data-parallel)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=6144)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)

    parser.add_argument("--batch-size", type=int, default=64) 
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--num-trajectories", type=int, default=4, help="Number of trajectories to generate per question")
    parser.add_argument(
        "--use-chat-template",
        type=_str_to_bool,
        default=True,
        metavar="{true,false}",
        help="Whether to use tokenizer chat template to build prompts.",
    )
    parser.add_argument(
        "--enable-thinking",
        type=_str_to_bool,
        default=False,
        metavar="{true,false}",
        help="Whether to pass enable_thinking to chat template (when supported).",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt used in chat-template mode. Use empty string to keep dataset/default behavior.",
    )
    # Run a single process by default so all samples are processed unless the user opts into sharding
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards (processes)")
    parser.add_argument("--shard-id", type=int, default=0, help="Shard index for this process")
    parser.add_argument("--gpu-ids", type=str, default="3", help="Comma-separated GPU ids for data-parallel inference (each GPU loads full model). Overrides num_shards.")
    parser.add_argument(
        "--accuracy-report-path",
        type=str,
        default=None,
        help="Where to save accuracy report JSON. Default: <output-path> with .accuracy.json suffix.",
    )
    parser.add_argument(
        "--report-accuracy",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()
    final_output_path = Path(args.output_path)
    accuracy_report_path = (
        Path(args.accuracy_report_path).expanduser()
        if args.accuracy_report_path
        else _default_accuracy_report_path(final_output_path)
    )
    if args.gpu_ids:
        # Multiprocessing with CUDA must use spawn to avoid forked CUDA init errors
        mp.set_start_method("spawn", force=True)
        devices = [d.strip() for d in args.gpu_ids.split(",") if d.strip()]
        if not devices:
            raise ValueError("No valid GPU ids provided in --gpu-ids.")

        processes = []
        shard_paths = []
        for shard_id, dev in enumerate(devices):
            worker_args = SimpleNamespace(**vars(args))
            worker_args.tensor_parallel_size = 1  # force one GPU per worker
            worker_args.shard_id = shard_id
            worker_args.num_shards = len(devices)
            worker_args.output_path = f"{args.output_path}.shard{shard_id}"
            p = mp.Process(target=run_worker, args=(worker_args, dev))
            p.start()
            processes.append(p)
            shard_paths.append(worker_args.output_path)

        for p in processes:
            p.join()

        # Merge shard outputs into the final file
        final_path = final_output_path
        final_path.parent.mkdir(parents=True, exist_ok=True)
        with final_path.open("w", encoding="utf-8") as fout:
            for shard_file in shard_paths:
                shard_path = Path(shard_file)
                if not shard_path.exists():
                    continue
                with shard_path.open("r", encoding="utf-8") as fin:
                    for line in fin:
                        fout.write(line)
        print(f"Merged {len(shard_paths)} shards into {args.output_path}")

        report_and_save_accuracy(final_path, "merged", accuracy_report_path)
    else:
        generate(args)
        if args.num_shards > 1:
            print(
                "[accuracy] Warning: --num-shards > 1 without --gpu-ids means this run is likely a partial shard."
            )
        report_and_save_accuracy(final_output_path, "single", accuracy_report_path)
