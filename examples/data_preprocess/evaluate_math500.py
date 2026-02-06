import argparse
import json
from pathlib import Path
from typing import List

import sys

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Ensure repo root is on PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score import math_verify as math_verify_metric

PROMPT_KEYS = ("prompt", "messages")
QUESTION_KEYS = ("problem", "question", "instruction", "input", "query")
ANSWER_KEYS = ("answer", "ground_truth")


def load_math500(path: Path) -> List[dict]:
    samples: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def _is_nonempty_text(value) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _get_nested(sample: dict, key: str):
    if not key:
        return None
    current = sample
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _normalize_messages(messages):
    if not isinstance(messages, list):
        return None
    normalized = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if isinstance(role, str) and _is_nonempty_text(content):
            normalized.append({"role": role, "content": content})
    return normalized or None


def extract_messages(sample: dict, preferred_key: str | None) -> list[dict] | None:
    keys_to_try = []
    if preferred_key in PROMPT_KEYS:
        keys_to_try.append(preferred_key)
    keys_to_try.extend([k for k in PROMPT_KEYS if k not in keys_to_try])

    for key in keys_to_try:
        messages = _normalize_messages(_get_nested(sample, key))
        if messages:
            return messages
    return None


def extract_question_and_messages(
    sample: dict, preferred_key: str | None, sample_idx: int
) -> tuple[str, list[dict] | None]:
    tried = []
    if preferred_key and preferred_key not in PROMPT_KEYS:
        tried.append(preferred_key)
        val = _get_nested(sample, preferred_key)
        if _is_nonempty_text(val):
            return val, None

    messages = extract_messages(sample, preferred_key)
    if messages:
        tried.extend(PROMPT_KEYS)
        for msg in reversed(messages):
            if msg.get("role") == "user" and _is_nonempty_text(msg.get("content")):
                return msg["content"], messages
        messages = None

    extra = sample.get("extra_info") or {}
    if isinstance(extra, dict):
        for key in ("question", "problem", "instruction", "input"):
            tried.append(f"extra_info.{key}")
            val = extra.get(key)
            if _is_nonempty_text(val):
                return val, messages

    for key in QUESTION_KEYS:
        if key == preferred_key:
            continue
        tried.append(key)
        val = sample.get(key)
        if _is_nonempty_text(val):
            return val, messages

    tried_str = ", ".join(tried) if tried else "question/problem/prompt"
    available = ", ".join(sample.keys())
    raise KeyError(
        f"Sample {sample_idx} is missing 'question'. Tried [{tried_str}] but only found [{available}]."
    )


def extract_answer(sample: dict, preferred_key: str | None, sample_idx: int) -> str:
    tried = []
    if preferred_key:
        tried.append(preferred_key)
        val = _get_nested(sample, preferred_key)
        if _is_nonempty_text(val):
            return val

    reward = sample.get("reward_model") or {}
    tried.append("reward_model.ground_truth")
    if isinstance(reward, dict):
        val = reward.get("ground_truth")
        if _is_nonempty_text(val):
            return val

    extra = sample.get("extra_info") or {}
    tried.append("extra_info.answer")
    if isinstance(extra, dict):
        val = extra.get("answer")
        if _is_nonempty_text(val):
            return val

    for key in ANSWER_KEYS:
        if key == preferred_key:
            continue
        tried.append(key)
        val = sample.get(key)
        if _is_nonempty_text(val):
            return val

    tried_str = ", ".join(tried) if tried else "answer/ground_truth"
    available = ", ".join(sample.keys())
    raise KeyError(
        f"Sample {sample_idx} is missing 'answer'. Tried [{tried_str}] but only found [{available}]."
    )


def extract_system_prompt(sample: dict) -> str | None:
    for key in ("system", "system_prompt"):
        val = sample.get(key)
        if _is_nonempty_text(val):
            return val

    extra = sample.get("extra_info") or {}
    if isinstance(extra, dict):
        val = extra.get("system")
        if _is_nonempty_text(val):
            return val
    return None


def build_prompt(tokenizer, sample: dict) -> str:
    messages = sample.get("messages")
    if not isinstance(messages, list):
        messages = []
        system_prompt = sample.get("system_prompt")
        if _is_nonempty_text(system_prompt):
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": sample["question"]})

    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )


def evaluate(args):
    data_path = Path(args.data_path).expanduser()
    samples = load_math500(data_path)

    processed_samples = []
    skipped = 0
    for idx, sample in enumerate(samples):
        try:
            question, messages = extract_question_and_messages(
                sample, args.question_key, idx
            )
            answer = extract_answer(sample, args.answer_key, idx)
        except KeyError as exc:
            skipped += 1
            if skipped <= 3:
                print(f"[warning] {exc}")
            continue

        processed_samples.append(
            {
                "question": question,
                "answer": answer,
                "messages": messages,
                "system_prompt": extract_system_prompt(sample),
            }
        )

    if not processed_samples:
        raise ValueError(
            f"No valid samples found in {data_path}. "
            f"Checked {len(samples)} entries but none had both question and answer fields."
        )
    if skipped:
        print(
            f"Skipped {skipped} samples missing required fields. "
            f"Evaluating {len(processed_samples)} samples."
        )

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
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        n=args.num_samples,
    )

    prompts = [build_prompt(tokenizer, s) for s in processed_samples]
    total = len(prompts)
    correct = 0.0

    for i in range(0, total, args.batch_size):
        batch_prompts = prompts[i : i + args.batch_size]
        outputs = llm.generate(batch_prompts, sampling_params)

        for j, out in enumerate(outputs):
            gt = processed_samples[i + j]["answer"]
            preds = [o.text for o in out.outputs if o.text]
            if not preds:
                preds = [""]

            scores = []
            for pred_text in preds:
                if args.metric == "math_verify":
                    score = math_verify_metric.compute_score(pred_text, gt)
                else:
                    score = default_compute_score(
                        data_source="HuggingFaceH4/MATH-500",
                        solution_str=pred_text,
                        ground_truth=gt,
                    )
                scores.append(float(score))

            correct += sum(scores) / len(scores)

        done = i + len(batch_prompts)
        print(f"Progress {done}/{total} | accuracy so far: {correct / done:.3f}")

    print(f"Final accuracy: {correct}/{total} = {correct / total:.3%}")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[3]
    default_data = "/data1/home/yunhochoi/verl/data/MATH-500/test_mathsystemprompt.jsonl"

    parser = argparse.ArgumentParser(description="Evaluate MATH-500")
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Local path or HF model ID (must be cached locally).",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(default_data),
        help="Path to MATH-500 jsonl.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of generations per problem; accuracy is the average over these attempts.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=6144)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="Fraction of GPU memory to use.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="vLLM dtype (e.g., float16, bfloat16, float32).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="math_verify",
        choices=["math_verify", "default"],
        help="Evaluation metric: math_verify or verl.utils.reward_score.default_compute_score (math_reward).",
    )
    parser.add_argument(
        "--question-key",
        type=str,
        default="problem",
        help=(
            "Preferred key name (or dotted path) for the question. "
            "If missing, falls back to prompt/messages, extra_info.question, or common keys."
        ),
    )
    parser.add_argument(
        "--answer-key",
        type=str,
        default="answer",
        help=(
            "Preferred key name (or dotted path) for the answer. "
            "If missing, falls back to reward_model.ground_truth, extra_info.answer, or common keys."
        ),
    )

    evaluate(parser.parse_args())
