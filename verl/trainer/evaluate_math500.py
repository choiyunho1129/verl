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

DEFAULT_QUESTION_KEYS = ("problem", "question")
DEFAULT_ANSWER_KEYS = ("answer")


def load_math500(path: Path) -> List[dict]:
    samples: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def build_prompt(tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def extract_field(
    sample: dict,
    preferred_key: str,
    fallbacks: tuple[str, ...],
    field_name: str,
    sample_idx: int,
) -> str:
    keys_to_try = []
    if preferred_key:
        keys_to_try.append(preferred_key)
    keys_to_try.extend([k for k in fallbacks if k not in keys_to_try])

    for key in keys_to_try:
        if key in sample and sample[key] not in (None, ""):
            return sample[key]

    tried = ", ".join(keys_to_try)
    available = ", ".join(sample.keys())
    raise KeyError(
        f"Sample {sample_idx} is missing '{field_name}'. Tried [{tried}] but only found [{available}]."
    )


def evaluate(args):
    data_path = Path(args.data_path).expanduser()
    samples = load_math500(data_path)

    processed_samples = []
    skipped = 0
    for idx, sample in enumerate(samples):
        try:
            question = extract_field(
                sample, args.question_key, DEFAULT_QUESTION_KEYS, "question", idx
            )
            answer = extract_field(
                sample, args.answer_key, DEFAULT_ANSWER_KEYS, "answer", idx
            )
        except KeyError as exc:
            skipped += 1
            if skipped <= 3:
                print(f"[warning] {exc}")
            continue

        processed_samples.append({"question": question, "answer": answer})

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
    )

    prompts = [build_prompt(tokenizer, s["question"]) for s in processed_samples]
    total = len(prompts)
    correct = 0.0

    for i in range(0, total, args.batch_size):
        batch_prompts = prompts[i : i + args.batch_size]
        outputs = llm.generate(batch_prompts, sampling_params)

        for j, out in enumerate(outputs):
            pred_text = out.outputs[0].text
            gt = processed_samples[i + j]["answer"]

            if args.metric == "math_verify":
                score = math_verify_metric.compute_score(pred_text, gt)
            else:
                score = default_compute_score(
                    data_source="HuggingFaceH4/MATH-500",
                    solution_str=pred_text,
                    ground_truth=gt,
                )

            correct += float(score)

        done = i + len(batch_prompts)
        print(f"Progress {done}/{total} | accuracy so far: {correct / done:.3f}")

    print(f"Final accuracy: {correct}/{total} = {correct / total:.3%}")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[3]
    default_data = "/data01/yunhochoi/verl/data/snapshots_variants_train/data.jsonl"

    parser = argparse.ArgumentParser(description="Evaluate MATH-500")
    parser.add_argument(
        "--model-path",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
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
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
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
            "Preferred key name for the question in each sample. "
            "Common fallbacks like question/prompt/query are tried automatically."
        ),
    )
    parser.add_argument(
        "--answer-key",
        type=str,
        default="answer",
        help="Preferred key name for the answer in each sample. Common fallbacks are tried automatically.",
    )

    evaluate(parser.parse_args())
