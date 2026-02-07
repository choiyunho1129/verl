import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verl.utils.reward_score.math_verify import compute_score as mv_compute_score


def normalize_messages(prompt: Any) -> Optional[List[Dict[str, str]]]:
    if prompt is None:
        return None
    if isinstance(prompt, list):
        return prompt
    if hasattr(prompt, "tolist") and not isinstance(prompt, (str, bytes)):
        try:
            prompt = prompt.tolist()
        except Exception:
            prompt = None
        if isinstance(prompt, list):
            return prompt
    if isinstance(prompt, str):
        try:
            parsed = json.loads(prompt)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            return [{"role": "user", "content": prompt}]
    return None


def build_chat_prompt(tokenizer, messages: List[Dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    except Exception:
        parts = []
        for msg in messages:
            role = (msg.get("role") or "user").capitalize()
            content = msg.get("content") or ""
            parts.append(f"{role}: {content}")
        return "\n\n".join(parts) + "\n\nAssistant:"


def build_variant_prompt(original_q: str, original_traj: str, critique: str, variant_q: str) -> str:
    return (
        f"Original Problem: {original_q}\n"
        f"Original Solution Trace: {original_traj}\n\n"
        f"Critique on the Original Solution: {critique}\n\n"
        f"Instruction: Using the critique above, solve the following variation problem. "
        f"Think step-by-step and put the final answer in \\boxed{{}}.\n\n"
        f"Variation Problem: {variant_q}"
    )


def batched_generate(llm, prompts: List[str], sampling_params, batch_size: int) -> List[str]:
    outputs: List[str] = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        results = llm.generate(batch_prompts, sampling_params)
        for result in results:
            if result.outputs:
                outputs.append(result.outputs[0].text)
            else:
                outputs.append("")
    return outputs


def extract_ground_truth(row: Dict[str, Any]) -> Optional[str]:
    reward = row.get("reward_model")
    if isinstance(reward, str):
        try:
            reward = json.loads(reward)
        except Exception:
            reward = None
    if isinstance(reward, dict):
        gt = reward.get("ground_truth")
        if gt:
            return gt
    gt = row.get("reward_model_data")
    if isinstance(gt, str) and gt:
        return gt
    return None


def parse_reward_meta(ground_truth: Any) -> Optional[Dict[str, Any]]:
    if isinstance(ground_truth, dict):
        return ground_truth
    if isinstance(ground_truth, str):
        try:
            return json.loads(ground_truth)
        except Exception:
            return None
    return None


def build_reward_prompts(
    metas: List[Optional[Dict[str, Any]]],
    critiques: List[str],
    valid_mask: List[bool],
    tokenizer,
    num_repeats: int,
) -> Tuple[List[str], List[Tuple[int, int, str]]]:
    prompts: List[str] = []
    mappings: List[Tuple[int, int, str]] = []
    for idx, meta in enumerate(metas):
        if not valid_mask[idx] or meta is None:
            continue
        original_q = meta.get("original_question") or meta.get("question", "") or ""
        original_traj = meta.get("original_trajectory") or meta.get("trajectory", "") or ""
        variants = meta.get("variants", []) or []
        if not variants:
            continue
        for v_idx, variant in enumerate(variants):
            var_q = variant.get("q") or variant.get("question")
            var_a = variant.get("a") or variant.get("answer")
            if not var_q or not var_a:
                continue
            prompt_text = build_variant_prompt(original_q, original_traj, critiques[idx], var_q)
            prompt = build_chat_prompt(tokenizer, [{"role": "user", "content": prompt_text}])
            for _ in range(num_repeats):
                prompts.append(prompt)
                mappings.append((idx, v_idx, var_a))
    return prompts, mappings


def aggregate_scores(
    generations: List[str],
    mappings: List[Tuple[int, int, str]],
) -> Tuple[Dict[int, float], Dict[int, int], Dict[int, int]]:
    variant_sum: Dict[Tuple[int, int], float] = {}
    variant_count: Dict[Tuple[int, int], int] = {}
    for gen, mapping in zip(generations, mappings):
        sample_idx, v_idx, answer = mapping
        if gen:
            gen = gen.strip()
        try:
            score = float(mv_compute_score(gen or "", answer))
        except Exception:
            score = 0.0
        key = (sample_idx, v_idx)
        variant_sum[key] = variant_sum.get(key, 0.0) + score
        variant_count[key] = variant_count.get(key, 0) + 1

    sample_variant_sum: Dict[int, float] = {}
    sample_variant_count: Dict[int, int] = {}
    sample_gen_count: Dict[int, int] = {}
    for key, sumscore in variant_sum.items():
        count = variant_count.get(key, 0)
        if count == 0:
            continue
        sample_idx, _ = key
        variant_avg = sumscore / count
        sample_variant_sum[sample_idx] = sample_variant_sum.get(sample_idx, 0.0) + variant_avg
        sample_variant_count[sample_idx] = sample_variant_count.get(sample_idx, 0) + 1
        sample_gen_count[sample_idx] = sample_gen_count.get(sample_idx, 0) + count
    return sample_variant_sum, sample_variant_count, sample_gen_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="data/snapshots_variants_test/prompt_llama_3b_instruct_trajectories_1.parquet",
        help="Path to the test parquet dataset.",
    )
    parser.add_argument("--output-jsonl", type=str, default="/data01/yunhochoi/verl/data/eval_critique_llama3b.jsonl")
    parser.add_argument(
        "--critique-cache",
        type=str,
        default="",
        help="Path to cache critique JSONL. If empty, uses <input>.critique.jsonl.",
    )
    parser.add_argument(
        "--overwrite-critique-cache",
        action="store_true",
        help="Regenerate critiques and overwrite cache.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=25)

    # Critique model (Qwen 2.5 7B Instruct)
    parser.add_argument("--critique-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--critique-dtype", type=str, default="auto")
    parser.add_argument("--critique-max-model-len", type=int, default=6144)
    parser.add_argument("--critique-gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--critique-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--critique-max-new-tokens", type=int, default=2048)
    parser.add_argument("--critique-temperature", type=float, default=0.6)
    parser.add_argument("--critique-top-p", type=float, default=1.0)
    parser.add_argument("--critique-batch-size", type=int, default=32)

    # Reward model (Llama 3.2 3B Instruct)
    parser.add_argument("--reward-model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--reward-dtype", type=str, default="auto")
    parser.add_argument("--reward-max-model-len", type=int, default=8192)
    parser.add_argument("--reward-gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--reward-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--reward-max-new-tokens", type=int, default=2048)
    parser.add_argument("--reward-temperature", type=float, default=0.6)
    parser.add_argument("--reward-top-p", type=float, default=1.0)
    parser.add_argument("--reward-batch-size", type=int, default=16)
    parser.add_argument("--reward-num-repeats", type=int, default=3)

    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    rows = df.to_dict(orient="records")
    if args.limit is not None:
        rows = rows[: args.limit]

    input_path = Path(args.input)
    critique_cache_path = (
        Path(args.critique_cache)
        if args.critique_cache
        else input_path.with_suffix(".critique.jsonl")
    )
    if critique_cache_path.exists() and not args.overwrite_critique_cache:
        print(f"[critique] Using cache: {critique_cache_path}")
    else:
        print(f"[critique] Cache path: {critique_cache_path}")

    # ---- Phase 1: generate critiques with Qwen ----
    critique_tokenizer = AutoTokenizer.from_pretrained(
        args.critique_model, trust_remote_code=True
    )
    critique_llm = LLM(
        model=args.critique_model,
        trust_remote_code=True,
        dtype=args.critique_dtype,
        max_model_len=args.critique_max_model_len,
        gpu_memory_utilization=args.critique_gpu_memory_utilization,
        tensor_parallel_size=args.critique_tensor_parallel_size,
    )
    critique_sampling = SamplingParams(
        max_tokens=args.critique_max_new_tokens,
        temperature=args.critique_temperature,
        top_p=args.critique_top_p,
    )

    critiques: List[str] = [""] * len(rows)
    valid_mask: List[bool] = [False] * len(rows)
    metas: List[Optional[Dict[str, Any]]] = [None] * len(rows)
    skipped = 0

    critique_done: List[bool] = [False] * len(rows)
    if critique_cache_path.exists() and not args.overwrite_critique_cache:
        with critique_cache_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                idx = rec.get("idx")
                if isinstance(idx, int) and 0 <= idx < len(rows):
                    critiques[idx] = rec.get("critique") or ""
                    critique_done[idx] = True

    critique_prompts: List[str] = []
    critique_indices: List[int] = []

    for idx, row in enumerate(rows):
        messages = normalize_messages(row.get("prompt"))
        ground_truth = extract_ground_truth(row)
        meta = parse_reward_meta(ground_truth) if ground_truth else None
        metas[idx] = meta

        if not messages or not meta:
            skipped += 1
            continue

        valid_mask[idx] = True
        if not critique_done[idx]:
            critique_prompt = build_chat_prompt(critique_tokenizer, messages)
            critique_prompts.append(critique_prompt)
            critique_indices.append(idx)

    if critique_prompts:
        if critique_cache_path.exists() and args.overwrite_critique_cache:
            critique_cache_path.unlink()
        critique_cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_mode = "a" if critique_cache_path.exists() and not args.overwrite_critique_cache else "w"
        with critique_cache_path.open(cache_mode, encoding="utf-8") as cache_f:
            critique_outputs = batched_generate(
                critique_llm,
                critique_prompts,
                critique_sampling,
                batch_size=args.critique_batch_size,
            )
            for i, (row_idx, text) in enumerate(zip(critique_indices, critique_outputs)):
                critique_text = text or ""
                critiques[row_idx] = critique_text
                cache_f.write(
                    json.dumps({"idx": row_idx, "critique": critique_text}) + "\n"
                )
                if args.log_every and (i + 1) % args.log_every == 0:
                    print(f"[critique {i+1}/{len(critique_outputs)}] skipped={skipped}")

    valid_count = sum(1 for v in valid_mask if v)
    if valid_count == 0:
        raise ValueError(
            "No valid samples to score. Check that `prompt` is a chat message list "
            "and `reward_model.ground_truth` is a JSON string."
        )

    # Release critique model before loading reward model
    del critique_llm
    del critique_tokenizer
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass

    # ---- Phase 2: score variants with Llama ----
    reward_tokenizer = AutoTokenizer.from_pretrained(
        args.reward_model, trust_remote_code=True
    )
    reward_llm = LLM(
        model=args.reward_model,
        trust_remote_code=True,
        dtype=args.reward_dtype,
        max_model_len=args.reward_max_model_len,
        gpu_memory_utilization=args.reward_gpu_memory_utilization,
        tensor_parallel_size=args.reward_tensor_parallel_size,
    )
    reward_sampling = SamplingParams(
        max_tokens=args.reward_max_new_tokens,
        temperature=args.reward_temperature,
        top_p=args.reward_top_p,
    )

    out_f = None
    if args.output_jsonl:
        Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
        out_f = open(args.output_jsonl, "w", encoding="utf-8")

    reward_prompts, reward_mappings = build_reward_prompts(
        metas=metas,
        critiques=critiques,
        valid_mask=valid_mask,
        tokenizer=reward_tokenizer,
        num_repeats=args.reward_num_repeats,
    )

    sample_variant_sum: Dict[int, float] = {}
    sample_variant_count: Dict[int, int] = {}
    sample_gen_count: Dict[int, int] = {}
    if reward_prompts:
        reward_generations = batched_generate(
            reward_llm,
            reward_prompts,
            reward_sampling,
            batch_size=args.reward_batch_size,
        )
        sample_variant_sum, sample_variant_count, sample_gen_count = aggregate_scores(
            reward_generations, reward_mappings
        )

    total_acc = 0.0
    total_count = 0

    for idx, _ in enumerate(rows):
        if not valid_mask[idx]:
            if out_f is not None:
                out_f.write(json.dumps({"idx": idx, "skipped": True}) + "\n")
            continue

        num_variants = sample_variant_count.get(idx, 0)
        num_generations = sample_gen_count.get(idx, 0)
        if num_variants > 0:
            acc = sample_variant_sum.get(idx, 0.0) / num_variants
        else:
            acc = 0.0

        total_acc += acc
        total_count += 1

        if out_f is not None:
            out_f.write(
                json.dumps(
                    {
                        "idx": idx,
                        "acc": acc,
                        "num_variants": num_variants,
                        "num_generations": num_generations,
                        "critique": critiques[idx],
                    }
                )
                + "\n"
            )

        if args.log_every and (idx + 1) % args.log_every == 0:
            avg_acc = total_acc / max(total_count, 1)
            print(f"[score {idx+1}/{len(rows)}] avg_acc={avg_acc:.4f} skipped={skipped}")

    if out_f is not None:
        out_f.close()

    avg_acc = total_acc / max(total_count, 1)
    print(f"Processed: {len(rows)}")
    print(f"Scored: {total_count}")
    print(f"Skipped: {skipped}")
    print(f"Final accuracy: {avg_acc:.6f}")


if __name__ == "__main__":
    main()
