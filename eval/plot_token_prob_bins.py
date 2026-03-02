#!/usr/bin/env python3
"""
Plot token probability histograms for Qwen3-1.7B variants on AIME.

This script:
- Loads data/eval/test.id.jsonl
- Filters AIME rows (optional dedupe or output-diff filtering)
- Reads model outputs (jsonl with fields: prompt, generated_text)
- Scores generated tokens with the corresponding model to get per-token probabilities
- Plots histograms by probability interval
- For think-enabled outputs, splits tokens inside/outside <think>...</think>

Notes:
- Requires torch, transformers, matplotlib, numpy
- Model weights must be available locally (no download in restricted envs)
"""

import argparse
import json
import math
import os
import re
from typing import Dict, List, Tuple, Union, Optional

import numpy as np

THINK_START = "<think>"
THINK_END = "</think>"


def load_jsonl(path: str) -> Union[List[dict], Dict[int, dict]]:
    items: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    if items and "index" in items[0]:
        by_idx = {int(it["index"]): it for it in items}
        return by_idx
    return items


def build_prompt_key(prompt_msgs: List[dict]) -> str:
    # Stable key for dedupe. Keep order and fields.
    return json.dumps(prompt_msgs, ensure_ascii=False, sort_keys=True)


def select_indices(
    dataset_path: str,
    data_source: Optional[str],
    aime_only: bool,
    dedupe_prompt: bool,
) -> Tuple[List[int], Dict[int, str]]:
    indices = []
    prompt_keys = {}
    seen = set()
    with open(dataset_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            obj = json.loads(line)
            if data_source and data_source != "all":
                if obj.get("data_source") != data_source:
                    continue
            elif aime_only and obj.get("data_source") != "aime":
                continue
            key = build_prompt_key(obj.get("prompt", []))
            if dedupe_prompt:
                if key in seen:
                    continue
                seen.add(key)
            indices.append(idx)
            prompt_keys[idx] = key
    return indices, prompt_keys


def filter_indices_by_output_diff(
    indices: List[int],
    outputs_a: Union[List[dict], Dict[int, dict]],
    outputs_b: Union[List[dict], Dict[int, dict]],
) -> List[int]:
    out = []
    for i in indices:
        ia = outputs_a.get(i) if isinstance(outputs_a, dict) else (outputs_a[i] if i < len(outputs_a) else None)
        ib = outputs_b.get(i) if isinstance(outputs_b, dict) else (outputs_b[i] if i < len(outputs_b) else None)
        if ia is None or ib is None:
            continue
        ta = ia.get("generated_text", "").strip()
        tb = ib.get("generated_text", "").strip()
        if ta != tb:
            out.append(i)
    return out


def find_think_spans(text: str) -> List[Tuple[int, int]]:
    spans = []
    start = 0
    while True:
        s = text.find(THINK_START, start)
        if s == -1:
            break
        s2 = s + len(THINK_START)
        e = text.find(THINK_END, s2)
        if e == -1:
            break
        spans.append((s2, e))
        start = e + len(THINK_END)
    return spans


def token_overlaps_span(tok_span: Tuple[int, int], span: Tuple[int, int]) -> bool:
    ts, te = tok_span
    ss, se = span
    return ts < se and te > ss


def find_tag_spans(text: str) -> List[Tuple[int, int]]:
    spans = []
    for m in re.finditer(re.escape(THINK_START), text):
        spans.append((m.start(), m.end()))
    for m in re.finditer(re.escape(THINK_END), text):
        spans.append((m.start(), m.end()))
    return spans


def apply_qwen_math_template(question: str, tokenizer, enable_thinking=None):
    messages = [
        {
            "role": "system",
            "content": "Please reason step by step, and put your final answer within \\boxed{}.",
        },
        {"role": "user", "content": question},
    ]
    apply_kwargs = {}
    if enable_thinking is not None:
        apply_kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **apply_kwargs,
    )


def simplerl_template(question: str) -> str:
    return (
        '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
        + question
        + "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n"
    )


def build_gen_prompt(messages: List[dict], tokenizer, template: str, enable_thinking: Optional[bool]):
    apply_kwargs = {}
    if enable_thinking is not None:
        apply_kwargs["enable_thinking"] = enable_thinking
    if template == "own":
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **apply_kwargs)
    if template == "simplerl":
        return simplerl_template(messages[0]["content"])
    if template == "qwen":
        return apply_qwen_math_template(messages[0]["content"], tokenizer, enable_thinking=enable_thinking)
    if template == "no":
        for m in reversed(messages):
            if m.get("role") == "user":
                return m.get("content", "")
        return messages[-1].get("content", "") if messages else ""
    raise ValueError(f"Invalid template: {template}")


def load_dataset_messages(dataset_path: str) -> List[List[dict]]:
    messages = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            messages.append(obj.get("prompt", []))
    return messages


def generate_outputs_vllm(
    dataset_path: str,
    indices: List[int],
    model_path: str,
    template: str,
    enable_thinking: Optional[bool],
    remove_system: bool,
    temperature: float,
    top_p: float,
    max_tokens: int,
    n: int,
    tp_size: Optional[int],
    gpu_mem_utilization: float,
) -> Dict[int, dict]:
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    all_messages = load_dataset_messages(dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    apply_kwargs = {}
    if enable_thinking is not None:
        apply_kwargs["enable_thinking"] = enable_thinking

    gen_prompts = []
    gen_indices = []
    for idx in indices:
        cur_message = all_messages[idx]
        if remove_system and cur_message and cur_message[0].get("role") == "system":
            cur_message = cur_message[1:]
        if not cur_message:
            continue
        prompt = build_gen_prompt(cur_message, tokenizer, template, enable_thinking)
        gen_prompts.append(prompt)
        gen_indices.append(idx)

    if tp_size is None:
        tp_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, n=n)
    llm = LLM(model=model_path, tensor_parallel_size=tp_size, gpu_memory_utilization=gpu_mem_utilization)
    outputs = llm.generate(gen_prompts, sampling_params)

    out = {}
    for idx, out_item in zip(gen_indices, outputs):
        text = out_item.outputs[0].text if out_item.outputs else ""
        out[idx] = {
            "index": idx,
            "prompt": out_item.prompt,
            "generated_text": text,
        }
    return out


def save_outputs_jsonl(path: str, outputs: Dict[int, dict]):
    with open(path, "w", encoding="utf-8") as f:
        for idx in sorted(outputs.keys()):
            f.write(json.dumps(outputs[idx], ensure_ascii=False) + "\n")


def load_model_and_tokenizer(model_path: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device.startswith("cuda") else None,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def score_output_tokens(
    model,
    tokenizer,
    prompt: str,
    output: str,
    device: str,
    max_seq_len: int = None,
):
    import torch
    import torch.nn.functional as F

    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    output_enc = tokenizer(output, add_special_tokens=False, return_offsets_mapping=True)
    output_ids = output_enc["input_ids"]
    offsets = output_enc.get("offset_mapping")

    full_ids = prompt_ids + output_ids
    if len(full_ids) < 2:
        return [], [], output_ids, offsets

    if max_seq_len is not None and len(full_ids) > max_seq_len:
        # Skip too-long sequences to avoid incorrect truncation.
        return None, None, output_ids, offsets

    input_ids = torch.tensor(full_ids, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_ids).logits  # [1, T, V]
    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    # Output tokens correspond to positions after prompt
    prompt_len = len(prompt_ids)
    # shift_labels indices are 0..T-2 corresponding to original positions 1..T-1
    start = max(prompt_len - 1, 0)
    end = shift_labels.size(1) - 1

    if start > end:
        return [], [], output_ids, offsets

    log_probs = F.log_softmax(shift_logits, dim=-1)
    entropy = -(log_probs.exp() * log_probs).sum(-1)
    # Slice to output token positions
    out_log_probs = log_probs[0, start : end + 1, :]
    out_labels = shift_labels[0, start : end + 1]
    token_logprobs = out_log_probs.gather(1, out_labels.unsqueeze(-1)).squeeze(-1)
    out_entropy = entropy[0, start : end + 1]

    return (
        token_logprobs.detach().cpu().tolist(),
        out_entropy.detach().cpu().tolist(),
        output_ids,
        offsets,
    )


def collect_probs(
    outputs: Union[List[dict], Dict[int, dict]],
    indices: List[int],
    model_path: str,
    device: str,
    max_seq_len: int,
    split_think: bool,
    include_think_tags: bool,
    max_samples: int,
    trajectory_path: Optional[str] = None,
    trajectory_series: Optional[str] = None,
):
    model, tokenizer = load_model_and_tokenizer(model_path, device)

    all_probs = []
    all_entropies = []
    think_probs = []
    think_entropies = []
    non_think_probs = []
    non_think_entropies = []
    skipped = 0
    traj_f = open(trajectory_path, "w", encoding="utf-8") if trajectory_path else None

    for n, idx in enumerate(indices):
        if max_samples is not None and n >= max_samples:
            break
        item = outputs.get(idx) if isinstance(outputs, dict) else (outputs[idx] if idx < len(outputs) else None)
        if item is None:
            continue
        prompt = item.get("prompt", "")
        output = item.get("generated_text", "")

        token_logprobs, token_entropies, output_ids, offsets = score_output_tokens(
            model,
            tokenizer,
            prompt,
            output,
            device,
            max_seq_len=max_seq_len,
        )
        if token_logprobs is None or token_entropies is None:
            skipped += 1
            continue

        probs = [math.exp(lp) for lp in token_logprobs]
        entropies = token_entropies

        keep_mask = [True] * len(probs)
        in_think_flags = None
        spans = None

        if offsets is not None and split_think:
            spans = find_think_spans(output)
            if spans:
                in_think_flags = [
                    any(token_overlaps_span(tok_span, span) for span in spans) if tok_span is not None else False
                    for tok_span in offsets
                ]
            else:
                in_think_flags = [False] * len(probs)

        if offsets is not None and not include_think_tags:
            tag_spans = find_tag_spans(output)
            if tag_spans:
                keep_mask = [
                    not any(token_overlaps_span(tok_span, t) for t in tag_spans) if tok_span is not None else True
                    for tok_span in offsets
                ]

        if len(keep_mask) != len(probs):
            keep_mask = [True] * len(probs)

        probs = [p for p, k in zip(probs, keep_mask) if k]
        logprobs = [lp for lp, k in zip(token_logprobs, keep_mask) if k]
        entropies = [e for e, k in zip(entropies, keep_mask) if k]
        all_probs.extend(probs)
        all_entropies.extend(entropies)

        if split_think:
            if offsets is None or not spans:
                non_think_probs.extend(probs)
                non_think_entropies.extend(entropies)
            else:
                if in_think_flags is None:
                    non_think_probs.extend(probs)
                    non_think_entropies.extend(entropies)
                else:
                    in_think_flags = [f for f, k in zip(in_think_flags, keep_mask) if k]
                    for p, e, flag in zip(probs, entropies, in_think_flags):
                        if flag:
                            think_probs.append(p)
                            think_entropies.append(e)
                        else:
                            non_think_probs.append(p)
                            non_think_entropies.append(e)

        if traj_f is not None:
            filtered_offsets = None
            if offsets is not None:
                filtered_offsets = [off for off, k in zip(offsets, keep_mask) if k]
            traj_item = {
                "index": idx,
                "series": trajectory_series,
                "output_text": output,
                "offsets": filtered_offsets,
                "probs": probs,
                "logprobs": logprobs,
                "entropies": entropies,
            }
            if split_think and in_think_flags is not None:
                traj_item["in_think"] = in_think_flags
            traj_f.write(json.dumps(traj_item, ensure_ascii=False) + "\n")

    # Cleanup to release memory
    try:
        import torch
        del model
        torch.cuda.empty_cache()
    except Exception:
        pass
    if traj_f is not None:
        traj_f.close()

    return {
        "all": all_probs,
        "think": think_probs,
        "non_think": non_think_probs,
        "entropy_all": all_entropies,
        "entropy_think": think_entropies,
        "entropy_non_think": non_think_entropies,
        "skipped": skipped,
    }


def make_bins(num_bins: int, bin_size: float):
    if bin_size is not None:
        edges = np.arange(0.0, 1.0 + bin_size, bin_size)
    else:
        edges = np.linspace(0.0, 1.0, num_bins + 1)
    # Ensure last edge is exactly 1.0
    edges[-1] = 1.0
    return edges


def make_bins_from_data(values: List[float], num_bins: int, bin_size: Optional[float]):
    if not values:
        return np.linspace(0.0, 1.0, num_bins + 1)
    vmin = float(min(values))
    vmax = float(max(values))
    if vmin == vmax:
        vmin = max(0.0, vmin - 1e-6)
        vmax = vmax + 1e-6
    if bin_size is not None:
        edges = np.arange(vmin, vmax + bin_size, bin_size)
    else:
        edges = np.linspace(vmin, vmax, num_bins + 1)
    return edges


def hist_counts(probs: List[float], bins: np.ndarray):
    counts, _ = np.histogram(probs, bins=bins)
    return counts


def save_hist_csv(path: str, bins: np.ndarray, series: Dict[str, List[float]]):
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["bin_left", "bin_right", "series", "count", "fraction"])
        for name, probs in series.items():
            counts = hist_counts(probs, bins)
            total = float(sum(counts)) or 1.0
            for i in range(len(counts)):
                writer.writerow([
                    f"{bins[i]:.6f}",
                    f"{bins[i+1]:.6f}",
                    name,
                    int(counts[i]),
                    f"{counts[i] / total:.6f}",
                ])


def plot_histograms(
    out_path: str,
    bins: np.ndarray,
    series: Dict[str, List[float]],
    log_y: bool,
    title: str,
    xlabel: str = "Token Probability",
):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5))
    labels = list(series.keys())
    data = [series[k] for k in labels]
    weights = []
    for arr in data:
        total = len(arr)
        if total == 0:
            weights.append([])
        else:
            weights.append([1.0 / total] * total)
    plt.hist(data, bins=bins, weights=weights, label=labels, alpha=0.65, edgecolor="black")
    plt.xlabel(xlabel)
    plt.ylabel("Proportion of Tokens")
    if log_y:
        plt.yscale("log")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_jsonl", default="data/eval/test.id.jsonl")
    parser.add_argument("--data_source", default=None)
    parser.add_argument("--aime_only", action="store_true", default=True)
    parser.add_argument("--all_sources", action="store_true", default=False)
    parser.add_argument("--dedupe_prompt", action="store_true", default=True)
    parser.add_argument("--keep_duplicates", action="store_true", default=False)
    parser.add_argument("--filter_output_diff", action="store_true", default=False)
    parser.add_argument("--diff_output_a", default="eval/results/qwen3_1_7b_think.jsonl")
    parser.add_argument("--diff_output_b", default="eval/results/qwen3_1_7b_no_think.jsonl")

    parser.add_argument("--outputs_base", default="eval/results/qwen3_1_7b_base.jsonl")
    parser.add_argument("--outputs_no_think", default="eval/results/qwen3_1_7b_no_think.jsonl")
    parser.add_argument("--outputs_think", default="eval/results/qwen3_1_7b_think.jsonl")

    parser.add_argument("--model_base", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--model_instruct", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--label_base", default=None)
    parser.add_argument("--label_instruct", default=None)
    parser.add_argument("--label_think", default=None)

    parser.add_argument("--preset", default=None, choices=["qwen3_aime", "qwen25_1_5b_math"])
    parser.add_argument("--skip_think", action="store_true", default=False)
    parser.add_argument("--generate", action="store_true", default=False)
    parser.add_argument("--regenerate", action="store_true", default=False)
    parser.add_argument("--template_base", default="no")
    parser.add_argument("--template_instruct", default="own")
    parser.add_argument("--remove_system", action="store_true", default=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--tp_size", type=int, default=None)
    parser.add_argument("--gpu_mem_utilization", type=float, default=0.85)
    parser.add_argument("--base_enable_thinking", default="False")
    parser.add_argument("--no_think_enable_thinking", default="False")
    parser.add_argument("--think_enable_thinking", default="True")

    parser.add_argument("--device", default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "") else "cpu")
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--include_think_tags", action="store_true", default=False)

    parser.add_argument("--num_bins", type=int, default=10)
    parser.add_argument("--bin_size", type=float, default=None)
    parser.add_argument("--log_y", action="store_true", default=False)
    parser.add_argument("--entropy_num_bins", type=int, default=None)
    parser.add_argument("--entropy_bin_size", type=float, default=None)
    parser.add_argument("--no_save_trajectories", action="store_false", dest="save_trajectories", default=True)
    parser.add_argument("--trajectory_dir", default=None)

    parser.add_argument("--out_dir", default="eval/plots")
    args = parser.parse_args()

    if args.preset == "qwen3_aime":
        args.data_source = "aime"
        args.model_base = "Qwen/Qwen3-1.7B-Base"
        args.model_instruct = "Qwen/Qwen3-1.7B"
        args.outputs_base = "eval/results/qwen3_1_7b_base.jsonl"
        args.outputs_no_think = "eval/results/qwen3_1_7b_no_think.jsonl"
        args.outputs_think = "eval/results/qwen3_1_7b_think.jsonl"
        args.template_base = "no"
        args.template_instruct = "own"
        args.skip_think = False
    elif args.preset == "qwen25_1_5b_math":
        args.data_source = "math"
        args.model_base = "Qwen/Qwen2.5-1.5B"
        args.model_instruct = "Qwen/Qwen2.5-1.5B-Instruct"
        args.outputs_base = "eval/results/qwen2_5_1_5b_base.jsonl"
        args.outputs_no_think = "eval/results/qwen2_5_1_5b_instruct.jsonl"
        args.outputs_think = "eval/results/qwen2_5_1_5b_think.jsonl"
        args.template_base = "no"
        args.template_instruct = "qwen"
        args.skip_think = True

    # Ensure multiprocessing uses spawn to avoid CUDA re-init errors in forked subprocesses.
    import multiprocessing as mp
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # Start method already set (e.g., in interactive environments).
        pass

    try:
        import torch
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA not available; falling back to CPU.")
            args.device = "cpu"
    except Exception:
        args.device = "cpu"

    if args.all_sources:
        args.aime_only = False
    if args.keep_duplicates:
        args.dedupe_prompt = False
    if args.data_source:
        args.aime_only = False
        args.all_sources = False
    elif args.all_sources:
        args.data_source = "all"

    def normalize_bool(val):
        if isinstance(val, bool) or val is None:
            return val
        s = str(val).strip().lower()
        if s in {"true", "1", "yes", "y", "t"}:
            return True
        if s in {"false", "0", "no", "n", "f"}:
            return False
        return None

    base_enable_thinking = normalize_bool(args.base_enable_thinking)
    no_think_enable_thinking = normalize_bool(args.no_think_enable_thinking)
    think_enable_thinking = normalize_bool(args.think_enable_thinking)
    do_think = not args.skip_think

    def sanitize_label(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")

    data_source_label = args.data_source or ("aime" if args.aime_only else "all")
    label_base = args.label_base or sanitize_label(args.model_base)
    label_instruct = args.label_instruct or sanitize_label(args.model_instruct)
    label_think = args.label_think or f"{label_instruct}_think"
    file_tag = sanitize_label(data_source_label)

    os.makedirs(args.out_dir, exist_ok=True)
    trajectory_dir = args.trajectory_dir or args.out_dir
    if args.save_trajectories:
        os.makedirs(trajectory_dir, exist_ok=True)
        traj_base_path = os.path.join(trajectory_dir, f"trajectory_{file_tag}_{label_base}.jsonl")
        traj_no_think_path = os.path.join(trajectory_dir, f"trajectory_{file_tag}_{label_instruct}.jsonl")
        traj_think_path = (
            os.path.join(trajectory_dir, f"trajectory_{file_tag}_{label_think}.jsonl") if do_think else None
        )
    else:
        traj_base_path = None
        traj_no_think_path = None
        traj_think_path = None

    indices, _ = select_indices(
        args.dataset_jsonl,
        data_source=args.data_source,
        aime_only=args.aime_only,
        dedupe_prompt=args.dedupe_prompt,
    )

    outputs_base = None
    outputs_no_think = None
    outputs_think = None

    if args.regenerate or (args.generate and not os.path.exists(args.outputs_base)):
        outputs_base = generate_outputs_vllm(
            args.dataset_jsonl,
            indices,
            model_path=args.model_base,
            template=args.template_base,
            enable_thinking=base_enable_thinking,
            remove_system=args.remove_system,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            n=args.n,
            tp_size=args.tp_size,
            gpu_mem_utilization=args.gpu_mem_utilization,
        )
        save_outputs_jsonl(args.outputs_base, outputs_base)
    elif os.path.exists(args.outputs_base):
        outputs_base = load_jsonl(args.outputs_base)
    else:
        raise FileNotFoundError(
            f"Missing output file: {args.outputs_base}. "
            "Pass --generate to create it with vLLM."
        )

    if args.regenerate or (args.generate and not os.path.exists(args.outputs_no_think)):
        outputs_no_think = generate_outputs_vllm(
            args.dataset_jsonl,
            indices,
            model_path=args.model_instruct,
            template=args.template_instruct,
            enable_thinking=no_think_enable_thinking,
            remove_system=args.remove_system,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            n=args.n,
            tp_size=args.tp_size,
            gpu_mem_utilization=args.gpu_mem_utilization,
        )
        save_outputs_jsonl(args.outputs_no_think, outputs_no_think)
    elif os.path.exists(args.outputs_no_think):
        outputs_no_think = load_jsonl(args.outputs_no_think)
    else:
        raise FileNotFoundError(
            f"Missing output file: {args.outputs_no_think}. "
            "Pass --generate to create it with vLLM."
        )

    if do_think:
        if args.regenerate or (args.generate and not os.path.exists(args.outputs_think)):
            outputs_think = generate_outputs_vllm(
                args.dataset_jsonl,
                indices,
                model_path=args.model_instruct,
                template=args.template_instruct,
                enable_thinking=think_enable_thinking,
                remove_system=args.remove_system,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                n=args.n,
                tp_size=args.tp_size,
                gpu_mem_utilization=args.gpu_mem_utilization,
            )
            save_outputs_jsonl(args.outputs_think, outputs_think)
        elif os.path.exists(args.outputs_think):
            outputs_think = load_jsonl(args.outputs_think)
        else:
            raise FileNotFoundError(
                f"Missing output file: {args.outputs_think}. "
                "Pass --generate to create it with vLLM."
            )

    if args.filter_output_diff:
        if do_think:
            outputs_a = load_jsonl(args.diff_output_a) if os.path.exists(args.diff_output_a) else outputs_think
            outputs_b = load_jsonl(args.diff_output_b) if os.path.exists(args.diff_output_b) else outputs_no_think
        else:
            outputs_a = load_jsonl(args.diff_output_a) if os.path.exists(args.diff_output_a) else outputs_base
            outputs_b = load_jsonl(args.diff_output_b) if os.path.exists(args.diff_output_b) else outputs_no_think
        indices = filter_indices_by_output_diff(indices, outputs_a, outputs_b)

    print(f"Selected indices: {len(indices)}")

    base_probs = collect_probs(
        outputs_base,
        indices,
        model_path=args.model_base,
        device=args.device,
        max_seq_len=args.max_seq_len,
        split_think=False,
        include_think_tags=args.include_think_tags,
        max_samples=args.max_samples,
        trajectory_path=traj_base_path,
        trajectory_series=label_base,
    )

    no_think_probs = collect_probs(
        outputs_no_think,
        indices,
        model_path=args.model_instruct,
        device=args.device,
        max_seq_len=args.max_seq_len,
        split_think=False,
        include_think_tags=args.include_think_tags,
        max_samples=args.max_samples,
        trajectory_path=traj_no_think_path,
        trajectory_series=label_instruct,
    )

    think_probs = None
    if do_think:
        think_probs = collect_probs(
            outputs_think,
            indices,
            model_path=args.model_instruct,
            device=args.device,
            max_seq_len=args.max_seq_len,
            split_think=True,
            include_think_tags=args.include_think_tags,
            max_samples=args.max_samples,
            trajectory_path=traj_think_path,
            trajectory_series=label_think,
        )

    bins = make_bins(args.num_bins, args.bin_size)
    entropy_bins_count = args.entropy_num_bins or args.num_bins
    entropy_values = base_probs["entropy_all"] + no_think_probs["entropy_all"]
    if do_think and think_probs is not None:
        entropy_values = entropy_values + think_probs["entropy_all"]
    entropy_bins = make_bins_from_data(entropy_values, entropy_bins_count, args.entropy_bin_size)

    # Plot 1: base vs instruct
    plot_histograms(
        os.path.join(args.out_dir, f"prob_bins_{file_tag}_{label_base}_vs_{label_instruct}.png"),
        bins,
        {
            label_base: base_probs["all"],
            label_instruct: no_think_probs["all"],
        },
        log_y=args.log_y,
        title=f"Token Probability Bins ({data_source_label}) - Base vs Instruct",
        xlabel="Token Probability",
    )
    plot_histograms(
        os.path.join(args.out_dir, f"entropy_bins_{file_tag}_{label_base}_vs_{label_instruct}.png"),
        entropy_bins,
        {
            label_base: base_probs["entropy_all"],
            label_instruct: no_think_probs["entropy_all"],
        },
        log_y=args.log_y,
        title=f"Token Entropy Bins ({data_source_label}) - Base vs Instruct",
        xlabel="Token Entropy",
    )

    if do_think and think_probs is not None:
        # Plot 2: think vs no-think
        plot_histograms(
            os.path.join(args.out_dir, f"prob_bins_{file_tag}_{label_think}_vs_{label_instruct}.png"),
            bins,
            {
                label_think: think_probs["all"],
                label_instruct: no_think_probs["all"],
            },
            log_y=args.log_y,
            title=f"Token Probability Bins ({data_source_label}) - Think vs No-Think",
            xlabel="Token Probability",
        )
        plot_histograms(
            os.path.join(args.out_dir, f"entropy_bins_{file_tag}_{label_think}_vs_{label_instruct}.png"),
            entropy_bins,
            {
                label_think: think_probs["entropy_all"],
                label_instruct: no_think_probs["entropy_all"],
            },
            log_y=args.log_y,
            title=f"Token Entropy Bins ({data_source_label}) - Think vs No-Think",
            xlabel="Token Entropy",
        )

        # Plot 3: think inside vs outside
        plot_histograms(
            os.path.join(args.out_dir, f"prob_bins_{file_tag}_{label_think}_inside_outside.png"),
            bins,
            {
                "think_inside": think_probs["think"],
                "think_outside": think_probs["non_think"],
            },
            log_y=args.log_y,
            title=f"Token Probability Bins ({data_source_label}) - Think Inside vs Outside",
            xlabel="Token Probability",
        )
        plot_histograms(
            os.path.join(args.out_dir, f"entropy_bins_{file_tag}_{label_think}_inside_outside.png"),
            entropy_bins,
            {
                "think_inside": think_probs["entropy_think"],
                "think_outside": think_probs["entropy_non_think"],
            },
            log_y=args.log_y,
            title=f"Token Entropy Bins ({data_source_label}) - Think Inside vs Outside",
            xlabel="Token Entropy",
        )

    # Save CSVs for reproducibility
    save_hist_csv(
        os.path.join(args.out_dir, f"prob_bins_{file_tag}_{label_base}_vs_{label_instruct}.csv"),
        bins,
        {
            label_base: base_probs["all"],
            label_instruct: no_think_probs["all"],
        },
    )
    save_hist_csv(
        os.path.join(args.out_dir, f"entropy_bins_{file_tag}_{label_base}_vs_{label_instruct}.csv"),
        entropy_bins,
        {
            label_base: base_probs["entropy_all"],
            label_instruct: no_think_probs["entropy_all"],
        },
    )
    if do_think and think_probs is not None:
        save_hist_csv(
            os.path.join(args.out_dir, f"prob_bins_{file_tag}_{label_think}_vs_{label_instruct}.csv"),
            bins,
            {
                label_think: think_probs["all"],
                label_instruct: no_think_probs["all"],
            },
        )
        save_hist_csv(
            os.path.join(args.out_dir, f"prob_bins_{file_tag}_{label_think}_inside_outside.csv"),
            bins,
            {
                "think_inside": think_probs["think"],
                "think_outside": think_probs["non_think"],
            },
        )
        save_hist_csv(
            os.path.join(args.out_dir, f"entropy_bins_{file_tag}_{label_think}_vs_{label_instruct}.csv"),
            entropy_bins,
            {
                label_think: think_probs["entropy_all"],
                label_instruct: no_think_probs["entropy_all"],
            },
        )
        save_hist_csv(
            os.path.join(args.out_dir, f"entropy_bins_{file_tag}_{label_think}_inside_outside.csv"),
            entropy_bins,
            {
                "think_inside": think_probs["entropy_think"],
                "think_outside": think_probs["entropy_non_think"],
            },
        )

    # Summary
    print("Done. Skipped sequences (too long):")
    print("  base:", base_probs["skipped"])
    print("  no_think:", no_think_probs["skipped"])
    if do_think and think_probs is not None:
        print("  think:", think_probs["skipped"])


if __name__ == "__main__":
    main()
