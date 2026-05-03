from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from classifer_training.extract_hidden_states import (
    _extract_messages,
    _infer_input_device,
    _render_prompt,
    _resolve_torch_dtype,
)
from classifer_training.utils import load_records, sanitize_name, write_jsonl

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_ROOT = REPO_ROOT / "classifer_training" / "artifacts" / "datasets"
DEFAULT_RUN_ROOT = REPO_ROOT / "classifer_training" / "artifacts" / "runs"

_THINK_PATTERN = re.compile(r"<think>\s*(.*?)\s*</think>", re.IGNORECASE | re.DOTALL)
_ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_LINE_ANSWER_PATTERNS = [
    re.compile(
        r"(?:^|\n)\s*(?:final\s+answer|answer|final)\s*[:：]\s*(.+?)\s*(?=\n|$)",
        re.IGNORECASE,
    ),
    re.compile(r"(?:^|\n)\s*(?:答案是|答案)\s*[:：]\s*(.+?)\s*(?=\n|$)"),
]

try:
    from math_verify import parse as math_parse
    from math_verify import verify as math_verify
except Exception:  # pragma: no cover - optional dependency
    math_parse = None
    math_verify = None

try:
    from verl.utils.reward_score.math_verify import compute_score as local_math_verify_score
except Exception:  # pragma: no cover - fallback only
    local_math_verify_score = None


def _resolve_input_records(input_path: Path) -> list[dict[str, Any]]:
    if input_path.is_dir():
        records: list[dict[str, Any]] = []
        for split_name in ("train", "validation", "test"):
            split_path = input_path / f"{split_name}.jsonl"
            if split_path.exists():
                records.extend(load_records(split_path))
        if records:
            return records
        raise FileNotFoundError(f"No train/validation/test JSONL files found under {input_path}.")
    return load_records(input_path)


def _chunked(items: list[Any], chunk_size: int) -> Iterable[list[Any]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    for start_idx in range(0, len(items), chunk_size):
        yield items[start_idx : start_idx + chunk_size]


def _split_reasoning_and_answer(generated_text: str) -> tuple[str, str]:
    reasoning_match = _THINK_PATTERN.search(generated_text)
    answer_match = _ANSWER_PATTERN.search(generated_text)

    reasoning_content = reasoning_match.group(1).strip() if reasoning_match else ""
    if answer_match:
        answer_content = answer_match.group(1).strip()
    elif reasoning_match and "</think>" in generated_text:
        answer_content = generated_text.split("</think>", maxsplit=1)[1].strip()
    else:
        heuristic_reasoning, heuristic_answer = _extract_final_answer(generated_text)
        reasoning_content = heuristic_reasoning
        answer_content = heuristic_answer or generated_text.strip()

    # When the model emits a long post-think explanation, prefer the final boxed
    # or "Final Answer:" span over the whole answer block.
    _, heuristic_answer = _extract_final_answer(answer_content or generated_text)
    if heuristic_answer:
        _, nested_heuristic_answer = _extract_final_answer(heuristic_answer)
        if nested_heuristic_answer:
            heuristic_answer = nested_heuristic_answer
    if heuristic_answer:
        answer_content = heuristic_answer

    return reasoning_content, answer_content


def _extract_boxed_answers(generated_text: str) -> list[tuple[str, int]]:
    answers: list[tuple[str, int]] = []
    search_start = 0
    marker = r"\boxed{"
    while True:
        boxed_start = generated_text.find(marker, search_start)
        if boxed_start == -1:
            break
        content_start = boxed_start + len(marker)
        depth = 1
        cursor = content_start
        while cursor < len(generated_text) and depth > 0:
            character = generated_text[cursor]
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
            cursor += 1
        if depth == 0:
            candidate = generated_text[content_start : cursor - 1].strip()
            if candidate:
                answers.append((candidate, boxed_start))
        search_start = content_start
    return answers


def _extract_final_answer(generated_text: str) -> tuple[str, str]:
    candidates: list[tuple[str, int]] = _extract_boxed_answers(generated_text)
    for pattern in _LINE_ANSWER_PATTERNS:
        for match in pattern.finditer(generated_text):
            candidate = match.group(1).strip()
            if candidate:
                candidates.append((candidate, match.start(1)))
    if not candidates:
        return "", ""
    answer_content, start_idx = candidates[-1]
    reasoning_content = generated_text[:start_idx].strip()
    return reasoning_content, answer_content


def _count_text_tokens(tokenizer, text: str) -> int:
    if not text:
        return 0
    encoded = tokenizer(text, add_special_tokens=False)
    input_ids = encoded.get("input_ids")
    if isinstance(input_ids, list):
        return len(input_ids)
    if torch.is_tensor(input_ids):
        return int(input_ids.numel())
    return 0


def _resolve_max_prompt_len(
    *,
    tokenizer,
    model_name_or_path: str,
    explicit_max_model_len: int | None,
    trust_remote_code: bool,
) -> int | None:
    if explicit_max_model_len is not None and explicit_max_model_len > 0:
        return int(explicit_max_model_len)

    for attr_owner in (
        AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        ),
        tokenizer,
    ):
        for attr_name in ("max_position_embeddings", "seq_length", "n_positions", "model_max_length"):
            value = getattr(attr_owner, attr_name, None)
            if isinstance(value, int) and 0 < value < 1_000_000_000:
                return int(value)
    return None


def _score_generated_answer(
    record: dict[str, Any],
    generated_text: str,
    answer_content: str,
    ground_truth: str,
    grader: str,
) -> tuple[int, dict[str, Any]]:
    normalized_ground_truth = str(ground_truth or "").strip()
    if not normalized_ground_truth:
        return 0, {}

    candidate_texts: list[str] = []
    seen_candidates: set[str] = set()
    for text in (answer_content, generated_text):
        normalized_text = str(text or "").strip()
        if not normalized_text:
            continue
        _, heuristic_answer = _extract_final_answer(normalized_text)
        if heuristic_answer:
            _, nested_heuristic_answer = _extract_final_answer(heuristic_answer)
            if nested_heuristic_answer:
                heuristic_answer = nested_heuristic_answer
        for candidate in (heuristic_answer, normalized_text):
            normalized_candidate = str(candidate or "").strip()
            if not normalized_candidate or normalized_candidate in seen_candidates:
                continue
            seen_candidates.add(normalized_candidate)
            candidate_texts.append(normalized_candidate)

    if not candidate_texts:
        return 0, {}

    if grader == "exact":
        return int(any(candidate_text == normalized_ground_truth for candidate_text in candidate_texts)), {}

    if math_parse is not None and math_verify is not None:
        try:
            gold = math_parse(f"${normalized_ground_truth}$")
            # Prefer answer-only parsing first. For long reasoning traces this tends
            # to be more stable than feeding the entire completion into Math-Verify.
            for text in candidate_texts:
                try:
                    predicted = math_parse(text)
                    if bool(math_verify(gold, predicted)):
                        return 1, {}
                except Exception:
                    continue
        except Exception:
            pass

    if local_math_verify_score is not None:
        for text in candidate_texts:
            try:
                if float(local_math_verify_score(text, normalized_ground_truth)) >= 1.0:
                    return 1, {}
            except Exception:
                continue

    return int(any(candidate_text == normalized_ground_truth for candidate_text in candidate_texts)), {}


def _build_experiment_row(
    *,
    record: dict[str, Any],
    dataset_name: str,
    config: dict[str, Any],
    tokenizer,
    generated_text: str,
    input_length: int,
    output_length: int,
    generation_time: float,
    sample_index: int = 0,
    sample_count: int = 1,
) -> tuple[dict[str, Any], int]:
    reasoning_content, answer_content = _split_reasoning_and_answer(generated_text)
    correctness, verification = _score_generated_answer(
        record=record,
        generated_text=generated_text,
        answer_content=answer_content,
        ground_truth=str(record.get("ground_truth", "")),
        grader=str(config["grader"]),
    )
    think_tokens = _count_text_tokens(tokenizer, reasoning_content)
    answer_tokens = _count_text_tokens(tokenizer, answer_content)

    experiment_row = {
        "dataset_name": dataset_name,
        "task_id": str(record.get("task_id", "")),
        "split": str(record.get("split", "train")),
        "user_input": str(record.get("user_input", "")),
        "ground_truth": str(record.get("ground_truth", "")),
        "test_cases": record.get("test_cases"),
        "messages": record.get("messages", []),
        "generated_text": generated_text,
        "reasoning_content": reasoning_content,
        "answer_content": answer_content,
        "input_length": int(input_length),
        "output_length": int(output_length),
        "generation_time": float(generation_time),
        "sample_index": int(sample_index),
        "sample_count": int(sample_count),
        "has_complete_answer": bool(answer_content.strip()),
        "token_stats": {
            "think_tokens": int(think_tokens),
            "answer_tokens": int(answer_tokens),
            "total_tokens": int(output_length),
        },
        "config": dict(config),
    }
    if verification:
        experiment_row["verification"] = verification
        if "score" in verification:
            experiment_row["score"] = float(verification["score"])
            experiment_row["reward"] = float(verification["score"])
    return experiment_row, correctness


def _generate_with_transformers(
    *,
    prompts: list[str],
    model_name_or_path: str,
    tokenizer,
    temperature: float,
    top_p: float,
    top_k: int | None,
    max_new_tokens: int,
    batch_size: int,
    trust_remote_code: bool,
    torch_dtype: str,
    seed: int,
    num_samples: int,
) -> tuple[list[list[str]], list[list[int]], list[list[float]]]:
    if not prompts:
        return [], [], []

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype=_resolve_torch_dtype(torch_dtype),
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    input_device = _infer_input_device(model)

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    generated_groups: list[list[str]] = []
    output_length_groups: list[list[int]] = []
    generation_time_groups: list[list[float]] = []
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if temperature > 0.0:
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
        if top_k is not None and top_k > 0:
            generation_kwargs["top_k"] = top_k
    else:
        generation_kwargs["do_sample"] = False
    if num_samples > 1:
        generation_kwargs["num_return_sequences"] = num_samples

    for prompt_batch in tqdm(list(_chunked(prompts, batch_size)), desc="Sampling", unit="batch"):
        tokenized = tokenizer(prompt_batch, return_tensors="pt", padding=True)
        input_lengths = tokenized["attention_mask"].sum(dim=1).tolist()
        tokenized = {key: value.to(input_device) for key, value in tokenized.items()}

        start_time = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**tokenized, **generation_kwargs)
        elapsed = time.perf_counter() - start_time
        per_sample_time = elapsed / max(len(prompt_batch) * max(num_samples, 1), 1)

        for batch_idx, input_length in enumerate(input_lengths):
            prompt_group_texts: list[str] = []
            prompt_group_lengths: list[int] = []
            prompt_group_times: list[float] = []
            for sample_idx in range(num_samples):
                generated_idx = batch_idx * num_samples + sample_idx
                generated_ids = generated[generated_idx, int(input_length) :]
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                prompt_group_texts.append(generated_text)
                prompt_group_lengths.append(int(generated_ids.numel()))
                prompt_group_times.append(float(per_sample_time))
            generated_groups.append(prompt_group_texts)
            output_length_groups.append(prompt_group_lengths)
            generation_time_groups.append(prompt_group_times)

    return generated_groups, output_length_groups, generation_time_groups


def _generate_with_vllm(
    *,
    prompts: list[str],
    tokenizer,
    model_name_or_path: str,
    temperature: float,
    top_p: float,
    top_k: int | None,
    max_new_tokens: int,
    batch_size: int,
    tensor_parallel_size: int | None,
    gpu_memory_utilization: float,
    max_model_len: int | None,
    max_num_batched_tokens: int | None,
    max_num_seqs: int | None,
    trust_remote_code: bool,
    seed: int,
    num_samples: int,
    enforce_eager: bool,
    disable_custom_all_reduce: bool,
) -> tuple[list[list[str]], list[list[int]], list[list[float]]]:
    # vLLM launches worker processes internally. In our environment, leaving this
    # unset can make it inherit a fork-based start method and crash during CUDA init.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise ImportError(
            "vLLM is not installed. Install vllm or rerun with --backend transformers."
        ) from exc

    if not prompts:
        return [], [], []

    resolved_tp = tensor_parallel_size or max(torch.cuda.device_count(), 1)
    llm = LLM(
        model=model_name_or_path,
        tensor_parallel_size=resolved_tp,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        trust_remote_code=trust_remote_code,
        enforce_eager=enforce_eager,
        disable_custom_all_reduce=disable_custom_all_reduce,
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k if top_k is not None and top_k > 0 else -1,
        max_tokens=max_new_tokens,
        seed=seed,
        n=num_samples,
    )

    generated_groups: list[list[str]] = []
    output_length_groups: list[list[int]] = []
    generation_time_groups: list[list[float]] = []
    for prompt_batch in tqdm(list(_chunked(prompts, batch_size)), desc="Sampling", unit="batch"):
        start_time = time.perf_counter()
        outputs = llm.generate(prompt_batch, sampling_params)
        elapsed = time.perf_counter() - start_time
        per_sample_time = elapsed / max(len(outputs) * max(num_samples, 1), 1)
        for output in outputs:
            prompt_group_texts: list[str] = []
            prompt_group_lengths: list[int] = []
            prompt_group_times: list[float] = []
            if not output.outputs:
                prompt_group_texts.append("")
                prompt_group_lengths.append(0)
                prompt_group_times.append(float(per_sample_time))
            else:
                for candidate in output.outputs:
                    generated_text = candidate.text
                    token_count = _count_text_tokens(tokenizer, generated_text)
                    prompt_group_texts.append(generated_text)
                    prompt_group_lengths.append(int(token_count))
                    prompt_group_times.append(float(per_sample_time))
            generated_groups.append(prompt_group_texts)
            output_length_groups.append(prompt_group_lengths)
            generation_time_groups.append(prompt_group_times)
    return generated_groups, output_length_groups, generation_time_groups


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample one stochastic run over a normalized dataset and write a run directory "
            "containing all_experiments.jsonl and evaluation_results.jsonl."
        )
    )
    parser.add_argument(
        "--model_name_or_path",
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        help="Hugging Face model id or local path.",
    )
    parser.add_argument(
        "--input_path",
        type=Path,
        required=True,
        help="Normalized dataset file or dataset directory produced by prepare_datasets.py.",
    )
    parser.add_argument("--dataset_name", default=None, help="Optional override for dataset_name in the output rows.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Run directory to write.")
    parser.add_argument("--backend", choices=("vllm", "transformers"), default="vllm")
    parser.add_argument("--grader", choices=("math_verify", "exact", "ifeval", "acecode"), default="math_verify")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0, help="Use <=0 to disable top-k filtering.")
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=1, help="Number of stochastic samples to draw per prompt.")
    parser.add_argument("--tensor_parallel_size", type=int, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--max_num_batched_tokens", type=int, default=None)
    parser.add_argument("--max_num_seqs", type=int, default=None)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--split_filter", nargs="*", default=None, help="Optional split names to keep, for example: train validation test")
    parser.add_argument(
        "--skip_overlong_prompts",
        "--skip-overlong-prompts",
        action="store_true",
        help="Skip prompts whose tokenized length exceeds the model context window instead of failing the shard.",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--torch_dtype", default="auto", choices=("auto", "float32", "float16", "bfloat16"))
    parser.add_argument("--disable_generation_prompt", action="store_true")
    parser.add_argument("--disable_thinking", action="store_true")
    parser.add_argument("--disable_chat_template", action="store_true")
    parser.add_argument("--enforce_eager", action="store_true", help="Force vLLM eager mode to avoid compile/cudagraph issues.")
    parser.add_argument(
        "--disable_custom_all_reduce",
        action="store_true",
        help="Disable vLLM's custom all-reduce kernels. Useful when TP startup fails in custom_all_reduce.cuh.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    input_path = args.input_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments_path = output_dir / "all_experiments.jsonl"
    evaluations_path = output_dir / "evaluation_results.jsonl"
    if (experiments_path.exists() or evaluations_path.exists()) and not args.overwrite:
        raise FileExistsError(
            f"Run artifacts already exist at {experiments_path} or {evaluations_path}. Pass --overwrite to replace."
        )

    records = _resolve_input_records(input_path)
    if args.split_filter:
        allowed_splits = {split_name.strip() for split_name in args.split_filter if split_name.strip()}
        records = [record for record in records if str(record.get("split", "")).strip() in allowed_splits]
    if args.max_examples is not None:
        records = records[: args.max_examples]
    if not records:
        raise ValueError(f"No records found in {args.input_path}.")

    dataset_name = args.dataset_name or str(records[0].get("dataset_name") or input_path.stem)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer.padding_side = "left"

    prompts: list[str] = []
    input_lengths: list[int] = []
    for record in records:
        messages = _extract_messages(record)
        prompt = _render_prompt(
            tokenizer,
            messages,
            add_generation_prompt=not args.disable_generation_prompt,
            enable_thinking=not args.disable_thinking,
            use_chat_template=not args.disable_chat_template,
        )
        prompts.append(prompt)
        encoded = tokenizer(prompt, return_tensors="pt")
        input_lengths.append(int(encoded["input_ids"].shape[1]))

    skipped_overlong_rows: list[dict[str, Any]] = []
    if args.skip_overlong_prompts:
        max_prompt_len = _resolve_max_prompt_len(
            tokenizer=tokenizer,
            model_name_or_path=args.model_name_or_path,
            explicit_max_model_len=args.max_model_len,
            trust_remote_code=args.trust_remote_code,
        )
        if max_prompt_len is None:
            print("Could not resolve model context length; overlong prompt skipping is disabled.")
        else:
            kept_records: list[dict[str, Any]] = []
            kept_prompts: list[str] = []
            kept_input_lengths: list[int] = []
            for row_idx, (record, prompt, input_length) in enumerate(zip(records, prompts, input_lengths)):
                if input_length > max_prompt_len:
                    skipped_overlong_rows.append(
                        {
                            "row_idx": int(row_idx),
                            "task_id": str(record.get("task_id", "")),
                            "split": str(record.get("split", "")),
                            "input_length": int(input_length),
                            "max_prompt_len": int(max_prompt_len),
                            "reason": "input_length_exceeds_model_context",
                        }
                    )
                    continue
                kept_records.append(record)
                kept_prompts.append(prompt)
                kept_input_lengths.append(input_length)
            records = kept_records
            prompts = kept_prompts
            input_lengths = kept_input_lengths
            if skipped_overlong_rows:
                write_jsonl(output_dir / "skipped_overlong_prompts.jsonl", skipped_overlong_rows)
                print(
                    f"Skipped {len(skipped_overlong_rows)} overlong prompt(s) "
                    f"with input_length > {max_prompt_len}."
                )
            if not records:
                raise ValueError("All records were skipped as overlong prompts.")

    if args.backend == "vllm":
        generated_groups, output_length_groups, generation_time_groups = _generate_with_vllm(
            prompts=prompts,
            tokenizer=tokenizer,
            model_name_or_path=args.model_name_or_path,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
            trust_remote_code=args.trust_remote_code,
            seed=args.seed,
            num_samples=args.num_samples,
            enforce_eager=args.enforce_eager,
            disable_custom_all_reduce=args.disable_custom_all_reduce,
        )
    else:
        generated_groups, output_length_groups, generation_time_groups = _generate_with_transformers(
            prompts=prompts,
            model_name_or_path=args.model_name_or_path,
            tokenizer=tokenizer,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=args.torch_dtype,
            seed=args.seed,
            num_samples=args.num_samples,
        )

    if not (len(records) == len(generated_groups) == len(output_length_groups) == len(generation_time_groups)):
        raise RuntimeError("Sampling outputs are misaligned with the dataset records.")

    config = {
        "model_name_or_path": args.model_name_or_path,
        "model_slug": sanitize_name(args.model_name_or_path),
        "backend": args.backend,
        "grader": args.grader,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "num_samples": args.num_samples,
        "skip_overlong_prompts": bool(args.skip_overlong_prompts),
        "num_skipped_overlong_prompts": int(len(skipped_overlong_rows)),
    }

    experiment_rows: list[dict[str, Any]] = []
    correctness: list[int] = []
    scores: list[float] = []
    for row_idx, record in enumerate(records):
        prompt_generated_texts = generated_groups[row_idx]
        prompt_output_lengths = output_length_groups[row_idx]
        prompt_generation_times = generation_time_groups[row_idx]
        if not (
            len(prompt_generated_texts)
            == len(prompt_output_lengths)
            == len(prompt_generation_times)
        ):
            raise RuntimeError(f"Prompt-level sampling outputs are misaligned for row {row_idx}.")
        for sample_idx, generated_text in enumerate(prompt_generated_texts):
            experiment_row, correct = _build_experiment_row(
                record=record,
                dataset_name=dataset_name,
                config=config,
                tokenizer=tokenizer,
                generated_text=generated_text,
                input_length=input_lengths[row_idx],
                output_length=prompt_output_lengths[sample_idx],
                generation_time=prompt_generation_times[sample_idx],
                sample_index=sample_idx,
                sample_count=len(prompt_generated_texts),
            )
            experiment_rows.append(experiment_row)
            correctness.append(int(correct))
            row_score = experiment_row.get("score")
            scores.append(float(row_score) if row_score is not None else float(correct))
        print(
            f"Processed {row_idx + 1}/{len(records)} "
            f"task_id={record.get('task_id', '')} "
            f"split={record.get('split', '')} "
            f"input_length={input_lengths[row_idx]} "
            f"samples={len(prompt_generated_texts)}"
        )

    evaluation_row = {
        "dataset_name": dataset_name,
        "num_examples": len(experiment_rows),
        "num_prompts": len(records),
        "num_skipped_overlong_prompts": int(len(skipped_overlong_rows)),
        "accuracy": float(sum(correctness) / len(correctness)),
        "correctness": correctness,
        "config": config,
    }
    if args.grader in {"ifeval", "acecode"}:
        evaluation_row["scores"] = scores
        evaluation_row["mean_score"] = float(sum(scores) / len(scores))
        if args.grader == "ifeval":
            evaluation_row["constraint_scores"] = scores
            evaluation_row["mean_constraint_score"] = float(sum(scores) / len(scores))
    write_jsonl(experiments_path, experiment_rows)
    write_jsonl(evaluations_path, [evaluation_row])
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "num_examples": len(experiment_rows),
                "accuracy": evaluation_row["accuracy"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
