import argparse
import json
import multiprocessing as mp
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Ensure repo root is on PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verl.utils.reward_score.math_verify import compute_score as mv_compute_score

VERIFIER_SYSTEM_PROMPT = (
    "You are a strict math solution verifier. "
    "Given a math question and a solution trajectory, decide whether the solution appears logically valid and the final answer is correct. "
    "Verify the trajectory step by step then provide a final verdict in the exact format: Final Verdict: CORRECT or Final Verdict: INCORRECT."
)


def _str_to_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}. Use true/false.")


@dataclass
class TrajectoryRecord:
    idx: int
    question_key: Tuple[str, ...]
    question: str
    answer: str
    trajectory: str
    actual_correct: bool
    model_raw: str = ""
    model_pred_correct: Optional[bool] = None
    pass_k_hit: Optional[bool] = None


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return num / den


def _extract_question(item: Dict[str, Any]) -> Optional[str]:
    for key in ("question", "problem", "instruction", "input"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for key in ("prompt", "messages"):
        messages = item.get(key)
        if not isinstance(messages, list):
            continue
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = (msg.get("role") or "").strip().lower()
            content = msg.get("content")
            if role == "user" and isinstance(content, str) and content.strip():
                return content.strip()
    return None


def _question_key(item: Dict[str, Any], question: str, answer: str) -> Tuple[str, ...]:
    unique_id = item.get("unique_id")
    if unique_id is not None and str(unique_id).strip():
        return ("uid", str(unique_id))
    return ("qa", question, answer)


def _build_verifier_user_prompt(record: TrajectoryRecord) -> str:
    return (
        f"Question:\n{record.question}\n\n"
        f"Solution Trajectory:\n{record.trajectory}\n\n"
    )


def _build_plain_verifier_prompt(record: TrajectoryRecord, system_prompt: Optional[str]) -> str:
    user_content = _build_verifier_user_prompt(record)
    if system_prompt:
        user_content = f"System Instruction:\n{system_prompt}\n\n{user_content}"
    return f"User:\n{user_content}\n\nAssistant:"


def _apply_chat_template(
    tokenizer,
    messages: List[Dict[str, str]],
    enable_thinking: bool,
) -> Tuple[str, bool]:
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
    except Exception:
        parts: List[str] = []
        for msg in messages:
            role = (msg.get("role") or "user").capitalize()
            content = msg.get("content") or ""
            parts.append(f"{role}: {content}")
        return "\n\n".join(parts) + "\n\nAssistant:", False


def _parse_verdict(text: str) -> Optional[bool]:
    if not text:
        return None
    normalized = text.strip().lower()

    verdict_matches = re.findall(
        r"final\s*verdict\s*[:\\-]\s*(correct|incorrect)\b",
        normalized,
    )
    if verdict_matches:
        return verdict_matches[-1] == "correct"

    token_matches = re.findall(r"\b(correct|incorrect)\b", normalized)
    if token_matches:
        return token_matches[-1] == "correct"

    return None


def _resolve_trajectory_path(input_path: Path) -> Path:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if input_path.suffix == ".json":
        try:
            payload = json.loads(input_path.read_text(encoding="utf-8"))
        except Exception:
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get("source_path"), str):
            src = Path(payload["source_path"]).expanduser()
            if src.exists():
                return src
            raise FileNotFoundError(
                f"Resolved source_path from accuracy report does not exist: {src}"
            )

    return input_path


def _load_records(path: Path, limit: Optional[int]) -> Tuple[List[TrajectoryRecord], int]:
    records: List[TrajectoryRecord] = []
    skipped = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            question = _extract_question(item)
            answer = item.get("answer")
            trajectory = item.get("trajectory")

            if not question or answer is None or trajectory is None:
                skipped += 1
                continue

            question_str = str(question)
            answer_str = str(answer)
            trajectory_str = str(trajectory)

            try:
                score = float(mv_compute_score(trajectory_str, answer_str))
            except Exception:
                score = 0.0
            actual_correct = score > 0.0

            records.append(
                TrajectoryRecord(
                    idx=len(records),
                    question_key=_question_key(item, question_str, answer_str),
                    question=question_str,
                    answer=answer_str,
                    trajectory=trajectory_str,
                    actual_correct=actual_correct,
                )
            )
            if limit is not None and len(records) >= limit:
                break
    return records, skipped


def _build_verifier_prompts(
    records: List[TrajectoryRecord],
    tokenizer,
    use_chat_template: bool,
    enable_thinking: bool,
    system_prompt: Optional[str],
) -> Tuple[List[str], bool]:
    prompts: List[str] = []
    thinking_arg_unsupported = False
    for rec in records:
        if use_chat_template:
            if system_prompt:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": _build_verifier_user_prompt(rec)},
                ]
            else:
                messages = [{"role": "user", "content": _build_verifier_user_prompt(rec)}]
            prompt, unsupported = _apply_chat_template(
                tokenizer=tokenizer,
                messages=messages,
                enable_thinking=enable_thinking,
            )
            if enable_thinking and unsupported:
                thinking_arg_unsupported = True
        else:
            prompt = _build_plain_verifier_prompt(
                record=rec,
                system_prompt=system_prompt,
            )
        prompts.append(prompt)
    return prompts, thinking_arg_unsupported


def _infer_verifier_results(
    records: List[TrajectoryRecord],
    model_path: str,
    dtype: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    use_chat_template: bool,
    enable_thinking: bool,
    system_prompt: Optional[str],
) -> Tuple[List[Tuple[int, str, Optional[bool]]], bool]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
    )
    sampling_params = SamplingParams(
        n=1,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    prompts, thinking_arg_unsupported = _build_verifier_prompts(
        records=records,
        tokenizer=tokenizer,
        use_chat_template=use_chat_template,
        enable_thinking=enable_thinking,
        system_prompt=system_prompt,
    )

    results: List[Tuple[int, str, Optional[bool]]] = []
    for start in range(0, len(prompts), batch_size):
        end = min(start + batch_size, len(prompts))
        outputs = llm.generate(prompts[start:end], sampling_params)
        for rec, output in zip(records[start:end], outputs):
            text = output.outputs[0].text if output.outputs else ""
            results.append((rec.idx, text, _parse_verdict(text)))

    return results, thinking_arg_unsupported


def _run_verifier_worker(
    records: List[TrajectoryRecord],
    device_id: str,
    model_path: str,
    dtype: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    use_chat_template: bool,
    enable_thinking: bool,
    system_prompt: Optional[str],
    queue,
) -> None:
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        results, thinking_arg_unsupported = _infer_verifier_results(
            records=records,
            model_path=model_path,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=1,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            use_chat_template=use_chat_template,
            enable_thinking=enable_thinking,
            system_prompt=system_prompt,
        )
        queue.put(
            {
                "device": device_id,
                "results": results,
                "thinking_arg_unsupported": thinking_arg_unsupported,
            }
        )
    except Exception as exc:
        queue.put({"device": device_id, "error": str(exc)})


def _run_verifier_distributed(
    records: List[TrajectoryRecord],
    gpu_ids: List[str],
    model_path: str,
    dtype: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    use_chat_template: bool,
    enable_thinking: bool,
    system_prompt: Optional[str],
) -> None:
    if not gpu_ids:
        raise ValueError("gpu_ids must not be empty in distributed mode.")

    if len(gpu_ids) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        results, thinking_arg_unsupported = _infer_verifier_results(
            records=records,
            model_path=model_path,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=1,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            use_chat_template=use_chat_template,
            enable_thinking=enable_thinking,
            system_prompt=system_prompt,
        )
        if enable_thinking and thinking_arg_unsupported:
            print(
                "[prompt] Tokenizer chat template does not support enable_thinking; "
                "falling back to default chat template behavior."
            )
        result_map = {idx: (raw, pred) for idx, raw, pred in results}
        for rec in records:
            raw_pred = result_map.get(rec.idx)
            if raw_pred is None:
                continue
            rec.model_raw, rec.model_pred_correct = raw_pred
        return

    mp.set_start_method("spawn", force=True)
    queue = mp.Queue()
    processes: List[mp.Process] = []
    num_shards = len(gpu_ids)

    for shard_id, dev in enumerate(gpu_ids):
        shard_records = records[shard_id::num_shards]
        if not shard_records:
            continue
        p = mp.Process(
            target=_run_verifier_worker,
            args=(
                shard_records,
                dev,
                model_path,
                dtype,
                max_model_len,
                gpu_memory_utilization,
                batch_size,
                max_new_tokens,
                temperature,
                top_p,
                use_chat_template,
                enable_thinking,
                system_prompt,
                queue,
            ),
        )
        p.start()
        processes.append(p)

    result_map: Dict[int, Tuple[str, Optional[bool]]] = {}
    thinking_arg_unsupported = False

    for _ in processes:
        payload = queue.get()
        if "error" in payload:
            for p in processes:
                p.join()
            raise RuntimeError(
                f"[verification] worker on device {payload.get('device')} failed: {payload['error']}"
            )
        thinking_arg_unsupported = thinking_arg_unsupported or bool(
            payload.get("thinking_arg_unsupported")
        )
        for idx, raw, pred in payload["results"]:
            result_map[idx] = (raw, pred)

    for p in processes:
        p.join()

    if enable_thinking and thinking_arg_unsupported:
        print(
            "[prompt] Tokenizer chat template does not support enable_thinking; "
            "falling back to default chat template behavior."
        )

    if len(result_map) != len(records):
        missing = [r.idx for r in records if r.idx not in result_map][:10]
        raise RuntimeError(
            f"[verification] Missing worker outputs for {len(records) - len(result_map)} records. "
            f"Example missing idx: {missing}"
        )

    for rec in records:
        rec.model_raw, rec.model_pred_correct = result_map[rec.idx]


def _run_verifier(
    records: List[TrajectoryRecord],
    model_path: str,
    dtype: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    use_chat_template: bool,
    enable_thinking: bool,
    system_prompt: Optional[str],
) -> None:
    results, thinking_arg_unsupported = _infer_verifier_results(
        records=records,
        model_path=model_path,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        use_chat_template=use_chat_template,
        enable_thinking=enable_thinking,
        system_prompt=system_prompt,
    )

    if enable_thinking and thinking_arg_unsupported:
        print(
            "[prompt] Tokenizer chat template does not support enable_thinking; "
            "falling back to default chat template behavior."
        )

    result_map = {idx: (raw, pred) for idx, raw, pred in results}
    for rec in records:
        raw_pred = result_map.get(rec.idx)
        if raw_pred is None:
            continue
        rec.model_raw, rec.model_pred_correct = raw_pred


def _metrics_from_records(records: List[TrajectoryRecord]) -> Dict[str, Any]:
    total = len(records)
    pred_correct = sum(1 for r in records if r.model_pred_correct is True)
    pred_incorrect = sum(1 for r in records if r.model_pred_correct is False)
    pred_unparsed = sum(1 for r in records if r.model_pred_correct is None)
    actual_correct = sum(1 for r in records if r.actual_correct)
    actual_incorrect = total - actual_correct
    matches = sum(
        1
        for r in records
        if r.model_pred_correct is not None and r.model_pred_correct == r.actual_correct
    )
    parsed_total = total - pred_unparsed

    return {
        "trajectory_total": total,
        "actual_correct_count": actual_correct,
        "actual_incorrect_count": actual_incorrect,
        "model_says_correct_count": pred_correct,
        "model_says_incorrect_count": pred_incorrect,
        "model_unparsed_count": pred_unparsed,
        # strict: treat unparsed as incorrect verification decision
        "verification_accuracy_strict": _safe_div(matches, total),
        # parsed-only: accuracy over trajectories with parseable model verdict
        "verification_accuracy_parsed_only": _safe_div(matches, parsed_total),
        "verification_matches": matches,
        "verification_parsed_total": parsed_total,
    }


def _average_question_accuracy(question_groups: Dict[Tuple[str, ...], List[TrajectoryRecord]]) -> Dict[str, float]:
    strict_scores: List[float] = []
    parsed_scores: List[float] = []
    for recs in question_groups.values():
        total = len(recs)
        parsed_total = sum(1 for r in recs if r.model_pred_correct is not None)
        matches = sum(
            1
            for r in recs
            if r.model_pred_correct is not None and r.model_pred_correct == r.actual_correct
        )
        strict_scores.append(_safe_div(matches, total))
        parsed_scores.append(_safe_div(matches, parsed_total))

    return {
        "avg_question_verification_accuracy_strict": _safe_div(sum(strict_scores), len(strict_scores)),
        "avg_question_verification_accuracy_parsed_only": _safe_div(sum(parsed_scores), len(parsed_scores)),
    }


def _build_report(records: List[TrajectoryRecord], pass_k: int) -> Dict[str, Any]:
    by_question: Dict[Tuple[str, ...], List[TrajectoryRecord]] = {}
    for rec in records:
        by_question.setdefault(rec.question_key, []).append(rec)
    for recs in by_question.values():
        recs.sort(key=lambda x: x.idx)

    pass_hit_questions: Dict[Tuple[str, ...], bool] = {}
    for key, recs in by_question.items():
        pass_hit = any(r.actual_correct for r in recs[:pass_k])
        pass_hit_questions[key] = pass_hit
        for r in recs:
            r.pass_k_hit = pass_hit

    hit_records = [r for r in records if r.pass_k_hit is True]
    miss_records = [r for r in records if r.pass_k_hit is False]

    hit_question_groups = {k: v for k, v in by_question.items() if pass_hit_questions[k]}
    miss_question_groups = {k: v for k, v in by_question.items() if not pass_hit_questions[k]}

    hit_questions = sum(1 for v in pass_hit_questions.values() if v)
    total_questions = len(by_question)
    miss_questions = total_questions - hit_questions

    return {
        "overall": _metrics_from_records(records),
        "pass_at_k_actual": {
            "k": pass_k,
            "question_total": total_questions,
            "question_pass_count": hit_questions,
            "question_fail_count": miss_questions,
            "pass_at_k": _safe_div(hit_questions, total_questions),
        },
        "verification_accuracy_by_pass_at_k_question": {
            "pass_hit_questions": {
                "trajectory_level": _metrics_from_records(hit_records),
                "question_level": _average_question_accuracy(hit_question_groups),
            },
            "pass_miss_questions": {
                "trajectory_level": _metrics_from_records(miss_records),
                "question_level": _average_question_accuracy(miss_question_groups),
            },
        },
    }


def _default_report_path(input_arg_path: Path) -> Path:
    if input_arg_path.suffix:
        return input_arg_path.with_suffix(".verification_report.json")
    return input_arg_path.with_name(f"{input_arg_path.name}.verification_report.json")


def _default_judgments_path(input_arg_path: Path) -> Path:
    if input_arg_path.suffix:
        return input_arg_path.with_suffix(".verification_judgments.jsonl")
    return input_arg_path.with_name(f"{input_arg_path.name}.verification_judgments.jsonl")


def _write_judgments(path: Path, records: List[TrajectoryRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(
                json.dumps(
                    {
                        "idx": rec.idx,
                        "question": rec.question,
                        "answer": rec.answer,
                        "trajectory": rec.trajectory,
                        "actual_correct": rec.actual_correct,
                        "model_pred_correct": rec.model_pred_correct,
                        "model_raw": rec.model_raw,
                        "pass_k_hit_question": rec.pass_k_hit,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="/data1/home/yunhochoi/verl/data/DeepMath-103K/train_1k_Qwen3_8B_trajectories_nothink_4_accuracy.json",
        help=(
            "Trajectory JSONL path, or an accuracy report JSON containing `source_path`. "
            "Input rows should include question/trajectory/answer (answer is used only for metric comparison)."
        ),
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="/data1/home/yunhochoi/verl/data/DeepMath-103K/train_1k_Qwen3_8B_trajectories_nothink_4_think_verification_report.json",
        help="Output JSON report path. Default: <input>.verification_report.json",
    )
    parser.add_argument(
        "--output-judgments",
        type=str,
        default="/data1/home/yunhochoi/verl/data/DeepMath-103K/train_1k_Qwen3_8B_trajectories_nothink_4_think_verification_judgement.jsonl",
        help="JSONL path to save per-trajectory model judgments. Default: <input>.verification_judgments.jsonl",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of trajectories to evaluate.")
    parser.add_argument("--pass-k", type=int, default=4, help="k for pass@k grouping (default: 4).")

    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=10240)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="3",
        help=(
            "Comma-separated GPU ids for data-parallel verification inference "
            "(each GPU loads a full model copy). Example: 0,1,2,3"
        ),
    )
    parser.add_argument(
        "--use-chat-template",
        type=_str_to_bool,
        default=True,
        metavar="{true,false}",
        help="Whether to use tokenizer chat template to build verifier prompts.",
    )
    parser.add_argument(
        "--enable-thinking",
        type=_str_to_bool,
        default=True,
        metavar="{true,false}",
        help="Whether to pass enable_thinking to chat template (when supported).",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=VERIFIER_SYSTEM_PROMPT,
        help=(
            "Verifier system prompt. "
            "If chat template is disabled, this text is prepended inside the user prompt."
        ),
    )

    args = parser.parse_args()

    input_arg_path = Path(args.input).expanduser()
    trajectory_path = _resolve_trajectory_path(input_arg_path)
    if trajectory_path.suffix != ".jsonl":
        raise ValueError(f"Resolved trajectory input must be JSONL. Got: {trajectory_path}")

    report_path = (
        Path(args.output_report).expanduser()
        if args.output_report
        else _default_report_path(input_arg_path)
    )
    judgments_path = (
        Path(args.output_judgments).expanduser()
        if args.output_judgments
        else _default_judgments_path(input_arg_path)
    )

    records, skipped = _load_records(trajectory_path, args.limit)
    if not records:
        raise ValueError(
            "No valid trajectory records found. Check that JSONL contains question/trajectory/answer fields "
            "(answer is only used to compute actual correctness for evaluation)."
        )

    system_prompt = args.system_prompt.strip() if args.system_prompt else None
    gpu_ids = [d.strip() for d in args.gpu_ids.split(",") if d.strip()] if args.gpu_ids else []

    if gpu_ids:
        print(f"[verification] Running data-parallel inference on GPUs: {','.join(gpu_ids)}")
        if args.tensor_parallel_size != 1:
            print(
                "[verification] --gpu-ids mode uses data parallel workers, "
                "forcing tensor_parallel_size=1 per worker."
            )
        _run_verifier_distributed(
            records=records,
            gpu_ids=gpu_ids,
            model_path=args.model_path,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            use_chat_template=args.use_chat_template,
            enable_thinking=args.enable_thinking,
            system_prompt=system_prompt,
        )
    else:
        _run_verifier(
            records=records,
            model_path=args.model_path,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            use_chat_template=args.use_chat_template,
            enable_thinking=args.enable_thinking,
            system_prompt=system_prompt,
        )

    report = _build_report(records, pass_k=args.pass_k)
    report["input_argument_path"] = str(input_arg_path)
    report["resolved_trajectory_path"] = str(trajectory_path)
    report["model_path"] = args.model_path
    report["skipped_input_rows"] = skipped
    report["prompt_config"] = {
        "use_chat_template": args.use_chat_template,
        "enable_thinking": args.enable_thinking,
        "system_prompt": args.system_prompt,
    }
    report["inference_config"] = {
        "gpu_ids": gpu_ids,
        "tensor_parallel_size": args.tensor_parallel_size if not gpu_ids else 1,
        "batch_size": args.batch_size,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    _write_judgments(judgments_path, records)
    print(f"[verification] Saved per-trajectory judgments to {judgments_path}")

    overall = report["overall"]
    pass_stats = report["pass_at_k_actual"]
    print(f"[verification] Loaded trajectories: {len(records)} (skipped rows: {skipped})")
    print(
        "[verification] Overall strict accuracy: "
        f"{overall['verification_accuracy_strict']:.4f} "
        f"({overall['verification_matches']}/{overall['trajectory_total']})"
    )
    print(
        "[verification] Model verdict counts - "
        f"CORRECT: {overall['model_says_correct_count']}, "
        f"INCORRECT: {overall['model_says_incorrect_count']}, "
        f"UNPARSED: {overall['model_unparsed_count']}"
    )
    print(
        f"[verification] pass@{pass_stats['k']} (actual): "
        f"{pass_stats['pass_at_k']:.4f} "
        f"({pass_stats['question_pass_count']}/{pass_stats['question_total']})"
    )
    print(f"[verification] Saved report to {report_path}")


if __name__ == "__main__":
    main()
