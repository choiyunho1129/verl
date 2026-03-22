from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable

sys.dont_write_bytecode = True

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from recipe.gkd.validation_visualizer import build_validation_feedback_record, dump_validation_feedback

try:
    from verl.utils.reward_score.math_verify import compute_score as math_verify_compute_score
except ImportError:
    math_verify_compute_score = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize on-policy distillation on a validation set without training. "
            "The student samples responses, the teacher scores the same sampled tokens, "
            "and the script renders the existing GKD token heatmap HTML."
        )
    )
    parser.add_argument("--student-model", required=True, help="Student model path or HF id.")
    teacher_group = parser.add_mutually_exclusive_group(required=True)
    teacher_group.add_argument("--teacher-model", help="Teacher model path or HF id for local vLLM scoring.")
    teacher_group.add_argument("--teacher-server", help="Teacher server address in HOST:PORT form.")
    parser.add_argument(
        "--val-file",
        default=str(Path(__file__).resolve().parents[2] / "data" / "MATH-500" / "test.parquet"),
        help="Validation parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[2] / "outputs" / "gkd_validation_viz"),
        help="Root directory for JSONL + HTML outputs.",
    )
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--student-max-model-len", type=int, default=None)
    parser.add_argument("--teacher-max-model-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=-1, help="-1 means all validation samples.")
    parser.add_argument(
        "--sample-indices",
        type=str,
        default=None,
        help="Comma-separated dataset indices to visualize after prompt filtering.",
    )
    parser.add_argument("--student-tp-size", type=int, default=1)
    parser.add_argument("--teacher-tp-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--student-gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--teacher-gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--teacher-server-workers", type=int, default=1)
    parser.add_argument(
        "--teacher-preview-max-tokens",
        type=int,
        default=256,
        help="Optional free-generation preview from the teacher. Use 0 to disable.",
    )
    parser.add_argument("--teacher-preview-temperature", type=float, default=0.6)
    parser.add_argument(
        "--hf-cache-dir",
        default=None,
        help="Writable Hugging Face cache root. Defaults to ~/.cache/huggingface for this script.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metric", default="advantage", choices=["advantage", "reverse_kl", "student_logprobs", "teacher_logprobs"])
    parser.add_argument("--select", default="first", choices=["first", "longest", "best_score", "worst_score", "all"])
    parser.add_argument("--render-limit", type=int, default=24)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--skip-score", action="store_true", help="Disable math_verify scoring even if ground truth exists.")
    parser.add_argument("--filter-overlong-prompts", dest="filter_overlong_prompts", action="store_true", default=True)
    parser.add_argument("--no-filter-overlong-prompts", dest="filter_overlong_prompts", action="store_false")
    parser.add_argument("--truncation", default="error", choices=["left", "right", "middle", "error"])
    return parser.parse_args()


def batched(items: list[dict], batch_size: int) -> Iterable[list[dict]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def parse_teacher_server(server: str) -> tuple[str, int]:
    host, port = server.rsplit(":", 1)
    return host, int(port)


def configure_hf_cache(cache_dir: str | None) -> Path:
    cache_root = Path(cache_dir).expanduser() if cache_dir else Path.home() / ".cache" / "huggingface"
    cache_root.mkdir(parents=True, exist_ok=True)

    hub_dir = cache_root / "hub"
    hub_dir.mkdir(parents=True, exist_ok=True)
    transformers_dir = cache_root / "transformers"
    transformers_dir.mkdir(parents=True, exist_ok=True)
    vllm_dir = cache_root / "vllm"
    vllm_dir.mkdir(parents=True, exist_ok=True)
    vllm_cache_root = cache_root / "vllm_cache_root"
    vllm_cache_root.mkdir(parents=True, exist_ok=True)
    torchinductor_dir = cache_root / "torchinductor"
    torchinductor_dir.mkdir(parents=True, exist_ok=True)
    triton_dir = cache_root / "triton"
    triton_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = cache_root / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_HUB_CACHE"] = str(hub_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(transformers_dir)
    os.environ["VLLM_DOWNLOAD_DIR"] = str(vllm_dir)
    os.environ["VLLM_CACHE_ROOT"] = str(vllm_cache_root)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(torchinductor_dir)
    os.environ["TRITON_CACHE_DIR"] = str(triton_dir)
    os.environ["TMPDIR"] = str(tmp_dir)
    tempfile.tempdir = str(tmp_dir)
    return cache_root


def configure_vllm_runtime() -> None:
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # Start method may already be fixed by the parent process.
        pass


def resolve_model_reference(model_ref: str, cache_dir: str | None) -> str:
    model_path = Path(model_ref).expanduser()
    if model_path.exists():
        return str(model_path)

    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=model_ref,
        cache_dir=cache_dir,
    )


def align_teacher_response_tensors(
    teacher_topk_logps: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    response_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if response_length == 0:
        return (
            torch.empty((0, 0), dtype=torch.float32),
            torch.empty((0, 0), dtype=torch.int32),
        )
    if teacher_topk_logps.ndim != 2 or teacher_topk_logps.size(-1) < 1:
        raise ValueError(f"Unexpected teacher_topk_logps shape: {tuple(teacher_topk_logps.shape)}")
    if teacher_topk_indices.ndim != 2 or teacher_topk_indices.shape != teacher_topk_logps.shape:
        raise ValueError(
            "Teacher top-k index tensor must match logprob tensor shape: "
            f"{tuple(teacher_topk_indices.shape)} vs {tuple(teacher_topk_logps.shape)}"
        )

    if teacher_topk_logps.size(0) >= response_length + 1:
        row_slice = slice(-response_length - 1, -1)
    elif teacher_topk_logps.size(0) >= response_length:
        row_slice = slice(-response_length, None)
    else:
        raise ValueError(
            "Teacher prompt logprobs are shorter than the student response length: "
            f"{teacher_topk_logps.size(0)} < {response_length}"
        )

    return (
        teacher_topk_logps[row_slice].to(torch.float32),
        teacher_topk_indices[row_slice].to(torch.int32),
    )


def extract_teacher_response_details(
    teacher_topk_logps: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    response_token_ids: list[int],
) -> tuple[list[float], list[int], list[float]]:
    aligned_logps, aligned_indices = align_teacher_response_tensors(
        teacher_topk_logps,
        teacher_topk_indices,
        response_length=len(response_token_ids),
    )
    if aligned_logps.numel() == 0:
        return [], [], []

    sampled_ids = aligned_indices[:, 0].tolist()
    if sampled_ids != response_token_ids:
        raise ValueError("Teacher sampled token ids are not aligned with the student response token ids.")

    row_ids = torch.arange(aligned_logps.size(0), device=aligned_logps.device)
    best_cols = torch.argmax(aligned_logps, dim=1)
    teacher_top1_ids = aligned_indices[row_ids, best_cols].tolist()
    teacher_top1_logprobs = aligned_logps[row_ids, best_cols].tolist()
    teacher_sampled_logprobs = aligned_logps[:, 0].tolist()

    return teacher_sampled_logprobs, teacher_top1_ids, teacher_top1_logprobs


def maybe_score_response(response_text: str, ground_truth: str | None, skip_score: bool) -> float | None:
    if skip_score or not ground_truth or math_verify_compute_score is None:
        return None
    try:
        return float(math_verify_compute_score(response_text, ground_truth))
    except Exception:
        return None


def get_ground_truth(sample: dict) -> str | None:
    reward_model = sample.get("reward_model")
    if isinstance(reward_model, dict):
        ground_truth = reward_model.get("ground_truth")
        if ground_truth not in (None, ""):
            return str(ground_truth)
    extra_info = sample.get("extra_info")
    if isinstance(extra_info, dict):
        answer = extra_info.get("answer")
        if answer not in (None, ""):
            return str(answer)
    return None


def infer_student_max_model_len(args: argparse.Namespace) -> int:
    if args.student_max_model_len is not None:
        return int(args.student_max_model_len)
    return int(args.max_prompt_length + args.max_new_tokens)


def infer_teacher_max_model_len(args: argparse.Namespace) -> int:
    if args.teacher_max_model_len is not None:
        return int(args.teacher_max_model_len)
    preview_budget = max(0, int(args.teacher_preview_max_tokens))
    # Teacher needs room for:
    # 1) preview generation from prompt only, and
    # 2) scoring the student's sampled trajectory with one extra sampled step.
    return int(args.max_prompt_length + max(preview_budget, args.max_new_tokens + 1))


def build_dataset(tokenizer, args: argparse.Namespace) -> RLHFDataset:
    from omegaconf import OmegaConf
    from verl.utils.dataset.rl_dataset import RLHFDataset

    dataset_cfg = OmegaConf.create(
        {
            "prompt_key": args.prompt_key,
            "max_prompt_length": args.max_prompt_length,
            "return_raw_chat": True,
            "return_full_prompt": True,
            "truncation": args.truncation,
            "filter_overlong_prompts": args.filter_overlong_prompts,
            "filter_overlong_prompts_workers": 1,
            "use_chat_template": args.use_chat_template,
            "shuffle": False,
        }
    )
    return RLHFDataset(data_files=args.val_file, tokenizer=tokenizer, config=dataset_cfg)


def select_samples(dataset: RLHFDataset, args: argparse.Namespace) -> tuple[list[int], list[dict]]:
    if args.sample_indices:
        indices = [int(token.strip()) for token in args.sample_indices.split(",") if token.strip()]
    else:
        total = len(dataset) if args.num_samples < 0 else min(len(dataset), args.num_samples)
        indices = list(range(total))
    samples = [dataset[index] for index in indices]
    return indices, samples


def ensure_local_tokenizer_compatibility(
    student_model: str,
    teacher_model: str,
    trust_remote_code: bool,
    cache_dir: str | None,
) -> None:
    from transformers import AutoTokenizer

    probe = "User: Find x.\nAssistant:"
    student_tokenizer = AutoTokenizer.from_pretrained(
        student_model,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
    )
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        teacher_model,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
    )
    student_ids = student_tokenizer.encode(probe, add_special_tokens=False)
    teacher_ids = teacher_tokenizer.encode(probe, add_special_tokens=False)
    if student_ids != teacher_ids:
        raise ValueError(
            "Teacher and student tokenizers are not aligned. "
            "On-policy distillation visualization needs the teacher to score the student's sampled token ids."
        )


def make_student_engine(args: argparse.Namespace) -> VLLMEngine:
    from recipe.gkd.teacher.vllm_engine import VLLMEngine

    return VLLMEngine(
        args.student_model,
        n_logprobs=1,
        tp_size=args.student_tp_size,
        gpu_memory_utilization=args.student_gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
        seed=args.seed,
        download_dir=args.hf_cache_dir,
        max_model_len=infer_student_max_model_len(args),
    )


def make_local_teacher_engine(args: argparse.Namespace) -> VLLMEngine:
    from recipe.gkd.teacher.vllm_engine import VLLMEngine

    return VLLMEngine(
        args.teacher_model,
        n_logprobs=1,
        tp_size=args.teacher_tp_size,
        gpu_memory_utilization=args.teacher_gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
        seed=args.seed,
        download_dir=args.hf_cache_dir,
        max_model_len=infer_teacher_max_model_len(args),
    )


def decode_response(tokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def run() -> None:
    from tqdm import tqdm
    from transformers import AutoTokenizer

    args = parse_args()
    cache_root = configure_hf_cache(args.hf_cache_dir)
    args.hf_cache_dir = str(cache_root)
    configure_vllm_runtime()
    args.student_model = resolve_model_reference(args.student_model, args.hf_cache_dir)
    if args.teacher_model:
        args.teacher_model = resolve_model_reference(args.teacher_model, args.hf_cache_dir)

    if args.teacher_model:
        ensure_local_tokenizer_compatibility(
            args.student_model,
            args.teacher_model,
            args.trust_remote_code,
            args.hf_cache_dir,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.student_model,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.hf_cache_dir,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = build_dataset(tokenizer, args)
    sample_indices, samples = select_samples(dataset, args)
    if not samples:
        raise ValueError("No validation samples selected.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    student_engine = make_student_engine(args)
    local_teacher_engine = make_local_teacher_engine(args) if args.teacher_model else None

    teacher_score_client = None
    teacher_preview_client = None
    if args.teacher_server:
        from recipe.gkd.teacher.client import TeacherClient

        host, port = parse_teacher_server(args.teacher_server)
        teacher_score_client = TeacherClient(
            server_ip=host,
            server_port=port,
            n_server_workers=args.teacher_server_workers,
            max_tokens=1,
            temperature=1.0,
            only_response=False,
            use_sampled_token_logprobs=True,
            return_full_logprobs=True,
        )
        if args.teacher_preview_max_tokens > 0:
            teacher_preview_client = TeacherClient(
                server_ip=host,
                server_port=port,
                n_server_workers=args.teacher_server_workers,
                max_tokens=args.teacher_preview_max_tokens,
                temperature=args.teacher_preview_temperature,
                only_response=True,
                use_sampled_token_logprobs=False,
            )

    records = []
    manifest = {
        "student_model": args.student_model,
        "teacher_model": args.teacher_model,
        "teacher_server": args.teacher_server,
        "val_file": args.val_file,
        "num_samples": len(samples),
        "sample_indices": sample_indices,
        "max_prompt_length": args.max_prompt_length,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    with open(output_dir / "run_config.json", "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)

    progress = tqdm(total=len(samples), desc="Building GKD heatmaps")
    global_index = 0
    for batch_samples in batched(samples, args.batch_size):
        prompt_ids_batch = [sample["raw_prompt_ids"] for sample in batch_samples]
        prompt_text_batch = [sample.get("full_prompts") or decode_response(tokenizer, sample["raw_prompt_ids"]) for sample in batch_samples]

        responses, student_topk_logps, _ = student_engine.get_topk_logprobs(
            prompt_ids_batch,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            only_response=True,
            use_sampled_token_logprobs=True,
        )

        full_input_ids_batch = []
        for prompt_ids, response in zip(prompt_ids_batch, responses, strict=False):
            full_input_ids_batch.append(prompt_ids + response.tolist())

        if local_teacher_engine is not None:
            _, teacher_topk_logps, teacher_topk_indices = local_teacher_engine.get_topk_logprobs(
                full_input_ids_batch,
                temperature=1.0,
                top_p=args.top_p,
                max_new_tokens=1,
                only_response=False,
                use_sampled_token_logprobs=True,
                return_full_logprobs=True,
            )
        else:
            teacher_fut = teacher_score_client.submit(full_input_ids_batch)
            _, teacher_topk_logps, teacher_topk_indices = teacher_fut.result()

        teacher_preview_texts = [None] * len(batch_samples)
        if args.teacher_preview_max_tokens > 0:
            if local_teacher_engine is not None:
                teacher_preview_responses, _, _ = local_teacher_engine.get_topk_logprobs(
                    prompt_ids_batch,
                    temperature=args.teacher_preview_temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.teacher_preview_max_tokens,
                    only_response=True,
                    use_sampled_token_logprobs=False,
                )
            else:
                preview_fut = teacher_preview_client.submit(prompt_ids_batch)
                teacher_preview_responses, _, _ = preview_fut.result()
            teacher_preview_texts = [
                decode_response(tokenizer, response.tolist()) for response in teacher_preview_responses
            ]

        for sample, prompt_text, response, student_logps, teacher_logps, teacher_indices, teacher_preview_text in zip(
            batch_samples,
            prompt_text_batch,
            responses,
            student_topk_logps,
            teacher_topk_logps,
            teacher_topk_indices,
            teacher_preview_texts,
            strict=False,
        ):
            response_token_ids = response.tolist()
            student_sampled_logprobs = student_logps[:, 0].to(torch.float32).tolist() if student_logps.numel() else []
            teacher_response_logprobs, teacher_top1_token_ids, teacher_top1_logprobs = extract_teacher_response_details(
                teacher_logps,
                teacher_indices,
                response_token_ids,
            )

            if len(student_sampled_logprobs) != len(teacher_response_logprobs):
                raise ValueError(
                    "Teacher/student response logprob length mismatch: "
                    f"{len(student_sampled_logprobs)} vs {len(teacher_response_logprobs)}"
                )

            student_text = decode_response(tokenizer, response_token_ids)
            ground_truth = get_ground_truth(sample)
            student_score = maybe_score_response(student_text, ground_truth, args.skip_score)
            teacher_preview_score = maybe_score_response(teacher_preview_text or "", ground_truth, args.skip_score)

            extra_info = sample.get("extra_info") if isinstance(sample.get("extra_info"), dict) else {}
            extra = {
                "ground_truth": ground_truth,
                "subject": extra_info.get("subject"),
                "ability": sample.get("ability"),
                "level": extra_info.get("level"),
                "student_score": student_score,
                "teacher_preview_score": teacher_preview_score,
                "teacher_preview_text": teacher_preview_text,
                "student_mean_logprob": (
                    float(sum(student_sampled_logprobs) / len(student_sampled_logprobs))
                    if student_sampled_logprobs
                    else None
                ),
                "teacher_mean_logprob_on_student": (
                    float(sum(teacher_response_logprobs) / len(teacher_response_logprobs))
                    if teacher_response_logprobs
                    else None
                ),
            }

            score = student_score
            if score is None and student_sampled_logprobs:
                reverse_kl = [student - teacher for student, teacher in zip(student_sampled_logprobs, teacher_response_logprobs, strict=False)]
                score = float(sum(reverse_kl) / len(reverse_kl))

            records.append(
                build_validation_feedback_record(
                    tokenizer=tokenizer,
                    prompt_text=str(prompt_text),
                    response_token_ids=response_token_ids,
                    student_logprobs=student_sampled_logprobs,
                    teacher_logprobs=teacher_response_logprobs,
                    teacher_top1_token_ids=teacher_top1_token_ids,
                    teacher_top1_logprobs=teacher_top1_logprobs,
                    score=score,
                    sample_index=global_index,
                    uid=str(sample.get("uid")) if sample.get("uid") is not None else None,
                    data_source=str(sample.get("data_source")) if sample.get("data_source") is not None else None,
                    extra=extra,
                )
            )
            global_index += 1
            progress.update(1)

    progress.close()

    dump_validation_feedback(
        dump_root=str(output_dir),
        step=0,
        records=records,
        metric=args.metric,
        select=args.select,
        limit=args.render_limit,
    )

    print(f"Wrote GKD visualization to {output_dir / 'token_feedback' / 'step_0' / 'index.html'}")


if __name__ == "__main__":
    run()
