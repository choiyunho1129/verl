from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from classifer_training.extract_hidden_states import (
    _extract_messages,
    _get_transformer_blocks,
    _infer_input_device,
    _render_prompt,
    _resolve_torch_dtype,
)
from classifer_training.rollout_utils import (
    extract_response_char_spans,
    extract_rollout_numeric_features,
    select_response_char_span,
)
from classifer_training.utils import coerce_float, load_records, parse_layer_spec, sanitize_name, write_jsonl

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN_ROOT = REPO_ROOT / "classifer_training" / "artifacts" / "runs"
DEFAULT_HIDDEN_ROOT = REPO_ROOT / "classifer_training" / "artifacts" / "rollout_hidden"
DEFAULT_INDEX_ROOT = REPO_ROOT / "classifer_training" / "artifacts" / "rollout_index"


def _append_shard_suffix(filename: str, shard_index: int, num_shards: int) -> str:
    if num_shards <= 1:
        return filename
    path = Path(filename)
    suffix = f".shard{shard_index:02d}of{num_shards:02d}"
    return f"{path.stem}{suffix}{path.suffix}"


def _checkpoint_dir_for_output(hidden_output_path: Path) -> Path:
    return hidden_output_path.parent / f"{hidden_output_path.name}.partial"


def _checkpoint_metadata_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "metadata.json"


def _checkpoint_chunk_path(checkpoint_dir: Path, chunk_index: int) -> Path:
    return checkpoint_dir / f"chunk_{chunk_index:06d}.pt"


def _write_checkpoint_metadata(
    checkpoint_dir: Path,
    *,
    processed_examples: int,
    total_examples: int,
    next_chunk_index: int,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "processed_examples": int(processed_examples),
        "total_examples": int(total_examples),
        "next_chunk_index": int(next_chunk_index),
    }
    _checkpoint_metadata_path(checkpoint_dir).write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _load_checkpoint_chunks(
    checkpoint_dir: Path,
    *,
    drop_incomplete_last_chunk: bool = True,
) -> list[tuple[Path, dict[str, Any]]]:
    chunk_paths = sorted(checkpoint_dir.glob("chunk_*.pt"))
    loaded_chunks: list[tuple[Path, dict[str, Any]]] = []
    for chunk_idx, chunk_path in enumerate(chunk_paths):
        try:
            payload = torch.load(chunk_path, map_location="cpu")
        except Exception:
            is_last_chunk = chunk_idx == len(chunk_paths) - 1
            if drop_incomplete_last_chunk and is_last_chunk:
                chunk_path.unlink(missing_ok=True)
                break
            raise
        loaded_chunks.append((chunk_path, payload))
    return loaded_chunks


def _resume_state_from_checkpoint(checkpoint_dir: Path) -> tuple[int, int]:
    if not checkpoint_dir.exists():
        return 0, 0

    loaded_chunks = _load_checkpoint_chunks(checkpoint_dir)
    processed_examples = 0
    next_chunk_index = 0
    for chunk_path, payload in loaded_chunks:
        hidden_examples = payload.get("hidden_examples")
        index_records = payload.get("index_records")
        if not isinstance(hidden_examples, list) or not isinstance(index_records, list):
            raise ValueError(f"Invalid checkpoint chunk format: {chunk_path}")
        if len(hidden_examples) != len(index_records):
            raise ValueError(f"Mismatched checkpoint chunk sizes: {chunk_path}")
        processed_examples += len(hidden_examples)
        next_chunk_index += 1
    return processed_examples, next_chunk_index


def _finalize_from_checkpoint(
    checkpoint_dir: Path,
    *,
    hidden_output_path: Path,
    index_output_path: Path,
    metadata: dict[str, Any],
) -> None:
    hidden_examples: list[dict[str, Any]] = []
    index_records: list[dict[str, Any]] = []
    for _chunk_path, payload in _load_checkpoint_chunks(checkpoint_dir, drop_incomplete_last_chunk=False):
        hidden_examples.extend(payload["hidden_examples"])
        index_records.extend(payload["index_records"])

    hidden_output_path.parent.mkdir(parents=True, exist_ok=True)
    index_output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": {
                **metadata,
                "num_examples": len(hidden_examples),
            },
            "examples": hidden_examples,
        },
        hidden_output_path,
    )
    write_jsonl(index_output_path, index_records)
    shutil.rmtree(checkpoint_dir, ignore_errors=True)


def _extract_batched_token_vectors(output: Any, token_indices: list[int]) -> torch.Tensor:
    if isinstance(output, tuple):
        output = output[0]
    if isinstance(output, dict):
        output = output.get("hidden_states", output.get("last_hidden_state", output))
    if not torch.is_tensor(output):
        raise TypeError(f"Unsupported hook output type: {type(output)!r}")

    tensor = output
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"Expected a 3D tensor, got shape {tuple(tensor.shape)}.")

    if len(token_indices) != tensor.shape[0]:
        raise ValueError(
            f"Expected {tensor.shape[0]} token indices for the batch, got {len(token_indices)}."
        )

    positions = torch.as_tensor(token_indices, device=tensor.device, dtype=torch.long)
    batch_indices = torch.arange(tensor.shape[0], device=tensor.device)
    return tensor[batch_indices, positions, :].detach().to(dtype=torch.float32, device="cpu")


def _extract_batched_group_pooled_vectors(
    output: Any,
    token_groups: list[list[int]],
    fallback_indices: list[int],
) -> torch.Tensor:
    if isinstance(output, tuple):
        output = output[0]
    if isinstance(output, dict):
        output = output.get("hidden_states", output.get("last_hidden_state", output))
    if not torch.is_tensor(output):
        raise TypeError(f"Unsupported hook output type: {type(output)!r}")

    tensor = output
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"Expected a 3D tensor, got shape {tuple(tensor.shape)}.")
    if len(token_groups) != tensor.shape[0] or len(fallback_indices) != tensor.shape[0]:
        raise ValueError("Batch size mismatch while pooling token groups.")

    pooled_rows: list[torch.Tensor] = []
    seq_len = tensor.shape[1]
    for batch_idx, token_ids in enumerate(token_groups):
        valid_ids = [int(idx) for idx in token_ids if 0 <= int(idx) < seq_len]
        if valid_ids:
            pooled = tensor[batch_idx, valid_ids, :].mean(dim=0)
        else:
            fallback = int(fallback_indices[batch_idx])
            fallback = min(max(fallback, 0), seq_len - 1)
            pooled = tensor[batch_idx, fallback, :]
        pooled_rows.append(pooled.detach().to(dtype=torch.float32, device="cpu"))
    return torch.stack(pooled_rows, dim=0)


def _extract_batched_window_vectors(
    output: Any,
    token_windows: list[list[int]],
    fallback_indices: list[int],
    *,
    window_size: int,
) -> torch.Tensor:
    if isinstance(output, tuple):
        output = output[0]
    if isinstance(output, dict):
        output = output.get("hidden_states", output.get("last_hidden_state", output))
    if not torch.is_tensor(output):
        raise TypeError(f"Unsupported hook output type: {type(output)!r}")

    tensor = output
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"Expected a 3D tensor, got shape {tuple(tensor.shape)}.")
    if len(token_windows) != tensor.shape[0] or len(fallback_indices) != tensor.shape[0]:
        raise ValueError("Batch size mismatch while extracting token windows.")

    window_rows: list[torch.Tensor] = []
    seq_len = tensor.shape[1]
    hidden_dim = tensor.shape[2]
    for batch_idx, token_ids in enumerate(token_windows):
        valid_ids = [int(idx) for idx in token_ids if 0 <= int(idx) < seq_len]
        if not valid_ids:
            fallback = int(fallback_indices[batch_idx])
            fallback = min(max(fallback, 0), seq_len - 1)
            valid_ids = [fallback]
        valid_ids = valid_ids[-window_size:]
        gathered = tensor[batch_idx, valid_ids, :].detach().to(dtype=torch.float32, device="cpu")
        if gathered.ndim == 1:
            gathered = gathered.unsqueeze(0)
        if gathered.shape[0] < window_size:
            pad = torch.zeros((window_size - gathered.shape[0], hidden_dim), dtype=torch.float32)
            gathered = torch.cat([pad, gathered], dim=0)
        window_rows.append(gathered.contiguous().clone())
    return torch.stack(window_rows, dim=0)


def _extract_token_vector(output: Any, token_index: int) -> torch.Tensor:
    if isinstance(output, tuple):
        output = output[0]
    if isinstance(output, dict):
        output = output.get("hidden_states", output.get("last_hidden_state", output))
    if not torch.is_tensor(output):
        raise TypeError(f"Unsupported hook output type: {type(output)!r}")

    tensor = output.detach().cpu().to(torch.float32)
    if tensor.ndim == 3:
        sequence_length = tensor.shape[1]
        if not 0 <= token_index < sequence_length:
            raise ValueError(f"Token index {token_index} is out of range for sequence length {sequence_length}.")
        return tensor[:, token_index, :].squeeze(0).contiguous().clone()
    if tensor.ndim == 2:
        if not 0 <= token_index < tensor.shape[0]:
            raise ValueError(f"Token index {token_index} is out of range for tensor shape {tuple(tensor.shape)}.")
        return tensor[token_index, :].contiguous().clone()
    if tensor.ndim == 1 and token_index == 0:
        return tensor.contiguous().clone()
    raise ValueError(f"Expected 2D or 3D tensor output, got shape {tuple(tensor.shape)}.")


def _last_token_index_for_span(
    offset_mapping: list[tuple[int, int]],
    start_char: int,
    end_char: int,
) -> int | None:
    last_idx: int | None = None
    for token_idx, (token_start, token_end) in enumerate(offset_mapping):
        if token_end <= token_start:
            continue
        if token_start < end_char and token_end > start_char:
            last_idx = token_idx
    return last_idx


def _last_token_index_before_char(offset_mapping: list[tuple[int, int]], end_char: int) -> int | None:
    last_idx: int | None = None
    for token_idx, (token_start, token_end) in enumerate(offset_mapping):
        if token_end <= token_start:
            continue
        if token_end <= end_char:
            last_idx = token_idx
    return last_idx


def _last_generated_token_index(offset_mapping: list[tuple[int, int]], prompt_char_count: int) -> int | None:
    last_idx: int | None = None
    for token_idx, (token_start, token_end) in enumerate(offset_mapping):
        if token_end <= token_start:
            continue
        if token_end > prompt_char_count:
            last_idx = token_idx
    return last_idx


def _token_indices_for_span(
    offset_mapping: list[tuple[int, int]],
    start_char: int,
    end_char: int,
) -> list[int]:
    indices: list[int] = []
    for token_idx, (token_start, token_end) in enumerate(offset_mapping):
        if token_end <= token_start:
            continue
        if token_start < end_char and token_end > start_char:
            indices.append(token_idx)
    return indices


def _token_indices_after_char(offset_mapping: list[tuple[int, int]], prompt_char_count: int) -> list[int]:
    indices: list[int] = []
    for token_idx, (token_start, token_end) in enumerate(offset_mapping):
        if token_end <= token_start:
            continue
        if token_end > prompt_char_count:
            indices.append(token_idx)
    return indices


def _to_offset_mapping(tokenized: dict[str, Any]) -> list[tuple[int, int]] | None:
    offset_mapping = tokenized.get("offset_mapping")
    if offset_mapping is None:
        return None
    if torch.is_tensor(offset_mapping):
        values = offset_mapping[0].detach().cpu().tolist()
    else:
        values = offset_mapping[0]
    return [tuple(int(value) for value in pair) for pair in values]


def _resolve_generated_token_indices(
    tokenizer,
    generated_text: str,
    response_anchor: str,
) -> tuple[dict[str, int], str, dict[str, int], dict[str, Any]]:
    try:
        tokenized = tokenizer(
            generated_text,
            return_tensors="pt",
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offset_mapping = _to_offset_mapping(tokenized)
    except Exception:
        tokenized = None
        offset_mapping = None
    if offset_mapping is None:
        fallback = tokenizer(generated_text, return_tensors="pt", add_special_tokens=False)
        sequence_length = int(fallback["input_ids"].shape[1])
        last_generated = max(sequence_length - 1, 0)
        return (
            {
                "answer_hidden": last_generated,
                "reasoning_hidden": last_generated,
                "think_end_hidden": last_generated,
                "response_hidden": last_generated,
            },
            "last_generated",
            {
                "last_generated": last_generated,
                "answer_last_token": last_generated,
                "reasoning_last_token": last_generated,
                "think_end_last_token": last_generated,
                "response_last_token": last_generated,
            },
            fallback,
        )
    local_spans = extract_response_char_spans(generated_text)
    answer_kind, local_response_span = select_response_char_span(generated_text, response_anchor)

    sequence_length = int(tokenized["input_ids"].shape[1])
    last_generated = max(sequence_length - 1, 0)

    reasoning_last_token = (
        _last_token_index_for_span(offset_mapping, local_spans["reasoning"][0], local_spans["reasoning"][1])
        if local_spans["reasoning"] is not None
        else None
    )
    think_end_last_token = (
        _last_token_index_for_span(offset_mapping, local_spans["think_end_tag"][0], local_spans["think_end_tag"][1])
        if local_spans.get("think_end_tag") is not None
        else None
    )
    answer_last_token = (
        _last_token_index_for_span(offset_mapping, local_spans["answer"][0], local_spans["answer"][1])
        if local_spans["answer"] is not None
        else None
    )
    response_last_token = (
        _last_token_index_for_span(offset_mapping, local_response_span[0], local_response_span[1])
        if local_response_span is not None
        else None
    )
    if response_last_token is None:
        response_last_token = last_generated

    token_positions = {
        "answer_hidden": answer_last_token if answer_last_token is not None else response_last_token,
        "reasoning_hidden": reasoning_last_token if reasoning_last_token is not None else response_last_token,
        "think_end_hidden": think_end_last_token if think_end_last_token is not None else (
            reasoning_last_token if reasoning_last_token is not None else response_last_token
        ),
        "response_hidden": response_last_token,
    }
    metadata = {
        "last_generated": last_generated,
        "answer_last_token": answer_last_token if answer_last_token is not None else response_last_token,
        "reasoning_last_token": reasoning_last_token if reasoning_last_token is not None else response_last_token,
        "think_end_last_token": think_end_last_token if think_end_last_token is not None else (
            reasoning_last_token if reasoning_last_token is not None else response_last_token
        ),
        "response_last_token": response_last_token,
    }

    tokenized.pop("offset_mapping", None)
    return token_positions, answer_kind, metadata, tokenized


def _resolve_run_dirs(run_dirs: list[str], run_glob: str | None) -> list[Path]:
    resolved = [Path(path).expanduser().resolve() for path in run_dirs]
    if run_glob:
        resolved.extend(sorted(Path().glob(run_glob)))
    deduped = sorted({path.resolve() for path in resolved})
    if not deduped:
        raise ValueError("At least one run directory is required.")
    return deduped


def _estimate_total_tokens(record: dict[str, Any]) -> int:
    input_length = coerce_float(record.get("input_length"))
    output_length = coerce_float(record.get("output_length"))
    if input_length is not None or output_length is not None:
        return int(max((input_length or 0.0) + (output_length or 0.0), 1.0))
    user_input = str(record.get("user_input", ""))
    generated_text = str(record.get("generated_text", ""))
    return max(len(user_input) + len(generated_text), 1)


def _iter_dynamic_batches(
    prepared_records: list[dict[str, Any]],
    *,
    batch_size: int,
    max_batch_tokens: int,
):
    current_batch: list[dict[str, Any]] = []
    current_max_tokens = 0

    for prepared in prepared_records:
        estimated_tokens = int(prepared["estimated_total_tokens"])
        proposed_max_tokens = max(current_max_tokens, estimated_tokens)
        proposed_batch_size = len(current_batch) + 1
        proposed_total_tokens = proposed_max_tokens * proposed_batch_size

        should_flush = bool(current_batch) and (
            proposed_batch_size > batch_size
            or (max_batch_tokens > 0 and proposed_total_tokens > max_batch_tokens)
        )
        if should_flush:
            yield current_batch
            current_batch = []
            current_max_tokens = 0

        current_batch.append(prepared)
        current_max_tokens = max(current_max_tokens, estimated_tokens)

    if current_batch:
        yield current_batch


def _compute_token_level_confidence_features(
    *,
    logits_row: torch.Tensor,
    input_ids_row: torch.Tensor,
    token_indices: list[int],
    prefix: str,
) -> dict[str, float]:
    usable_indices = [token_index for token_index in token_indices if token_index > 0]
    if not usable_indices:
        return {}

    predictor_positions = torch.as_tensor(
        [token_index - 1 for token_index in usable_indices],
        device=logits_row.device,
        dtype=torch.long,
    )
    target_positions = torch.as_tensor(usable_indices, device=input_ids_row.device, dtype=torch.long)
    selected_logits = logits_row.index_select(0, predictor_positions)
    target_ids = input_ids_row.index_select(0, target_positions)

    log_probs = torch.log_softmax(selected_logits, dim=-1)
    probs = torch.exp(log_probs)
    token_entropies = (-(probs * log_probs)).sum(dim=-1)
    token_log_probs = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)

    top2 = torch.topk(selected_logits[-1], k=2).values
    margin = float((top2[0] - top2[1]).item()) if top2.numel() >= 2 else 0.0

    return {
        f"{prefix}_mean_logprob": float(token_log_probs.mean().item()),
        f"{prefix}_min_logprob": float(token_log_probs.min().item()),
        f"{prefix}_last_token_logprob": float(token_log_probs[-1].item()),
        f"{prefix}_mean_token_entropy": float(token_entropies.mean().item()),
        f"{prefix}_min_token_entropy": float(token_entropies.min().item()),
        f"{prefix}_max_token_entropy": float(token_entropies.max().item()),
        f"{prefix}_last_token_entropy": float(token_entropies[-1].item()),
        f"{prefix}_last_token_margin": margin,
    }


def _compute_batch_confidence_features(
    *,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    metadata_rows: list[dict[str, Any]],
) -> list[dict[str, float]]:
    confidence_rows: list[dict[str, float]] = []
    for batch_idx, metadata in enumerate(metadata_rows):
        logits_row = logits[batch_idx]
        input_ids_row = input_ids[batch_idx]
        token_groups = metadata.get("token_groups", {})
        features: dict[str, float] = {}
        features.update(
            _compute_token_level_confidence_features(
                logits_row=logits_row,
                input_ids_row=input_ids_row,
                token_indices=list(token_groups.get("output", [])),
                prefix="output",
            )
        )
        features.update(
            _compute_token_level_confidence_features(
                logits_row=logits_row,
                input_ids_row=input_ids_row,
                token_indices=list(token_groups.get("reasoning", [])),
                prefix="reasoning",
            )
        )
        features.update(
            _compute_token_level_confidence_features(
                logits_row=logits_row,
                input_ids_row=input_ids_row,
                token_indices=list(token_groups.get("answer", [])),
                prefix="answer",
            )
        )
        confidence_rows.append(features)
    return confidence_rows


def _resolve_batch_token_metadata(
    *,
    tokenizer,
    full_texts: list[str],
    prompt_texts: list[str],
    generated_texts: list[str],
    response_anchor: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tokenized = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        return_offsets_mapping=True,
    )
    offset_mapping = tokenized.pop("offset_mapping")
    attention_mask = tokenized.get("attention_mask")

    batch_component_positions: dict[str, list[int]] = {}
    metadata_rows: list[dict[str, Any]] = []

    for batch_idx, (prompt_text, generated_text) in enumerate(zip(prompt_texts, generated_texts)):
        if torch.is_tensor(offset_mapping):
            offsets = offset_mapping[batch_idx].detach().cpu().tolist()
        else:
            offsets = offset_mapping[batch_idx]

        if attention_mask is not None:
            valid_length = int(attention_mask[batch_idx].detach().cpu().sum().item())
            offsets = offsets[:valid_length]
        normalized_offsets = [tuple(int(value) for value in pair) for pair in offsets]

        prompt_char_count = len(prompt_text)
        prompt_last_token = _last_token_index_before_char(normalized_offsets, prompt_char_count)
        if prompt_last_token is None:
            prompt_last_token = max(len(normalized_offsets) - 1, 0)

        local_spans = extract_response_char_spans(generated_text)
        answer_kind, local_response_span = select_response_char_span(generated_text, response_anchor)
        last_generated = _last_generated_token_index(normalized_offsets, prompt_char_count)

        if last_generated is None:
            last_generated = prompt_last_token

        def shift_span(span: tuple[int, int] | None) -> tuple[int, int] | None:
            if span is None:
                return None
            return (prompt_char_count + span[0], prompt_char_count + span[1])

        reasoning_span = shift_span(local_spans["reasoning"])
        answer_span = shift_span(local_spans["answer"])
        think_end_tag_span = shift_span(local_spans.get("think_end_tag"))
        response_span = shift_span(local_response_span)
        output_token_indices = _token_indices_after_char(normalized_offsets, prompt_char_count)
        reasoning_token_indices = (
            _token_indices_for_span(normalized_offsets, reasoning_span[0], reasoning_span[1])
            if reasoning_span is not None
            else []
        )
        answer_token_indices = (
            _token_indices_for_span(normalized_offsets, answer_span[0], answer_span[1])
            if answer_span is not None
            else []
        )

        reasoning_last_token = (
            _last_token_index_for_span(normalized_offsets, reasoning_span[0], reasoning_span[1])
            if reasoning_span is not None
            else None
        )
        think_end_last_token = (
            _last_token_index_for_span(normalized_offsets, think_end_tag_span[0], think_end_tag_span[1])
            if think_end_tag_span is not None
            else None
        )
        answer_last_token = (
            _last_token_index_for_span(normalized_offsets, answer_span[0], answer_span[1])
            if answer_span is not None
            else None
        )
        response_last_token = (
            _last_token_index_for_span(normalized_offsets, response_span[0], response_span[1])
            if response_span is not None
            else None
        )
        if response_last_token is None:
            response_last_token = last_generated

        token_positions = {
            "prompt_hidden": prompt_last_token,
            "answer_hidden": answer_last_token if answer_last_token is not None else response_last_token,
            "reasoning_hidden": reasoning_last_token if reasoning_last_token is not None else response_last_token,
            "think_end_hidden": think_end_last_token if think_end_last_token is not None else (
                reasoning_last_token if reasoning_last_token is not None else response_last_token
            ),
            "response_hidden": response_last_token,
        }
        think_end_window = list(
            range(max(0, int(token_positions["think_end_hidden"]) - 9), int(token_positions["think_end_hidden"]) + 1)
        )
        generated_length = sum(
            1
            for token_start, token_end in normalized_offsets
            if token_end > token_start and token_end > prompt_char_count
        )
        metadata_rows.append(
            {
                "response_anchor_kind": answer_kind,
                "token_positions": {
                    "prompt_last_token": prompt_last_token,
                    "answer_last_token": token_positions["answer_hidden"],
                    "reasoning_last_token": token_positions["reasoning_hidden"],
                    "think_end_last_token": token_positions["think_end_hidden"],
                    "response_last_token": token_positions["response_hidden"],
                    "last_generated": last_generated,
                },
                "token_groups": {
                    "output": output_token_indices,
                    "reasoning": reasoning_token_indices,
                    "answer": answer_token_indices,
                    "think_end_last10": think_end_window,
                },
                "prompt_length": max(prompt_last_token + 1, 0),
                "generated_length": generated_length,
            }
        )
        for component_name, token_index in token_positions.items():
            batch_component_positions.setdefault(component_name, []).append(int(token_index))

    return tokenized, [{"component_positions": batch_component_positions, **row} for row in metadata_rows]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract prompt-side and rollout-side hidden states from sampled inference runs. "
            "Each output row corresponds to one sampled rollout and keeps the original prompt-level task_id."
        )
    )
    parser.add_argument(
        "--model_name_or_path",
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        help="Hugging Face model id or local path.",
    )
    parser.add_argument("--run_dirs", nargs="*", default=[])
    parser.add_argument("--run_glob", type=str, default=None)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument(
        "--components",
        nargs="*",
        default=["prompt_hidden", "response_hidden"],
        choices=(
            "prompt_hidden",
            "reasoning_hidden",
            "think_end_hidden",
            "think_end_last10_hidden",
            "answer_hidden",
            "response_hidden",
            "reasoning_mean_hidden",
            "answer_mean_hidden",
            "output_mean_hidden",
        ),
    )
    parser.add_argument("--layers", type=str, default="27")
    parser.add_argument(
        "--response_anchor",
        choices=("answer", "reasoning", "reasoning_or_answer", "last_generated"),
        default="reasoning_or_answer",
    )
    parser.add_argument("--hidden_root", type=Path, default=DEFAULT_HIDDEN_ROOT)
    parser.add_argument("--index_root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--hidden_filename", default="rollout_hidden_states.pt")
    parser.add_argument("--index_filename", default="rollout_index.jsonl")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--max_batch_tokens",
        type=int,
        default=24000,
        help="Approximate per-batch token budget using input_length + output_length estimates.",
    )
    parser.add_argument(
        "--disable_length_sort",
        action="store_true",
        help="Disable sorting by estimated total tokens before batching.",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--torch_dtype", default="auto", choices=("auto", "float32", "float16", "bfloat16"))
    parser.add_argument("--disable_generation_prompt", action="store_true")
    parser.add_argument("--disable_thinking", action="store_true")
    parser.add_argument("--disable_chat_template", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.num_shards < 1:
        raise ValueError("--num_shards must be at least 1.")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard_index must be in [0, num_shards).")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be at least 1.")
    run_dirs = _resolve_run_dirs(args.run_dirs, args.run_glob)

    records: list[tuple[int, Path, int, dict[str, Any]]] = []
    for run_dir in run_dirs:
        experiments_path = run_dir / "all_experiments.jsonl"
        evaluations_path = run_dir / "evaluation_results.jsonl"
        if not experiments_path.exists():
            raise FileNotFoundError(f"Expected sampled run file at {experiments_path}.")
        experiment_rows = load_records(experiments_path)
        correctness: list[int] | None = None
        if evaluations_path.exists():
            evaluation_rows = load_records(evaluations_path)
            if evaluation_rows:
                raw_correctness = evaluation_rows[-1].get("correctness")
                if isinstance(raw_correctness, list):
                    correctness = [int(value) for value in raw_correctness]
        usable = min(len(experiment_rows), len(correctness)) if correctness is not None else len(experiment_rows)
        for row_idx, record in enumerate(experiment_rows[:usable]):
            if correctness is not None:
                record = dict(record)
                record["reward"] = int(correctness[row_idx])
                record["score"] = int(correctness[row_idx])
            records.append((len(records), run_dir, row_idx, record))

    if args.max_examples is not None:
        records = records[: args.max_examples]
    total_source_records = len(records)
    if args.num_shards > 1:
        records = [record for record in records if record[0] % args.num_shards == args.shard_index]
    if not records:
        raise ValueError("No rollout records were loaded from the provided run directories.")

    dataset_name = args.dataset_name or str(records[0][3].get("dataset_name") or "rollouts")
    model_slug = sanitize_name(args.model_name_or_path)
    hidden_filename = _append_shard_suffix(args.hidden_filename, args.shard_index, args.num_shards)
    index_filename = _append_shard_suffix(args.index_filename, args.shard_index, args.num_shards)
    hidden_output_path = (
        args.hidden_root.expanduser().resolve() / dataset_name / model_slug / hidden_filename
    )
    index_output_path = (
        args.index_root.expanduser().resolve() / dataset_name / model_slug / index_filename
    )
    checkpoint_dir = _checkpoint_dir_for_output(hidden_output_path)
    if (hidden_output_path.exists() or index_output_path.exists()) and not args.overwrite:
        raise FileExistsError(
            f"Output already exists at {hidden_output_path} or {index_output_path}. Pass --overwrite to replace."
        )
    if args.overwrite:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)

    resumed_examples, next_chunk_index = _resume_state_from_checkpoint(checkpoint_dir)
    if resumed_examples:
        print(
            f"Resuming from checkpoint {checkpoint_dir} "
            f"at {resumed_examples}/{len(records)} examples."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        torch_dtype=_resolve_torch_dtype(args.torch_dtype),
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    model.eval()
    input_device = _infer_input_device(model)

    blocks = list(_get_transformer_blocks(model))
    selected_layers = parse_layer_spec(args.layers, len(blocks))
    selected_set = set(selected_layers)

    current_batch_component_specs: dict[str, dict[str, Any]] = {}
    current_batch_vectors: list[dict[str, dict[int, torch.Tensor | None]]] = []
    hook_handles = []

    def make_hidden_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            if layer_idx not in selected_set:
                return
            gathered_by_component = {}
            for component_name, spec in current_batch_component_specs.items():
                if spec["kind"] == "position":
                    gathered_by_component[component_name] = _extract_batched_token_vectors(output, spec["values"])
                elif spec["kind"] == "group_mean":
                    gathered_by_component[component_name] = _extract_batched_group_pooled_vectors(
                        output,
                        spec["groups"],
                        spec["fallback"],
                    )
                elif spec["kind"] == "window":
                    gathered_by_component[component_name] = _extract_batched_window_vectors(
                        output,
                        spec["windows"],
                        spec["fallback"],
                        window_size=int(spec["window_size"]),
                    )
                else:
                    raise ValueError(f"Unsupported component spec kind: {spec['kind']}")
            for component_name, gathered_vectors in gathered_by_component.items():
                for batch_idx, vector in enumerate(gathered_vectors):
                    current_batch_vectors[batch_idx][component_name][layer_idx] = vector.contiguous().clone()
        return hook

    for layer_idx, block in enumerate(blocks):
        if layer_idx not in selected_set:
            continue
        hook_handles.append(block.register_forward_hook(make_hidden_hook(layer_idx)))

    final_metadata = {
        "dataset_name": dataset_name,
        "model_name_or_path": args.model_name_or_path,
        "components": args.components,
        "selected_layers": selected_layers,
        "num_source_examples": total_source_records,
        "response_anchor": args.response_anchor,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "batch_size": args.batch_size,
        "max_batch_tokens": args.max_batch_tokens,
        "length_sorted": not args.disable_length_sort,
    }
    try:
        prepared_records = []
        for global_idx, run_dir, row_idx, record in records:
            messages = _extract_messages(record)
            prompt_text = _render_prompt(
                tokenizer,
                messages,
                add_generation_prompt=not args.disable_generation_prompt,
                enable_thinking=not args.disable_thinking,
                use_chat_template=not args.disable_chat_template,
            )
            generated_text = str(record.get("generated_text", ""))
            prepared_records.append(
                {
                    "global_idx": global_idx,
                    "run_dir": run_dir,
                    "row_idx": row_idx,
                    "record": record,
                    "prompt_text": prompt_text,
                    "generated_text": generated_text,
                    "full_text": prompt_text + generated_text,
                    "estimated_total_tokens": _estimate_total_tokens(record),
                }
            )

        if not args.disable_length_sort:
            prepared_records.sort(key=lambda item: item["estimated_total_tokens"], reverse=True)

        if resumed_examples > len(prepared_records):
            raise ValueError(
                f"Checkpoint has {resumed_examples} processed examples, "
                f"but only {len(prepared_records)} records are available."
            )
        if resumed_examples:
            prepared_records = prepared_records[resumed_examples:]

        processed_examples = resumed_examples
        for batch in _iter_dynamic_batches(
            prepared_records,
            batch_size=args.batch_size,
            max_batch_tokens=args.max_batch_tokens,
        ):
            full_texts = [item["full_text"] for item in batch]
            prompt_texts = [item["prompt_text"] for item in batch]
            generated_texts = [item["generated_text"] for item in batch]

            tokenized, metadata_rows = _resolve_batch_token_metadata(
                tokenizer=tokenizer,
                full_texts=full_texts,
                prompt_texts=prompt_texts,
                generated_texts=generated_texts,
                response_anchor=args.response_anchor,
            )

            current_batch_vectors = [
                {
                    component_name: {layer_idx: None for layer_idx in selected_layers}
                    for component_name in args.components
                }
                for _ in batch
            ]
            current_batch_component_specs = {}
            for component_name in args.components:
                if component_name in ("prompt_hidden", "reasoning_hidden", "think_end_hidden", "answer_hidden", "response_hidden"):
                    current_batch_component_specs[component_name] = {
                        "kind": "position",
                        "values": metadata_rows[0]["component_positions"][component_name],
                    }
                elif component_name == "reasoning_mean_hidden":
                    current_batch_component_specs[component_name] = {
                        "kind": "group_mean",
                        "groups": [list(row["token_groups"].get("reasoning", [])) for row in metadata_rows],
                        "fallback": metadata_rows[0]["component_positions"]["reasoning_hidden"],
                    }
                elif component_name == "answer_mean_hidden":
                    current_batch_component_specs[component_name] = {
                        "kind": "group_mean",
                        "groups": [list(row["token_groups"].get("answer", [])) for row in metadata_rows],
                        "fallback": metadata_rows[0]["component_positions"]["answer_hidden"],
                    }
                elif component_name == "output_mean_hidden":
                    current_batch_component_specs[component_name] = {
                        "kind": "group_mean",
                        "groups": [list(row["token_groups"].get("output", [])) for row in metadata_rows],
                        "fallback": metadata_rows[0]["component_positions"]["response_hidden"],
                    }
                elif component_name == "think_end_last10_hidden":
                    current_batch_component_specs[component_name] = {
                        "kind": "window",
                        "windows": [list(row["token_groups"].get("think_end_last10", [])) for row in metadata_rows],
                        "fallback": metadata_rows[0]["component_positions"]["think_end_hidden"],
                        "window_size": 10,
                    }
                else:
                    raise ValueError(f"Unsupported component: {component_name}")
            tokenized = {key: value.to(input_device) for key, value in tokenized.items()}
            with torch.inference_mode():
                model_outputs = model(**tokenized, use_cache=False)
            confidence_feature_rows = _compute_batch_confidence_features(
                logits=model_outputs.logits.detach(),
                input_ids=tokenized["input_ids"],
                metadata_rows=metadata_rows,
            )

            batch_hidden_examples = []
            batch_index_records = []
            for batch_idx, prepared in enumerate(batch):
                record = prepared["record"]
                run_dir = prepared["run_dir"]
                row_idx = prepared["row_idx"]
                global_idx = prepared["global_idx"]
                batch_metadata = metadata_rows[batch_idx]

                example_payload = {
                    "dataset_name": str(record.get("dataset_name", dataset_name)),
                    "task_id": str(record.get("task_id", row_idx)),
                }
                for component_name in args.components:
                    missing_layers = [
                        layer_idx
                        for layer_idx, value in current_batch_vectors[batch_idx][component_name].items()
                        if value is None
                    ]
                    if missing_layers:
                        raise RuntimeError(
                            f"Missing rollout hidden vectors for component {component_name!r} "
                            f"on layers {missing_layers} for task {example_payload['task_id']}."
                        )
                    example_payload[component_name] = [
                        current_batch_vectors[batch_idx][component_name][layer_idx] for layer_idx in selected_layers
                    ]
                batch_hidden_examples.append(example_payload)

                index_record = dict(record)
                index_record["dataset_name"] = str(record.get("dataset_name", dataset_name))
                index_record["task_id"] = str(record.get("task_id", row_idx))
                index_record["rollout_row_index"] = int(row_idx)
                index_record["run_dir"] = str(run_dir)
                index_record["run_name"] = run_dir.name
                index_record["global_example_index"] = int(global_idx)
                index_record["response_anchor_kind"] = batch_metadata["response_anchor_kind"]
                index_record["selected_layers"] = list(selected_layers)
                index_record["prompt_length"] = batch_metadata["prompt_length"]
                index_record["generated_length"] = batch_metadata["generated_length"]
                index_record["token_positions"] = batch_metadata["token_positions"]
                rollout_features = extract_rollout_numeric_features(record)
                rollout_features.update(confidence_feature_rows[batch_idx])
                index_record["rollout_features"] = rollout_features
                batch_index_records.append(index_record)

                processed_examples += 1
                print(
                    f"Processed {processed_examples}/{len(records)} "
                    f"task_id={example_payload['task_id']} "
                    f"run={run_dir.name} "
                    f"split={index_record.get('split')} "
                    f"shard={args.shard_index + 1}/{args.num_shards} "
                    f"batch={len(batch)}"
                )

            chunk_path = _checkpoint_chunk_path(checkpoint_dir, next_chunk_index)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "hidden_examples": batch_hidden_examples,
                    "index_records": batch_index_records,
                },
                chunk_path,
            )
            next_chunk_index += 1
            _write_checkpoint_metadata(
                checkpoint_dir,
                processed_examples=processed_examples,
                total_examples=len(records),
                next_chunk_index=next_chunk_index,
            )
    finally:
        for handle in hook_handles:
            handle.remove()

    _finalize_from_checkpoint(
        checkpoint_dir,
        hidden_output_path=hidden_output_path,
        index_output_path=index_output_path,
        metadata=final_metadata,
    )

    print(
        json.dumps(
            {
                "hidden_output_path": str(hidden_output_path),
                "index_output_path": str(index_output_path),
                "num_examples": int(processed_examples),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
