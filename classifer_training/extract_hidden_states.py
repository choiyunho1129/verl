from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from classifer_training.utils import load_records, sanitize_name, write_jsonl

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_ROOT = REPO_ROOT / "classifer_training" / "artifacts" / "datasets"
DEFAULT_HIDDEN_ROOT = REPO_ROOT / "classifer_training" / "artifacts" / "hidden"
DEFAULT_INDEX_ROOT = REPO_ROOT / "classifer_training" / "artifacts" / "index"


def _extract_pooled_token_vector(output: Any, *, token_count: int, pooling: str, last_n: int) -> torch.Tensor:
    if isinstance(output, tuple):
        output = output[0]
    if isinstance(output, dict):
        output = output.get("hidden_states", output.get("last_hidden_state", output))
    if not torch.is_tensor(output):
        raise TypeError(f"Unsupported hook output type: {type(output)!r}")

    tensor = output.detach().cpu().to(torch.float32)
    if tensor.ndim >= 3:
        token_slice = tensor[:, :token_count, :]
        if pooling == "last":
            pooled = token_slice[:, -1, :]
        elif pooling == "lastn_mean":
            pooled = token_slice[:, -min(token_count, max(last_n, 1)) :, :].mean(dim=1)
        elif pooling == "last5_mean":
            pooled = token_slice[:, -min(token_count, 5) :, :].mean(dim=1)
        elif pooling == "last10_mean":
            pooled = token_slice[:, -min(token_count, 10) :, :].mean(dim=1)
        elif pooling == "mean":
            pooled = token_slice.mean(dim=1)
        elif pooling == "max":
            pooled = token_slice.max(dim=1).values
        else:
            raise ValueError(f"Unsupported token pooling mode: {pooling}")
        return pooled.squeeze(0).contiguous().clone()
    if tensor.ndim == 2:
        token_slice = tensor[:token_count, :]
        if pooling == "last":
            pooled = token_slice[-1, :]
        elif pooling == "lastn_mean":
            pooled = token_slice[-min(token_count, max(last_n, 1)) :, :].mean(dim=0)
        elif pooling == "last5_mean":
            pooled = token_slice[-min(token_count, 5) :, :].mean(dim=0)
        elif pooling == "last10_mean":
            pooled = token_slice[-min(token_count, 10) :, :].mean(dim=0)
        elif pooling == "mean":
            pooled = token_slice.mean(dim=0)
        elif pooling == "max":
            pooled = token_slice.max(dim=0).values
        else:
            raise ValueError(f"Unsupported token pooling mode: {pooling}")
        return pooled.contiguous().clone()
    if tensor.ndim == 1:
        return tensor.contiguous().clone()
    raise ValueError(f"Expected at least 1D tensor output, got shape {tuple(tensor.shape)}.")


def _extract_pooled_batch(output: Any, *, attention_mask: torch.Tensor, pooling: str, last_n: int) -> torch.Tensor:
    if isinstance(output, tuple):
        output = output[0]
    if isinstance(output, dict):
        output = output.get("hidden_states", output.get("last_hidden_state", output))
    if not torch.is_tensor(output):
        raise TypeError(f"Unsupported hook output type: {type(output)!r}")

    tensor = output.detach().cpu().to(torch.float32)
    if tensor.ndim != 3:
        if tensor.ndim == 2:
            return tensor
        raise ValueError(f"Expected batched hidden tensor with shape [B, T, D], got {tuple(tensor.shape)}.")

    lengths = attention_mask.sum(dim=1).tolist()
    pooled_rows: list[torch.Tensor] = []
    for row_idx, valid_len in enumerate(lengths):
        row = tensor[row_idx, : int(valid_len), :]
        if pooling == "last":
            pooled = row[-1, :]
        elif pooling == "lastn_mean":
            pooled = row[-min(int(valid_len), max(last_n, 1)) :, :].mean(dim=0)
        elif pooling == "last5_mean":
            pooled = row[-min(int(valid_len), 5) :, :].mean(dim=0)
        elif pooling == "last10_mean":
            pooled = row[-min(int(valid_len), 10) :, :].mean(dim=0)
        elif pooling == "mean":
            pooled = row.mean(dim=0)
        elif pooling == "max":
            pooled = row.max(dim=0).values
        else:
            raise ValueError(f"Unsupported token pooling mode: {pooling}")
        pooled_rows.append(pooled.contiguous())
    return torch.stack(pooled_rows, dim=0)


def _get_transformer_blocks(model) -> Iterable:
    candidates = [
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    ]
    for parent_name, layers_name in candidates:
        parent = getattr(model, parent_name, None)
        if parent is None:
            continue
        layers = getattr(parent, layers_name, None)
        if layers is not None:
            return layers
    raise AttributeError(
        "Could not find transformer blocks on the model. Expected one of model.layers, transformer.h, or gpt_neox.layers."
    )


def _infer_input_device(model) -> torch.device:
    model_device = getattr(model, "device", None)
    if isinstance(model_device, torch.device) and model_device.type != "meta":
        return model_device
    return next(model.parameters()).device


def _extract_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    for key in ("messages", "source_prompt"):
        messages = record.get(key)
        if isinstance(messages, list) and messages:
            normalized = []
            for message in messages:
                if not isinstance(message, dict):
                    raise TypeError(f"{key} must be a list of dicts.")
                normalized.append(
                    {
                        "role": str(message.get("role", "user")),
                        "content": str(message.get("content", "")),
                    }
                )
            return normalized

    for key in ("user_input", "prompt", "question"):
        value = record.get(key)
        if value is not None:
            return [{"role": "user", "content": str(value)}]

    raise KeyError("Each dataset row must contain one of messages/source_prompt/user_input/prompt/question.")


def _render_prompt(
    tokenizer,
    messages: list[dict[str, str]],
    add_generation_prompt: bool,
    enable_thinking: bool,
    *,
    use_chat_template: bool = True,
) -> str:
    if not use_chat_template:
        return "\n\n".join(message["content"] for message in messages)
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            return "\n\n".join(message["content"] for message in messages)
    except Exception:
        return "\n\n".join(message["content"] for message in messages)


def _resolve_input_records(input_path: Path) -> list[dict[str, Any]]:
    if input_path.is_dir():
        records: list[dict[str, Any]] = []
        for split_name in ("train", "validation", "test"):
            split_path = input_path / f"{split_name}.jsonl"
            if split_path.exists():
                records.extend(load_records(split_path))
        if records:
            return records
        raise FileNotFoundError(
            f"No train/validation/test JSONL files found under {input_path}."
        )
    return load_records(input_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract last-token hidden states from a normalized dataset for a Hugging Face causal LM."
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
    parser.add_argument("--dataset_name", default=None, help="Optional override. Defaults to the first row's dataset_name or file stem.")
    parser.add_argument(
        "--components",
        nargs="*",
        default=["hidden"],
        choices=("hidden", "attn", "ffn"),
        help="Components to save. hidden uses model output hidden_states. attn and ffn use forward hooks.",
    )
    parser.add_argument("--hidden_root", type=Path, default=DEFAULT_HIDDEN_ROOT)
    parser.add_argument("--index_root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--hidden_filename", default="hidden_states.pt")
    parser.add_argument("--index_filename", default="index.jsonl")
    parser.add_argument("--model_slug", default=None, help="Optional override for artifact directory naming.")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--torch_dtype", default="auto", choices=("auto", "float32", "float16", "bfloat16"))
    parser.add_argument("--token_pooling", default="last", choices=("last", "lastn_mean", "last5_mean", "last10_mean", "mean", "max"))
    parser.add_argument("--last_n", type=int, default=10, help="Token count used when --token_pooling=lastn_mean.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--disable_generation_prompt", action="store_true")
    parser.add_argument("--disable_thinking", action="store_true")
    parser.add_argument("--disable_chat_template", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cuda_device", type=int, default=None, help="CUDA device index (e.g. 3). If set, bypasses CUDA_VISIBLE_DEVICES.")
    return parser.parse_args(argv)


def _resolve_torch_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    input_path = args.input_path.expanduser().resolve()
    records = _resolve_input_records(input_path)
    if args.max_examples is not None:
        records = records[: args.max_examples]
    if not records:
        raise ValueError(f"No records found in {args.input_path}.")

    dataset_name = args.dataset_name or str(records[0].get("dataset_name") or input_path.stem)
    model_slug = args.model_slug or sanitize_name(args.model_name_or_path)
    hidden_output_path = (args.hidden_root.expanduser().resolve() / dataset_name / model_slug / args.hidden_filename)
    index_output_path = (args.index_root.expanduser().resolve() / dataset_name / model_slug / args.index_filename)
    if (hidden_output_path.exists() or index_output_path.exists()) and not args.overwrite:
        raise FileExistsError(
            f"Output already exists at {hidden_output_path} or {index_output_path}. Pass --overwrite to replace."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    if args.cuda_device is not None:
        _device_map = {"": f"cuda:{args.cuda_device}"}
    else:
        _cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        _device_map = {"": "cuda:0"} if _cvd else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map=_device_map,
        dtype=_resolve_torch_dtype(args.torch_dtype),
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    input_device = _infer_input_device(model)

    blocks = list(_get_transformer_blocks(model))
    num_layers = len(blocks)
    save_attn = "attn" in args.components
    save_ffn = "ffn" in args.components
    save_hidden = "hidden" in args.components

    attn_outs = [None for _ in range(num_layers)]
    ffn_outs = [None for _ in range(num_layers)]
    current_token_count = {"value": 0}

    hook_handles = []
    if save_attn or save_ffn:
        def make_attn_hook(layer_idx):
            def hook(_module, _inputs, output):
                attn_outs[layer_idx] = _extract_pooled_token_vector(
                    output,
                    token_count=current_token_count["value"],
                    pooling=args.token_pooling,
                    last_n=args.last_n,
                )
            return hook

        def make_ffn_hook(layer_idx):
            def hook(_module, _inputs, output):
                ffn_outs[layer_idx] = _extract_pooled_token_vector(
                    output,
                    token_count=current_token_count["value"],
                    pooling=args.token_pooling,
                    last_n=args.last_n,
                )
            return hook

        for layer_idx, block in enumerate(blocks):
            attn_module = getattr(block, "self_attn", None) or getattr(block, "attn", None)
            mlp_module = getattr(block, "mlp", None) or getattr(block, "feed_forward", None)
            if save_attn:
                if attn_module is None:
                    raise AttributeError(f"Layer {layer_idx} is missing an attention module.")
                hook_handles.append(attn_module.register_forward_hook(make_attn_hook(layer_idx)))
            if save_ffn:
                if mlp_module is None:
                    raise AttributeError(f"Layer {layer_idx} is missing an MLP/feed-forward module.")
                hook_handles.append(mlp_module.register_forward_hook(make_ffn_hook(layer_idx)))

    if args.batch_size > 1 and (save_attn or save_ffn):
        raise ValueError("batch_size > 1 is currently supported only for hidden-state extraction.")

    hidden_examples = []
    index_records = []
    try:
        for batch_start in range(0, len(records), max(args.batch_size, 1)):
            batch_records = records[batch_start : batch_start + max(args.batch_size, 1)]
            prompts = []
            for record in batch_records:
                messages = _extract_messages(record)
                prompts.append(
                    _render_prompt(
                        tokenizer,
                        messages,
                        add_generation_prompt=not args.disable_generation_prompt,
                        enable_thinking=not args.disable_thinking,
                        use_chat_template=not args.disable_chat_template,
                    )
                )
            tokenized = tokenizer(prompts, return_tensors="pt", padding=True)
            attention_mask_cpu = tokenized["attention_mask"].cpu()
            prompt_lengths = attention_mask_cpu.sum(dim=1).tolist()

            if args.batch_size == 1:
                current_token_count["value"] = int(prompt_lengths[0])
            tokenized = {key: value.to(input_device) for key, value in tokenized.items()}

            for layer_idx in range(num_layers):
                attn_outs[layer_idx] = None
                ffn_outs[layer_idx] = None

            with torch.inference_mode():
                outputs = model(
                    **tokenized,
                    output_hidden_states=save_hidden,
                    use_cache=False,
                )

            pooled_hidden_by_layer: list[torch.Tensor] = []
            if save_hidden:
                hidden_states = outputs.hidden_states
                if hidden_states is None:
                    raise RuntimeError("Model did not return hidden_states even though output_hidden_states=True.")
                pooled_hidden_by_layer = [
                    _extract_pooled_batch(
                        hidden_state,
                        attention_mask=attention_mask_cpu,
                        pooling=args.token_pooling,
                        last_n=args.last_n,
                    )
                    for hidden_state in hidden_states[1:]
                ]

            for batch_idx, record in enumerate(batch_records):
                row_idx = batch_start + batch_idx
                prompt_length = int(prompt_lengths[batch_idx])
                example_payload = {
                    "dataset_name": str(record.get("dataset_name", dataset_name)),
                    "task_id": str(record.get("task_id", row_idx)),
                }
                if save_hidden:
                    example_payload["hidden"] = [layer_batch[batch_idx].clone() for layer_batch in pooled_hidden_by_layer]
                if save_attn:
                    if any(value is None for value in attn_outs):
                        raise RuntimeError(f"Attention hooks did not fire for every layer on row {row_idx}.")
                    example_payload["attn"] = [value.clone() for value in attn_outs]
                if save_ffn:
                    if any(value is None for value in ffn_outs):
                        raise RuntimeError(f"FFN hooks did not fire for every layer on row {row_idx}.")
                    example_payload["ffn"] = [value.clone() for value in ffn_outs]
                hidden_examples.append(example_payload)

                index_record = dict(record)
                index_record["dataset_name"] = str(record.get("dataset_name", dataset_name))
                index_record["task_id"] = str(record.get("task_id", row_idx))
                index_record["input_length"] = prompt_length
                index_records.append(index_record)

                print(
                    f"Processed {row_idx + 1}/{len(records)} "
                    f"task_id={example_payload['task_id']} split={index_record.get('split')} input_length={prompt_length}"
                )
    finally:
        for handle in hook_handles:
            handle.remove()

    hidden_output_path.parent.mkdir(parents=True, exist_ok=True)
    index_output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": {
                "dataset_name": dataset_name,
                "model_name_or_path": args.model_name_or_path,
                "components": args.components,
                "token_pooling": args.token_pooling,
                "num_layers": num_layers,
                "num_examples": len(hidden_examples),
            },
            "examples": hidden_examples,
        },
        hidden_output_path,
    )
    write_jsonl(index_output_path, index_records)

    print(json.dumps({"hidden_output_path": str(hidden_output_path), "index_output_path": str(index_output_path)}, indent=2))


if __name__ == "__main__":
    main()
