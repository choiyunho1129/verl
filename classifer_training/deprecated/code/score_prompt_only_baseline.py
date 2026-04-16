from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

from classifer_training.prompt_only_experiments import (
    _build_dataset,
    _hidden_relation_features,
    _prompt_features,
)
from classifer_training.utils import load_records, write_jsonl


def _extract_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    for key in ("messages", "source_prompt"):
        messages = record.get(key)
        if isinstance(messages, list) and messages:
            return [
                {
                    "role": str(message.get("role", "user")),
                    "content": str(message.get("content", "")),
                }
                for message in messages
            ]
    user_input = record.get("user_input")
    if user_input is not None:
        return [{"role": "user", "content": str(user_input)}]
    prompt = record.get("prompt")
    if prompt is not None:
        return [{"role": "user", "content": str(prompt)}]
    raise KeyError("Record must contain one of messages/source_prompt/user_input/prompt.")


def _render_prompt(tokenizer, messages: list[dict[str, str]], add_generation_prompt: bool, enable_thinking: bool) -> str:
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


def _extract_pooled_hidden(layer_tensor: torch.Tensor, attention_mask: torch.Tensor, pooling: str) -> np.ndarray:
    lengths = attention_mask.sum(dim=1)
    outputs: list[torch.Tensor] = []
    for row_idx in range(layer_tensor.shape[0]):
        valid_len = int(lengths[row_idx].item())
        row = layer_tensor[row_idx, :valid_len, :]
        if pooling == "last10_mean":
            take = min(valid_len, 10)
            pooled = row[-take:, :].mean(dim=0)
        elif pooling == "last":
            pooled = row[-1, :]
        else:
            raise ValueError(f"Unsupported pooling: {pooling}")
        outputs.append(pooled.detach().cpu().to(torch.float32))
    return torch.stack(outputs, dim=0).numpy()


def _fit_baseline(hidden_dir: Path, index_dir: Path, labels_path: Path) -> Pipeline:
    labels = {str(row["task_id"]): row for row in load_records(labels_path)}
    dataset = _build_dataset(hidden_dir, index_dir, labels)
    X = np.concatenate([dataset["hidden_modes"]["layer17"], dataset["scalar"]], axis=1).astype(np.float32)
    y = dataset["y_reg"].astype(np.float32)
    splits = dataset["splits"]
    train_mask = np.isin(splits, ["train", "validation"])
    model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=3000.0))])
    model.fit(X[train_mask], y[train_mask])
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score full DAPO with the prompt-only baseline.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--hf_dataset_id", default="open-r1/DAPO-Math-17k-Processed")
    parser.add_argument("--hf_split", default="train")
    parser.add_argument("--hidden_dir", type=Path, required=True)
    parser.add_argument("--index_dir", type=Path, required=True)
    parser.add_argument("--labels_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--disable_generation_prompt", action="store_true")
    parser.add_argument("--disable_thinking", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = _fit_baseline(
        args.hidden_dir.expanduser().resolve(),
        args.index_dir.expanduser().resolve(),
        args.labels_path.expanduser().resolve(),
    )

    dataset = load_dataset(args.hf_dataset_id, split=args.hf_split)
    records = [dict(row) for idx, row in enumerate(dataset) if idx % args.num_shards == args.shard_idx]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    device = next(model.parameters()).device

    rows = []
    for start in range(0, len(records), args.batch_size):
        batch_records = records[start : start + args.batch_size]
        messages_batch = [_extract_messages(record) for record in batch_records]
        prompts = [
            _render_prompt(
                tokenizer,
                messages,
                add_generation_prompt=not args.disable_generation_prompt,
                enable_thinking=not args.disable_thinking,
            )
            for messages in messages_batch
        ]
        user_inputs = [str(record.get("user_input") or messages[-1]["content"]) for record, messages in zip(batch_records, messages_batch)]
        tokenized = tokenizer(prompts, return_tensors="pt", padding=True)
        attention_mask = tokenized["attention_mask"]
        input_lengths = attention_mask.sum(dim=1).cpu().numpy().astype(np.int64)
        tokenized = {key: value.to(device) for key, value in tokenized.items()}
        with torch.inference_mode():
            outputs = model(**tokenized, output_hidden_states=True, use_cache=False)
        hidden_states = outputs.hidden_states[1:]
        pooled_layer17 = _extract_pooled_hidden(hidden_states[17], tokenized["attention_mask"], pooling="last10_mean")

        all_layers_np = []
        for layer_tensor in hidden_states:
            all_layers_np.append(_extract_pooled_hidden(layer_tensor, tokenized["attention_mask"], pooling="last"))

        features = []
        for batch_idx, record in enumerate(batch_records):
            layer_vectors = [layer_np[batch_idx] for layer_np in all_layers_np]
            scalar = np.concatenate(
                [
                    _prompt_features(user_inputs[batch_idx], int(input_lengths[batch_idx])),
                    _hidden_relation_features(layer_vectors),
                ],
                axis=0,
            )
            features.append(np.concatenate([pooled_layer17[batch_idx], scalar], axis=0))

        X_batch = np.stack(features, axis=0).astype(np.float32)
        preds = np.clip(baseline.predict(X_batch).astype(np.float32), 0.0, 1.0)

        for record, user_input, pred in zip(batch_records, user_inputs, preds):
            task_id = (
                record.get("task_id")
                or record.get("extra_info", {}).get("index")
                or record.get("id")
                or None
            )
            rows.append(
                {
                    "task_id": str(task_id),
                    "user_input": user_input,
                    "predicted_difficulty": float(pred),
                    "probe": "prompt_only_last10mean_layer17_ridge_a3000",
                }
            )

    output_path = args.output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, rows)
    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "num_rows": len(rows),
                "shard_idx": args.shard_idx,
                "num_shards": args.num_shards,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
