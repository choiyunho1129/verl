#!/usr/bin/env python3
"""Offline per-step training of the SPPO-style prompt-level value critic.

Replays CRRL's prompt_reward_logs in order. For each global_step file we:
  1. Load (prompt, rollout_reward) pairs from records[*].rollout_rewards
  2. Run --epochs_per_step passes of BCE training
  3. Optionally evaluate on the matching validation_data jsonl
  4. Save a checkpoint at <output_dir>/global_step_<N>/

Critic = AutoModelForCausalLM base + linear v_head over the last non-pad
hidden state. Loss = BCEWithLogitsLoss against the rollout reward (treated
as a 0/1 success label), matching SPPO's dp_critic.update_critic.

Single-GPU. fp32 master weights, bf16 autocast forward. For a 4B base on
B200/H100-80GB this fits with gradient checkpointing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


VALIDATION_INPUT_RE = re.compile(r"^user\n(.*?)\nassistant\s*$", re.DOTALL)


@dataclass
class Sample:
    raw_prompt: str
    reward: float


def load_step_records(log_path: Path) -> list[Sample]:
    with log_path.open() as f:
        data = json.load(f)
    out: list[Sample] = []
    for rec in data["records"]:
        rp = rec["raw_prompt"]
        for r in rec["rollout_rewards"]:
            out.append(Sample(raw_prompt=rp, reward=float(r)))
    return out


def load_validation(jsonl_path: Path) -> list[Sample]:
    out: list[Sample] = []
    with jsonl_path.open() as f:
        for line in f:
            d = json.loads(line)
            inp = d["input"]
            m = VALIDATION_INPUT_RE.match(inp)
            raw = m.group(1) if m else inp
            score = d.get("score", d.get("reward"))
            if score is None:
                continue
            out.append(Sample(raw_prompt=raw, reward=float(score)))
    return out


def find_validation_file(val_dir: Path | None, step: int) -> Path | None:
    if val_dir is None or not val_dir.exists():
        return None
    candidates = sorted(int(p.stem) for p in val_dir.glob("*.jsonl"))
    if not candidates:
        return None
    closest = min(candidates, key=lambda c: abs(c - step))
    return val_dir / f"{closest}.jsonl"


class CriticModel(nn.Module):
    """Qwen-style CausalLM base + scalar v_head on last non-pad hidden state."""

    def __init__(
        self,
        base_model_path: str,
        gradient_checkpointing: bool = True,
        attn_impl: str | None = None,
    ):
        super().__init__()
        kwargs = {"torch_dtype": torch.float32}
        if attn_impl:
            kwargs["attn_implementation"] = attn_impl
        self.base = AutoModelForCausalLM.from_pretrained(base_model_path, **kwargs)
        if gradient_checkpointing:
            self.base.gradient_checkpointing_enable()
            self.base.config.use_cache = False
        hidden = self.base.config.hidden_size
        self.v_head = nn.Linear(hidden, 1, bias=False)
        nn.init.normal_(self.v_head.weight, std=1.0 / math.sqrt(hidden))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = out.hidden_states[-1]  # (B, T, H)
        last_idx = attention_mask.sum(dim=1) - 1
        last_hidden = hidden[torch.arange(hidden.size(0), device=hidden.device), last_idx]
        return self.v_head(last_hidden.float()).squeeze(-1)  # (B,) fp32


class PromptDataset(Dataset):
    def __init__(self, samples: list[Sample], tokenizer, max_length: int):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": s.raw_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "reward": torch.tensor(s.reward, dtype=torch.float32),
        }


def collate_fn(batch, pad_id: int):
    max_len = max(b["input_ids"].size(0) for b in batch)
    bsz = len(batch)
    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["input_ids"].size(0)
        input_ids[i, :n] = b["input_ids"]
        attention_mask[i, :n] = b["attention_mask"]
    rewards = torch.stack([b["reward"] for b in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "rewards": rewards}


@torch.no_grad()
def evaluate(model, loader, device, autocast_dtype) -> dict:
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    total_loss = 0.0
    n = 0
    correct = 0
    sum_pos_score = 0.0
    sum_neg_score = 0.0
    n_pos = 0
    n_neg = 0
    sum_score = 0.0
    sum_label = 0.0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        rewards = batch["rewards"].to(device)
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            logits = model(input_ids, attn)
        per_loss = loss_fn(logits, rewards)
        bs = rewards.size(0)
        total_loss += per_loss.sum().item()
        n += bs
        scores = torch.sigmoid(logits)
        preds = (scores > 0.5).float()
        correct += (preds == rewards).sum().item()
        pos_mask = rewards > 0.5
        sum_pos_score += scores[pos_mask].sum().item()
        sum_neg_score += scores[~pos_mask].sum().item()
        n_pos += int(pos_mask.sum().item())
        n_neg += int((~pos_mask).sum().item())
        sum_score += scores.sum().item()
        sum_label += rewards.sum().item()
    return {
        "val/bce_loss": total_loss / max(n, 1),
        "val/accuracy": correct / max(n, 1),
        "val/avg_score_pos": sum_pos_score / max(n_pos, 1),
        "val/avg_score_neg": sum_neg_score / max(n_neg, 1),
        "val/score_mean": sum_score / max(n, 1),
        "val/label_mean": sum_label / max(n, 1),
        "val/n": n,
    }


def save_checkpoint(model: CriticModel, tokenizer, optim, step: int, ckpt_dir: Path,
                    save_base: bool):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if save_base:
        # HF format so this checkpoint can be reloaded with AutoModelForCausalLM
        # by SPPO's verl critic worker if you want to warm-start online training.
        model.base.save_pretrained(ckpt_dir / "base", safe_serialization=True)
        tokenizer.save_pretrained(ckpt_dir / "base")
    torch.save(model.v_head.state_dict(), ckpt_dir / "v_head.pt")
    torch.save({"optim": optim.state_dict(), "step": step}, ckpt_dir / "optim.pt")
    with (ckpt_dir / "meta.json").open("w") as f:
        json.dump({"step": step, "saved_base": save_base}, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt_reward_logs", required=True,
                    help="Dir containing N.json files (one per global_step)")
    ap.add_argument("--validation_data", default=None,
                    help="Dir containing N.jsonl files (validation per step)")
    ap.add_argument("--base_model", required=True,
                    help="HF id or path to base model (e.g. Qwen/Qwen3-4B)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--eval_batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--v_head_lr_mult", type=float, default=10.0)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--epochs_per_step", type=int, default=1)
    ap.add_argument("--start_step", type=int, default=None)
    ap.add_argument("--end_step", type=int, default=None)
    ap.add_argument("--save_every", type=int, default=1,
                    help="Save full base every N steps; v_head saved every step")
    ap.add_argument("--no_save_base", action="store_true",
                    help="Never save the base model (only v_head + optim). "
                         "Useful when disk budget is tight and you only need v_head deltas.")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    log_dir = Path(args.prompt_reward_logs)
    val_dir = Path(args.validation_data) if args.validation_data else None
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step_files = sorted(log_dir.glob("*.json"), key=lambda p: int(p.stem))
    if args.start_step is not None:
        step_files = [p for p in step_files if int(p.stem) >= args.start_step]
    if args.end_step is not None:
        step_files = [p for p in step_files if int(p.stem) <= args.end_step]
    if not step_files:
        raise SystemExit(f"No step files matched in {log_dir}")
    print(f"[init] {len(step_files)} step files, "
          f"range {int(step_files[0].stem)}..{int(step_files[-1].stem)}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"  # preserve assistant-prompt suffix
    pad_id = tokenizer.pad_token_id

    print(f"[init] loading base model from {args.base_model}")
    model = CriticModel(args.base_model, gradient_checkpointing=True)
    model.to(args.device)
    autocast_dtype = torch.bfloat16

    decay_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and not n.endswith(".bias") and "norm" not in n.lower()]
    no_decay_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and (n.endswith(".bias") or "norm" in n.lower())]
    v_head_params = list(model.v_head.parameters())
    decay_ids = {id(p) for p in v_head_params}
    decay_params = [p for p in decay_params if id(p) not in decay_ids]
    optim = torch.optim.AdamW(
        [
            {"params": decay_params, "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "lr": args.lr, "weight_decay": 0.0},
            {"params": v_head_params, "lr": args.lr * args.v_head_lr_mult,
             "weight_decay": args.weight_decay},
        ],
    )
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    metrics_path = out_dir / "metrics.jsonl"
    for step_path in step_files:
        step = int(step_path.stem)
        t0 = time.time()
        train_samples = load_step_records(step_path)
        train_loader = DataLoader(
            PromptDataset(train_samples, tokenizer, args.max_length),
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, pad_id),
            num_workers=args.num_workers,
            pin_memory=True,
        )

        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for _ in range(args.epochs_per_step):
            for batch in train_loader:
                input_ids = batch["input_ids"].to(args.device, non_blocking=True)
                attn = batch["attention_mask"].to(args.device, non_blocking=True)
                rewards = batch["rewards"].to(args.device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    logits = model(input_ids, attn)
                per_loss = loss_fn(logits, rewards)
                loss = per_loss.mean()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if torch.isfinite(grad_norm):
                    optim.step()
                else:
                    print(f"[step {step}] non-finite grad_norm={grad_norm}, skipping")
                optim.zero_grad(set_to_none=True)
                train_loss_sum += per_loss.detach().sum().item()
                train_n += rewards.size(0)
        train_dt = time.time() - t0
        rec: dict = {
            "step": step,
            "train/bce_loss": train_loss_sum / max(train_n, 1),
            "train/n": train_n,
            "train/seconds": round(train_dt, 2),
        }

        if val_dir is not None:
            vp = find_validation_file(val_dir, step)
            if vp is not None:
                val_samples = load_validation(vp)
                if val_samples:
                    val_loader = DataLoader(
                        PromptDataset(val_samples, tokenizer, args.max_length),
                        batch_size=args.eval_batch_size,
                        shuffle=False,
                        collate_fn=lambda b: collate_fn(b, pad_id),
                        num_workers=args.num_workers,
                        pin_memory=True,
                    )
                    val_metrics = evaluate(model, val_loader, args.device, autocast_dtype)
                    val_metrics["val/source_file"] = vp.name
                    rec.update(val_metrics)

        ckpt_dir = out_dir / f"global_step_{step}"
        save_base = (not args.no_save_base) and (step % args.save_every == 0)
        save_checkpoint(model, tokenizer, optim, step, ckpt_dir, save_base=save_base)

        with metrics_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[step {step:>3}] " + " ".join(f"{k}={v}" for k, v in rec.items()))

    print(f"[done] checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
