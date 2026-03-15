#!/usr/bin/env python3
"""
Visualize per-token distillation feedback for a single rollout trajectory.

This script is designed for the GKD setup used in `recipe/gkd/test_qwen.sh`.
It reads one validation JSONL dump from `trainer.validation_data_dir`,
recomputes student and teacher log-probs on the generated trajectory, and
renders an HTML heatmap similar to token-level attribution views.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a token-level KD heatmap for one validation trajectory.")
    parser.add_argument(
        "--validation_source",
        required=True,
        help="Validation JSONL file or directory produced by trainer.validation_data_dir.",
    )
    parser.add_argument(
        "--step",
        default="latest",
        help="Validation step to load when --validation_source is a directory. Use an integer or 'latest'.",
    )
    parser.add_argument(
        "--select",
        default="row",
        choices=["row", "longest", "shortest", "best_score", "worst_score"],
        help="How to choose a sample from the selected JSONL file.",
    )
    parser.add_argument("--row", type=int, default=0, help="Row index used when --select=row.")
    parser.add_argument(
        "--contains",
        default=None,
        help="Optional substring filter applied to the generated output before sample selection.",
    )
    parser.add_argument("--student_model", required=True, help="Student HF model or checkpoint path.")
    parser.add_argument("--teacher_model", required=True, help="Teacher HF model path.")
    parser.add_argument(
        "--student_device",
        default="auto",
        help="Student device: auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument(
        "--teacher_device",
        default="auto",
        help="Teacher device: auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument(
        "--metric",
        default="advantage",
        choices=[
            "advantage",
            "reverse_kl",
            "abs_gap",
            "student_logprob",
            "teacher_logprob",
            "student_prob",
            "teacher_prob",
        ],
        help="Token metric used for the inline heatmap.",
    )
    parser.add_argument(
        "--out_html",
        default=None,
        help="Output HTML path. Defaults to a file next to the selected validation JSONL.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True,
        help="Pass trust_remote_code=True to Hugging Face loaders.",
    )
    parser.add_argument(
        "--no_trust_remote_code",
        action="store_false",
        dest="trust_remote_code",
        help="Disable trust_remote_code.",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=20,
        help="How many high/low-signal tokens to show in the summary tables.",
    )
    return parser.parse_args()


def resolve_validation_file(source: str, step: str) -> Path:
    source_path = Path(source)
    if source_path.is_file():
        return source_path
    if not source_path.is_dir():
        raise FileNotFoundError(f"Validation source does not exist: {source}")

    candidates = sorted(p for p in source_path.glob("*.jsonl") if p.is_file())
    if not candidates:
        raise FileNotFoundError(f"No JSONL files found in {source}")

    if step == "latest":
        numeric = []
        for path in candidates:
            try:
                numeric.append((int(path.stem), path))
            except ValueError:
                continue
        if numeric:
            return max(numeric, key=lambda item: item[0])[1]
        return max(candidates, key=lambda path: path.stat().st_mtime)

    target = source_path / f"{step}.jsonl"
    if not target.exists():
        raise FileNotFoundError(f"Requested step file does not exist: {target}")
    return target


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_prompt_and_output(row: dict[str, Any]) -> tuple[str, str]:
    prompt = row.get("input")
    if prompt is None:
        prompt = row.get("prompt")
    output = row.get("output")
    if output is None:
        output = row.get("generated_text")
    if output is None:
        output = row.get("output_text")

    if not isinstance(prompt, str) or not isinstance(output, str):
        raise ValueError(
            "Could not find prompt/output strings. Expected fields like "
            "`input`+`output` or `prompt`+`generated_text`."
        )
    return prompt, output


def select_sample(
    rows: list[dict[str, Any]],
    select: str,
    row_index: int,
    contains: str | None,
) -> tuple[int, dict[str, Any]]:
    indexed_rows = list(enumerate(rows))

    if contains:
        filtered: list[tuple[int, dict[str, Any]]] = []
        for idx, row in indexed_rows:
            _, output = extract_prompt_and_output(row)
            if contains in output:
                filtered.append((idx, row))
        indexed_rows = filtered
        if not indexed_rows:
            raise ValueError(f"No rows matched --contains={contains!r}")

    if select == "row":
        if row_index < 0 or row_index >= len(indexed_rows):
            raise IndexError(f"--row={row_index} is out of range for {len(indexed_rows)} rows")
        return indexed_rows[row_index]

    if select == "longest":
        return max(indexed_rows, key=lambda item: len(extract_prompt_and_output(item[1])[1]))
    if select == "shortest":
        return min(indexed_rows, key=lambda item: len(extract_prompt_and_output(item[1])[1]))
    if select == "best_score":
        return max(indexed_rows, key=lambda item: (_as_float(item[1].get("score")) is not None, _as_float(item[1].get("score")) or float("-inf")))
    if select == "worst_score":
        return min(indexed_rows, key=lambda item: (_as_float(item[1].get("score")) is None, _as_float(item[1].get("score")) or float("inf")))

    raise ValueError(f"Unsupported selection mode: {select}")


def load_model_and_tokenizer(model_path: str, device: str, trust_remote_code: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": "auto",
        "low_cpu_mem_usage": True,
    }

    if device == "auto":
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.eval()

    if device != "auto":
        model.to(device)

    model_device = next(model.parameters()).device
    return model, tokenizer, model_device


def score_output_tokens(
    model,
    tokenizer,
    model_device,
    prompt: str,
    output: str,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    output_enc = tokenizer(output, add_special_tokens=False, return_offsets_mapping=True)
    output_ids = output_enc["input_ids"]
    offsets = output_enc.get("offset_mapping")

    full_ids = prompt_ids + output_ids
    if len(full_ids) < 2:
        raise ValueError("Prompt/output pair is too short to score.")

    input_ids = torch.tensor(full_ids, device=model_device).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_ids).logits

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    entropy = -(log_probs.exp() * log_probs).sum(-1)

    prompt_len = len(prompt_ids)
    start = max(prompt_len - 1, 0)
    end = shift_labels.size(1) - 1
    if start > end:
        raise ValueError("Prompt consumes the full sequence; no response tokens were found.")

    out_log_probs = log_probs[0, start : end + 1, :]
    out_labels = shift_labels[0, start : end + 1]
    token_logprobs = out_log_probs.gather(1, out_labels.unsqueeze(-1)).squeeze(-1)
    out_entropy = entropy[0, start : end + 1]

    return {
        "output_ids": output_ids,
        "offsets": offsets,
        "logprobs": token_logprobs.detach().cpu().tolist(),
        "entropies": out_entropy.detach().cpu().tolist(),
    }


def percentile_scale(values: list[float], signed: bool) -> float:
    if not values:
        return 1.0
    arr = np.asarray(values, dtype=float)
    if signed:
        arr = np.abs(arr)
    scale = float(np.percentile(arr, 95))
    if not math.isfinite(scale) or scale <= 0:
        scale = float(np.max(arr)) if arr.size else 1.0
    return scale if scale > 0 else 1.0


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def rgba_for_metric(value: float, scale: float, signed: bool) -> str:
    intensity = clamp01(abs(value) / max(scale, 1e-8))
    alpha = 0.10 + 0.75 * intensity
    if signed:
        if value >= 0:
            return f"rgba(191, 51, 44, {alpha:.3f})"
        return f"rgba(46, 109, 164, {alpha:.3f})"
    return f"rgba(191, 51, 44, {alpha:.3f})"


def token_title(token: dict[str, Any]) -> str:
    pieces = [
        f"idx={token['index']}",
        f"text={token['text']!r}",
        f"student_logprob={token['student_logprob']:.4f}",
        f"teacher_logprob={token['teacher_logprob']:.4f}",
        f"advantage={token['advantage']:.4f}",
        f"reverse_kl={token['reverse_kl']:.4f}",
        f"student_prob={token['student_prob']:.4f}",
        f"teacher_prob={token['teacher_prob']:.4f}",
        f"student_entropy={token['student_entropy']:.4f}",
        f"teacher_entropy={token['teacher_entropy']:.4f}",
    ]
    return " | ".join(pieces)


def render_output_heatmap(
    output_text: str,
    tokens: list[dict[str, Any]],
    metric: str,
    scale: float,
    signed_metric: bool,
) -> str:
    fragments: list[str] = []
    cursor = 0

    for token in tokens:
        start = token["start"]
        end = token["end"]
        if start > cursor:
            fragments.append(html.escape(output_text[cursor:start]))

        text = output_text[start:end]
        bg = rgba_for_metric(float(token[metric]), scale=scale, signed=signed_metric)
        title = html.escape(token_title(token), quote=True)
        token_html = html.escape(text) if text else "&nbsp;"
        fragments.append(
            f'<span class="tok" style="background:{bg}" title="{title}">{token_html}</span>'
        )
        cursor = end

    if cursor < len(output_text):
        fragments.append(html.escape(output_text[cursor:]))

    return "".join(fragments)


def summarize_positions(tokens: list[dict[str, Any]], metric: str) -> list[tuple[str, float]]:
    if not tokens:
        return []

    values = np.asarray([float(tok[metric]) for tok in tokens], dtype=float)
    sections = []
    boundaries = [0.0, 0.25, 0.5, 0.75, 1.0]
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        start = int(len(values) * left)
        end = int(len(values) * right)
        if end <= start:
            continue
        label = f"{int(left * 100)}-{int(right * 100)}%"
        sections.append((label, float(np.mean(values[start:end]))))
    return sections


def render_token_table(tokens: list[dict[str, Any]], metric: str, top_n: int, reverse: bool) -> str:
    ordered = sorted(tokens, key=lambda tok: float(tok[metric]), reverse=reverse)[:top_n]
    rows = []
    for tok in ordered:
        rows.append(
            "<tr>"
            f"<td>{tok['index']}</td>"
            f"<td><code>{html.escape(repr(tok['text']))}</code></td>"
            f"<td>{tok[metric]:.4f}</td>"
            f"<td>{tok['advantage']:.4f}</td>"
            f"<td>{tok['student_logprob']:.4f}</td>"
            f"<td>{tok['teacher_logprob']:.4f}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def metric_description(metric: str) -> str:
    descriptions = {
        "advantage": "teacher_logprob - student_logprob; positive values are reinforced by the current GKD recipe.",
        "reverse_kl": "student_logprob - teacher_logprob; this matches the signed token gap behind val/reverse_kl.",
        "abs_gap": "absolute teacher/student logprob gap.",
        "student_logprob": "student log-probability of the sampled token.",
        "teacher_logprob": "teacher log-probability of the sampled token.",
        "student_prob": "student probability of the sampled token.",
        "teacher_prob": "teacher probability of the sampled token.",
    }
    return descriptions[metric]


def build_tokens(
    output_text: str,
    offsets: list[list[int]] | list[tuple[int, int]] | None,
    student_scores: dict[str, Any],
    teacher_scores: dict[str, Any],
) -> list[dict[str, Any]]:
    if offsets is None:
        raise ValueError("Tokenizer did not provide offset_mapping; cannot render inline heatmap.")

    if len(student_scores["logprobs"]) != len(teacher_scores["logprobs"]):
        raise ValueError(
            "Student/teacher token counts do not match: "
            f"{len(student_scores['logprobs'])} vs {len(teacher_scores['logprobs'])}"
        )

    tokens: list[dict[str, Any]] = []
    for idx, (offset, s_lp, t_lp, s_ent, t_ent) in enumerate(
        zip(
            offsets,
            student_scores["logprobs"],
            teacher_scores["logprobs"],
            student_scores["entropies"],
            teacher_scores["entropies"],
        )
    ):
        start, end = int(offset[0]), int(offset[1])
        text = output_text[start:end]
        advantage = float(t_lp - s_lp)
        reverse_kl = float(s_lp - t_lp)
        tokens.append(
            {
                "index": idx,
                "text": text,
                "start": start,
                "end": end,
                "student_logprob": float(s_lp),
                "teacher_logprob": float(t_lp),
                "student_prob": float(math.exp(s_lp)),
                "teacher_prob": float(math.exp(t_lp)),
                "advantage": advantage,
                "reverse_kl": reverse_kl,
                "abs_gap": abs(advantage),
                "student_entropy": float(s_ent),
                "teacher_entropy": float(t_ent),
            }
        )
    return tokens


def default_output_path(validation_file: Path, metric: str, selected_row: int, step: Any) -> Path:
    stem = validation_file.stem
    return validation_file.with_name(f"{stem}.row{selected_row}.{metric}.html")


def render_html(
    *,
    source_file: Path,
    step_value: Any,
    selected_row: int,
    row: dict[str, Any],
    prompt: str,
    output: str,
    student_model: str,
    teacher_model: str,
    metric: str,
    tokens: list[dict[str, Any]],
) -> str:
    signed_metric = metric in {"advantage", "reverse_kl"}
    scale = percentile_scale([float(tok[metric]) for tok in tokens], signed=signed_metric)
    heatmap_html = render_output_heatmap(output, tokens, metric=metric, scale=scale, signed_metric=signed_metric)

    metric_values = np.asarray([float(tok[metric]) for tok in tokens], dtype=float)
    advantage_values = np.asarray([float(tok["advantage"]) for tok in tokens], dtype=float)
    reverse_kl_values = np.asarray([float(tok["reverse_kl"]) for tok in tokens], dtype=float)
    position_sections = summarize_positions(tokens, metric=metric)

    positive_frac = float(np.mean(advantage_values > 0)) if len(advantage_values) else 0.0
    negative_frac = float(np.mean(advantage_values < 0)) if len(advantage_values) else 0.0

    meta_rows = []
    for key, value in row.items():
        if key in {"input", "output", "prompt", "generated_text", "output_text"}:
            continue
        text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
        meta_rows.append(
            "<tr>"
            f"<th>{html.escape(str(key))}</th>"
            f"<td>{html.escape(text)}</td>"
            "</tr>"
        )

    position_rows = []
    for label, mean_value in position_sections:
        position_rows.append(f"<tr><td>{label}</td><td>{mean_value:.4f}</td></tr>")

    best_rows = render_token_table(tokens, metric=metric, top_n=20 if not tokens else min(len(tokens), 20), reverse=True)
    worst_rows = render_token_table(tokens, metric=metric, top_n=20 if not tokens else min(len(tokens), 20), reverse=False)

    title = f"KD trajectory heatmap: step {step_value}, row {selected_row}"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --paper: #fffdfa;
      --ink: #1f1812;
      --muted: #6f6256;
      --border: #d8cfc2;
      --accent: #8f2f24;
      --cool: #2e6da4;
    }}
    body {{
      margin: 0;
      font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
      background:
        radial-gradient(circle at top left, rgba(143, 47, 36, 0.08), transparent 22rem),
        linear-gradient(180deg, #f7f2ea 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    main {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 32px 24px 48px;
    }}
    h1, h2, h3 {{
      margin: 0 0 12px;
      font-weight: 600;
    }}
    h1 {{
      font-size: 2rem;
      letter-spacing: -0.02em;
    }}
    h2 {{
      font-size: 1.15rem;
      color: var(--accent);
      margin-top: 28px;
    }}
    p, li {{
      line-height: 1.55;
    }}
    .card {{
      background: var(--paper);
      border: 1px solid var(--border);
      border-radius: 16px;
      box-shadow: 0 10px 24px rgba(48, 33, 18, 0.05);
      padding: 18px 20px;
      margin-top: 18px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 12px;
    }}
    .stat {{
      background: rgba(255, 255, 255, 0.75);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px 14px;
    }}
    .stat .label {{
      display: block;
      font-size: 0.82rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 4px;
    }}
    .stat .value {{
      font-size: 1.3rem;
      font-weight: 600;
    }}
    .mono {{
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
      font-size: 0.92rem;
    }}
    .text-block {{
      white-space: pre-wrap;
      font-size: 1.08rem;
      line-height: 1.9;
    }}
    .tok {{
      border-radius: 4px;
      padding: 0.04em 0.02em;
      transition: transform 120ms ease, box-shadow 120ms ease;
    }}
    .tok:hover {{
      transform: translateY(-1px);
      box-shadow: 0 4px 8px rgba(31, 24, 18, 0.14);
    }}
    .legend {{
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      align-items: center;
      color: var(--muted);
      font-size: 0.92rem;
    }}
    .chip {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }}
    .swatch {{
      width: 32px;
      height: 14px;
      border-radius: 999px;
      border: 1px solid rgba(31, 24, 18, 0.08);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.94rem;
    }}
    th, td {{
      text-align: left;
      padding: 8px 10px;
      border-bottom: 1px solid var(--border);
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-weight: 600;
    }}
    .two-col {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
    }}
    @media (max-width: 640px) {{
      main {{
        padding: 20px 14px 32px;
      }}
      .text-block {{
        font-size: 1rem;
        line-height: 1.75;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>{html.escape(title)}</h1>
    <p>Metric: <strong>{html.escape(metric)}</strong>. {html.escape(metric_description(metric))}</p>

    <div class="card">
      <div class="grid">
        <div class="stat"><span class="label">Validation file</span><span class="value mono">{html.escape(str(source_file))}</span></div>
        <div class="stat"><span class="label">Step</span><span class="value">{html.escape(str(step_value))}</span></div>
        <div class="stat"><span class="label">Selected row</span><span class="value">{selected_row}</span></div>
        <div class="stat"><span class="label">Response tokens</span><span class="value">{len(tokens)}</span></div>
        <div class="stat"><span class="label">Mean advantage</span><span class="value">{float(np.mean(advantage_values)):.4f}</span></div>
        <div class="stat"><span class="label">Mean reverse KL</span><span class="value">{float(np.mean(reverse_kl_values)):.4f}</span></div>
        <div class="stat"><span class="label">Positive-advantage fraction</span><span class="value">{positive_frac:.3f}</span></div>
        <div class="stat"><span class="label">Negative-advantage fraction</span><span class="value">{negative_frac:.3f}</span></div>
      </div>
    </div>

    <div class="card">
      <div class="legend">
        <span class="chip"><span class="swatch" style="background:rgba(191, 51, 44, 0.75)"></span>teacher favors token</span>
        <span class="chip"><span class="swatch" style="background:rgba(46, 109, 164, 0.75)"></span>student overcommits token</span>
        <span class="chip">color scale = 95th percentile = {scale:.4f}</span>
      </div>
    </div>

    <h2>Prompt</h2>
    <div class="card text-block">{html.escape(prompt)}</div>

    <h2>Response Heatmap</h2>
    <div class="card text-block">{heatmap_html}</div>

    <h2>Position Summary</h2>
    <div class="card">
      <table>
        <thead>
          <tr><th>Response span</th><th>Mean {html.escape(metric)}</th></tr>
        </thead>
        <tbody>
          {"".join(position_rows)}
        </tbody>
      </table>
    </div>

    <h2>Most Positive Tokens</h2>
    <div class="card">
      <table>
        <thead>
          <tr><th>Idx</th><th>Token</th><th>{html.escape(metric)}</th><th>Advantage</th><th>Student logp</th><th>Teacher logp</th></tr>
        </thead>
        <tbody>
          {best_rows}
        </tbody>
      </table>
    </div>

    <h2>Most Negative Tokens</h2>
    <div class="card">
      <table>
        <thead>
          <tr><th>Idx</th><th>Token</th><th>{html.escape(metric)}</th><th>Advantage</th><th>Student logp</th><th>Teacher logp</th></tr>
        </thead>
        <tbody>
          {worst_rows}
        </tbody>
      </table>
    </div>

    <div class="two-col">
      <div>
        <h2>Model Paths</h2>
        <div class="card">
          <table>
            <tbody>
              <tr><th>Student</th><td class="mono">{html.escape(student_model)}</td></tr>
              <tr><th>Teacher</th><td class="mono">{html.escape(teacher_model)}</td></tr>
            </tbody>
          </table>
        </div>
      </div>
      <div>
        <h2>Row Metadata</h2>
        <div class="card">
          <table>
            <tbody>
              {"".join(meta_rows) if meta_rows else "<tr><td>No extra metadata</td></tr>"}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()

    validation_file = resolve_validation_file(args.validation_source, args.step)
    rows = load_jsonl(validation_file)
    selected_row_index, row = select_sample(rows, args.select, args.row, args.contains)
    prompt, output = extract_prompt_and_output(row)

    step_value = row.get("step", validation_file.stem)

    print(f"[1/4] validation file: {validation_file}")
    print(f"[1/4] selected row: {selected_row_index}, step: {step_value}, metric: {args.metric}")

    print(f"[2/4] scoring student model: {args.student_model}")
    student_model, student_tokenizer, student_device = load_model_and_tokenizer(
        args.student_model, args.student_device, args.trust_remote_code
    )
    student_scores = score_output_tokens(student_model, student_tokenizer, student_device, prompt, output)

    try:
        import torch

        del student_model
        torch.cuda.empty_cache()
    except Exception:
        pass

    print(f"[3/4] scoring teacher model: {args.teacher_model}")
    teacher_model, teacher_tokenizer, teacher_device = load_model_and_tokenizer(
        args.teacher_model, args.teacher_device, args.trust_remote_code
    )
    teacher_scores = score_output_tokens(teacher_model, teacher_tokenizer, teacher_device, prompt, output)

    try:
        import torch

        del teacher_model
        torch.cuda.empty_cache()
    except Exception:
        pass

    tokens = build_tokens(
        output_text=output,
        offsets=student_scores["offsets"],
        student_scores=student_scores,
        teacher_scores=teacher_scores,
    )

    out_html = Path(args.out_html) if args.out_html else default_output_path(validation_file, args.metric, selected_row_index, step_value)
    html_doc = render_html(
        source_file=validation_file,
        step_value=step_value,
        selected_row=selected_row_index,
        row=row,
        prompt=prompt,
        output=output,
        student_model=args.student_model,
        teacher_model=args.teacher_model,
        metric=args.metric,
        tokens=tokens,
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_doc)

    print(f"[4/4] wrote {out_html}")


if __name__ == "__main__":
    main()
