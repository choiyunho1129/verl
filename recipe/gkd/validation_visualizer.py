from __future__ import annotations

import html
import json
import math
import os
from typing import Any

import numpy as np


DEFAULT_METRIC = os.getenv("VERL_GKD_VAL_VIS_METRIC", "advantage")
DEFAULT_SELECT = os.getenv("VERL_GKD_VAL_VIS_SELECT", "longest")
DEFAULT_LIMIT = int(os.getenv("VERL_GKD_VAL_VIS_LIMIT", "8"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def build_validation_feedback_record(
    *,
    tokenizer,
    prompt_text: str,
    response_token_ids: list[int],
    student_logprobs: list[float],
    teacher_logprobs: list[float],
    teacher_top1_token_ids: list[int] | None = None,
    teacher_top1_logprobs: list[float] | None = None,
    score: float | None,
    sample_index: int,
    uid: str | None = None,
    data_source: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if len(response_token_ids) != len(student_logprobs) or len(response_token_ids) != len(teacher_logprobs):
        raise ValueError(
            "response ids and logprob lengths must match: "
            f"{len(response_token_ids)=}, {len(student_logprobs)=}, {len(teacher_logprobs)=}"
        )

    kept_ids = [int(token_id) for token_id in response_token_ids]
    kept_student = [float(student_logprob) for student_logprob in student_logprobs]
    kept_teacher = [float(teacher_logprob) for teacher_logprob in teacher_logprobs]
    if teacher_top1_token_ids is not None and len(teacher_top1_token_ids) != len(kept_ids):
        raise ValueError(
            "teacher_top1_token_ids length must match response ids length: "
            f"{len(teacher_top1_token_ids)=}, {len(kept_ids)=}"
        )
    if teacher_top1_logprobs is not None and len(teacher_top1_logprobs) != len(kept_ids):
        raise ValueError(
            "teacher_top1_logprobs length must match response ids length: "
            f"{len(teacher_top1_logprobs)=}, {len(kept_ids)=}"
        )
    teacher_top1_ids = [int(token_id) for token_id in teacher_top1_token_ids] if teacher_top1_token_ids is not None else None
    teacher_top1_logps = (
        [float(logprob) for logprob in teacher_top1_logprobs] if teacher_top1_logprobs is not None else None
    )
    teacher_top1_texts = (
        [tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False) for token_id in teacher_top1_ids]
        if teacher_top1_ids is not None
        else None
    )

    output_text = tokenizer.decode(
        kept_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    offsets = None
    token_texts = None
    if kept_ids:
        try:
            encoded = tokenizer(output_text, add_special_tokens=False, return_offsets_mapping=True)
            offsets = encoded.get("offset_mapping")
            if offsets is None or len(offsets) != len(kept_ids):
                offsets = None
        except Exception:
            offsets = None

        if offsets is None:
            token_texts = [
                tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                for token_id in kept_ids
            ]

    advantages = [teacher - student for teacher, student in zip(kept_teacher, kept_student)]
    reverse_kls = [student - teacher for teacher, student in zip(kept_teacher, kept_student)]

    record = {
        "sample_index": sample_index,
        "uid": uid,
        "data_source": data_source,
        "score": score,
        "prompt_text": prompt_text,
        "output_text": output_text,
        "response_length": len(kept_ids),
        "token_ids": kept_ids,
        "offsets": offsets,
        "token_texts": token_texts,
        "student_logprobs": kept_student,
        "teacher_logprobs": kept_teacher,
        "teacher_top1_token_ids": teacher_top1_ids,
        "teacher_top1_logprobs": teacher_top1_logps,
        "teacher_top1_token_texts": teacher_top1_texts,
        "advantage": advantages,
        "reverse_kl": reverse_kls,
        "mean_advantage": float(np.mean(advantages)) if advantages else 0.0,
        "mean_reverse_kl": float(np.mean(reverse_kls)) if reverse_kls else 0.0,
    }
    if extra:
        record["extra"] = _json_safe(extra)
    return record


def dump_validation_feedback(
    *,
    dump_root: str,
    step: int,
    records: list[dict[str, Any]],
    metric: str | None = None,
    select: str | None = None,
    limit: int | None = None,
) -> None:
    if not records:
        return

    metric = metric or DEFAULT_METRIC
    select = select or DEFAULT_SELECT
    limit = DEFAULT_LIMIT if limit is None else limit

    step_dir = os.path.join(dump_root, "token_feedback", f"step_{step}")
    os.makedirs(step_dir, exist_ok=True)

    jsonl_path = os.path.join(step_dir, "records.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    html_path = os.path.join(step_dir, "index.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(_render_step_html(step=step, records=records, metric=metric, select=select, limit=limit))


def _select_records(records: list[dict[str, Any]], select: str, limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or select == "all":
        return list(records)

    if select == "first":
        return records[:limit]
    if select == "longest":
        return sorted(records, key=lambda record: record.get("response_length", 0), reverse=True)[:limit]
    if select == "best_score":
        return sorted(records, key=lambda record: _score_or_neg_inf(record), reverse=True)[:limit]
    if select == "worst_score":
        return sorted(records, key=lambda record: _score_or_pos_inf(record))[:limit]
    return records[:limit]


def _score_or_neg_inf(record: dict[str, Any]) -> float:
    score = record.get("score")
    return float(score) if score is not None else float("-inf")


def _score_or_pos_inf(record: dict[str, Any]) -> float:
    score = record.get("score")
    return float(score) if score is not None else float("inf")


def _metric_description(metric: str) -> str:
    descriptions = {
        "advantage": "teacher_logprob - student_logprob. Positive values are reinforced by the current GKD update.",
        "reverse_kl": "student_logprob - teacher_logprob. This is the signed token gap behind val/reverse_kl.",
        "student_logprob": "student log-probability of the sampled token.",
        "teacher_logprob": "teacher log-probability of the sampled token.",
    }
    return descriptions.get(metric, metric)


def _record_metric_values(record: dict[str, Any], metric: str) -> list[float]:
    if metric in record:
        values = record[metric]
    else:
        raise ValueError(f"Metric {metric} not found in record.")
    return [float(value) for value in values]


def _render_step_html(step: int, records: list[dict[str, Any]], metric: str, select: str, limit: int) -> str:
    selected_records = _select_records(records, select=select, limit=limit)
    all_lengths = np.asarray([record.get("response_length", 0) for record in records], dtype=float)
    all_advantages = np.asarray([record.get("mean_advantage", 0.0) for record in records], dtype=float)
    all_reverse_kls = np.asarray([record.get("mean_reverse_kl", 0.0) for record in records], dtype=float)
    scored = [float(record["score"]) for record in records if record.get("score") is not None]
    mean_score = float(np.mean(scored)) if scored else None
    mean_score_text = f"{mean_score:.4f}" if mean_score is not None else "n/a"

    detail_sections = [_render_record_detail(record, metric=metric) for record in selected_records]
    summary_rows = [_render_summary_row(record) for record in records]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Validation token feedback step {step}</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --paper: #fffdfa;
      --ink: #1f1812;
      --muted: #6f6256;
      --border: #d8cfc2;
      --accent: #8f2f24;
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
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 22px 44px;
    }}
    h1, h2, h3 {{
      margin: 0 0 10px;
      font-weight: 600;
    }}
    h1 {{
      font-size: 2rem;
      letter-spacing: -0.02em;
    }}
    h2 {{
      font-size: 1.15rem;
      color: var(--accent);
      margin-top: 24px;
    }}
    p {{
      line-height: 1.55;
    }}
    .card {{
      background: var(--paper);
      border: 1px solid var(--border);
      border-radius: 16px;
      box-shadow: 0 10px 24px rgba(48, 33, 18, 0.05);
      padding: 18px 20px;
      margin-top: 16px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    }}
    .stat {{
      background: rgba(255, 255, 255, 0.7);
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
      font-size: 1.28rem;
      font-weight: 600;
    }}
    .mono {{
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
      font-size: 0.92rem;
    }}
    .text-block {{
      white-space: pre-wrap;
      font-size: 1.03rem;
      line-height: 1.82;
    }}
    .aux-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
      margin: 12px 0 16px;
    }}
    .aux-block {{
      background: rgba(255, 255, 255, 0.7);
      border: 1px dashed var(--border);
      border-radius: 12px;
      padding: 12px 14px;
      margin: 12px 0 16px;
    }}
    .aux-block h4 {{
      margin: 0 0 8px;
      color: var(--muted);
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .tok {{
      border-radius: 4px;
      padding: 0.04em 0.02em;
    }}
    .token-strip {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px 4px;
      align-items: flex-start;
      margin-top: 8px;
    }}
    .token-chip {{
      display: inline-flex;
      flex-direction: column;
      gap: 2px;
      max-width: 12rem;
      padding: 4px 6px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.78);
    }}
    .token-chip.diff {{
      border-color: rgba(143, 47, 36, 0.38);
      background: rgba(143, 47, 36, 0.08);
    }}
    .token-chip .chip-label {{
      font-size: 0.68rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .token-chip .chip-text {{
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
      font-size: 0.82rem;
      white-space: pre-wrap;
      line-height: 1.35;
      word-break: break-word;
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
    @media (max-width: 640px) {{
      main {{
        padding: 18px 12px 30px;
      }}
      .text-block {{
        font-size: 0.98rem;
        line-height: 1.74;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>Validation token feedback: step {step}</h1>
    <p>Inline colors use <strong>{html.escape(metric)}</strong>. {_metric_description(metric)}</p>

    <div class="card">
      <div class="grid">
        <div class="stat"><span class="label">Validation samples</span><span class="value">{len(records)}</span></div>
        <div class="stat"><span class="label">Mean response length</span><span class="value">{float(np.mean(all_lengths)):.2f}</span></div>
        <div class="stat"><span class="label">Mean advantage</span><span class="value">{float(np.mean(all_advantages)):.4f}</span></div>
        <div class="stat"><span class="label">Mean reverse KL</span><span class="value">{float(np.mean(all_reverse_kls)):.4f}</span></div>
        <div class="stat"><span class="label">Mean score</span><span class="value">{mean_score_text}</span></div>
        <div class="stat"><span class="label">HTML subset</span><span class="value mono">{html.escape(select)} / {limit}</span></div>
      </div>
    </div>

    <h2>Sample summary</h2>
    <div class="card">
      <table>
        <thead>
          <tr>
            <th>sample_index</th>
            <th>uid</th>
            <th>data_source</th>
            <th>score</th>
            <th>response_len</th>
            <th>mean_advantage</th>
            <th>mean_reverse_kl</th>
          </tr>
        </thead>
        <tbody>
          {"".join(summary_rows)}
        </tbody>
      </table>
    </div>

    <h2>Detailed heatmaps</h2>
    {"".join(detail_sections)}
  </main>
</body>
</html>
"""


def _render_summary_row(record: dict[str, Any]) -> str:
    uid = record.get("uid") or ""
    data_source = record.get("data_source") or ""
    score = record.get("score")
    score_text = f"{float(score):.4f}" if score is not None else "n/a"
    return (
        "<tr>"
        f"<td>{record.get('sample_index')}</td>"
        f"<td class='mono'>{html.escape(str(uid))}</td>"
        f"<td>{html.escape(str(data_source))}</td>"
        f"<td>{score_text}</td>"
        f"<td>{record.get('response_length', 0)}</td>"
        f"<td>{record.get('mean_advantage', 0.0):.4f}</td>"
        f"<td>{record.get('mean_reverse_kl', 0.0):.4f}</td>"
        "</tr>"
    )


def _render_record_detail(record: dict[str, Any], metric: str) -> str:
    values = _record_metric_values(record, metric)
    scale = _percentile_scale(values, signed=metric in {"advantage", "reverse_kl"})
    prompt_text = record.get("prompt_text", "")
    score = record.get("score")
    score_text = f"{float(score):.4f}" if score is not None else "n/a"
    extra_sections = _render_extra_sections(record)

    return f"""
    <div class="card">
      <h3>sample {record.get("sample_index")} | score={score_text} | len={record.get("response_length", 0)}</h3>
      <p class="mono">uid={html.escape(str(record.get("uid") or ""))} | data_source={html.escape(str(record.get("data_source") or ""))} | color_scale={scale:.4f}</p>
      <h3>Prompt</h3>
      <div class="text-block">{html.escape(prompt_text)}</div>
      {extra_sections}
      <h3>Response</h3>
      <div class="text-block">{_render_response_tokens(record, metric=metric, scale=scale)}</div>
      {_render_teacher_top1_tokens(record)}
    </div>
    """


def _format_extra_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}" if math.isfinite(value) else str(value)
    return str(value)


def _render_extra_sections(record: dict[str, Any]) -> str:
    extra = record.get("extra")
    if not isinstance(extra, dict) or not extra:
        return ""

    sections = []

    stat_pairs = []
    for key, label in (
        ("subject", "Subject"),
        ("ability", "Ability"),
        ("level", "Level"),
        ("student_score", "Student score"),
        ("teacher_preview_score", "Teacher preview score"),
        ("student_mean_logprob", "Student mean logprob"),
        ("teacher_mean_logprob_on_student", "Teacher mean logprob"),
    ):
        value = extra.get(key)
        if value not in (None, ""):
            stat_pairs.append(
                "<div class='stat'>"
                f"<span class='label'>{html.escape(label)}</span>"
                f"<span class='value'>{html.escape(_format_extra_value(value))}</span>"
                "</div>"
            )

    if stat_pairs:
        sections.append(f"<div class='aux-grid'>{''.join(stat_pairs)}</div>")

    ground_truth = extra.get("ground_truth")
    if ground_truth not in (None, ""):
        sections.append(
            "<div class='aux-block'>"
            "<h4>Ground Truth</h4>"
            f"<div class='text-block'>{html.escape(str(ground_truth))}</div>"
            "</div>"
        )

    teacher_preview_text = extra.get("teacher_preview_text")
    if teacher_preview_text not in (None, ""):
        sections.append(
            "<div class='aux-block'>"
            "<h4>Teacher Preview</h4>"
            f"<div class='text-block'>{html.escape(str(teacher_preview_text))}</div>"
            "</div>"
        )

    return "".join(sections)


def _percentile_scale(values: list[float], signed: bool) -> float:
    if not values:
        return 1.0
    arr = np.asarray(values, dtype=float)
    if signed:
        arr = np.abs(arr)
    scale = float(np.percentile(arr, 95))
    if not math.isfinite(scale) or scale <= 0:
        scale = float(np.max(arr)) if arr.size else 1.0
    return scale if scale > 0 else 1.0


def _rgba_for_metric(value: float, scale: float, signed: bool) -> str:
    intensity = max(0.0, min(1.0, abs(value) / max(scale, 1e-8)))
    alpha = 0.10 + 0.75 * intensity
    if signed:
        if value >= 0:
            return f"rgba(191, 51, 44, {alpha:.3f})"
        return f"rgba(46, 109, 164, {alpha:.3f})"
    return f"rgba(191, 51, 44, {alpha:.3f})"


def _render_response_tokens(record: dict[str, Any], metric: str, scale: float) -> str:
    output_text = record.get("output_text", "")
    values = _record_metric_values(record, metric)
    student_logprobs = record.get("student_logprobs", [])
    teacher_logprobs = record.get("teacher_logprobs", [])
    teacher_top1_texts = record.get("teacher_top1_token_texts") or []
    teacher_top1_logprobs = record.get("teacher_top1_logprobs") or []
    teacher_top1_token_ids = record.get("teacher_top1_token_ids") or []
    signed = metric in {"advantage", "reverse_kl"}

    offsets = record.get("offsets")
    if offsets:
        fragments = []
        cursor = 0
        for idx, (offset, value, student_logprob, teacher_logprob) in enumerate(
            zip(offsets, values, student_logprobs, teacher_logprobs)
        ):
            start, end = int(offset[0]), int(offset[1])
            if start > cursor:
                fragments.append(html.escape(output_text[cursor:start]))
            token_text = output_text[start:end]
            title = html.escape(
                " | ".join(
                    [
                        f"idx={idx}",
                        f"text={token_text!r}",
                        f"student_logprob={float(student_logprob):.4f}",
                        f"teacher_logprob={float(teacher_logprob):.4f}",
                        (
                            f"teacher_top1={teacher_top1_texts[idx]!r}"
                            if idx < len(teacher_top1_texts)
                            else "teacher_top1=n/a"
                        ),
                        (
                            f"teacher_top1_id={int(teacher_top1_token_ids[idx])}"
                            if idx < len(teacher_top1_token_ids)
                            else "teacher_top1_id=n/a"
                        ),
                        (
                            f"teacher_top1_logprob={float(teacher_top1_logprobs[idx]):.4f}"
                            if idx < len(teacher_top1_logprobs)
                            else "teacher_top1_logprob=n/a"
                        ),
                        f"advantage={float(record['advantage'][idx]):.4f}",
                        f"reverse_kl={float(record['reverse_kl'][idx]):.4f}",
                    ]
                ),
                quote=True,
            )
            bg = _rgba_for_metric(float(value), scale=scale, signed=signed)
            fragments.append(
                f'<span class="tok" style="background:{bg}" title="{title}">{html.escape(token_text) if token_text else "&nbsp;"}</span>'
            )
            cursor = end
        if cursor < len(output_text):
            fragments.append(html.escape(output_text[cursor:]))
        return "".join(fragments)

    token_texts = record.get("token_texts") or []
    if not token_texts:
        return html.escape(output_text)

    pieces = []
    for idx, (token_text, value, student_logprob, teacher_logprob) in enumerate(
        zip(token_texts, values, student_logprobs, teacher_logprobs)
    ):
        title = html.escape(
            " | ".join(
                [
                    f"idx={idx}",
                    f"text={token_text!r}",
                    f"student_logprob={float(student_logprob):.4f}",
                    f"teacher_logprob={float(teacher_logprob):.4f}",
                    (
                        f"teacher_top1={teacher_top1_texts[idx]!r}"
                        if idx < len(teacher_top1_texts)
                        else "teacher_top1=n/a"
                    ),
                    (
                        f"teacher_top1_id={int(teacher_top1_token_ids[idx])}"
                        if idx < len(teacher_top1_token_ids)
                        else "teacher_top1_id=n/a"
                    ),
                    (
                        f"teacher_top1_logprob={float(teacher_top1_logprobs[idx]):.4f}"
                        if idx < len(teacher_top1_logprobs)
                        else "teacher_top1_logprob=n/a"
                    ),
                    f"advantage={float(record['advantage'][idx]):.4f}",
                    f"reverse_kl={float(record['reverse_kl'][idx]):.4f}",
                ]
            ),
            quote=True,
        )
        bg = _rgba_for_metric(float(value), scale=scale, signed=signed)
        pieces.append(
            f'<span class="tok" style="background:{bg}" title="{title}">{html.escape(token_text) if token_text else "&nbsp;"}</span>'
        )
    return "".join(pieces)


def _render_teacher_top1_tokens(record: dict[str, Any]) -> str:
    teacher_top1_texts = record.get("teacher_top1_token_texts") or []
    if not teacher_top1_texts:
        return ""

    token_ids = record.get("token_ids") or []
    top1_ids = record.get("teacher_top1_token_ids") or []
    top1_logprobs = record.get("teacher_top1_logprobs") or []

    chips = []
    for idx, token_text in enumerate(teacher_top1_texts):
        sampled_token_id = token_ids[idx] if idx < len(token_ids) else None
        teacher_top1_id = top1_ids[idx] if idx < len(top1_ids) else None
        differs = sampled_token_id is not None and teacher_top1_id is not None and sampled_token_id != teacher_top1_id
        classes = "token-chip diff" if differs else "token-chip"
        title = html.escape(
            " | ".join(
                [
                    f"idx={idx}",
                    (
                        f"teacher_top1_id={int(teacher_top1_id)}"
                        if teacher_top1_id is not None
                        else "teacher_top1_id=n/a"
                    ),
                    (
                        f"teacher_top1_logprob={float(top1_logprobs[idx]):.4f}"
                        if idx < len(top1_logprobs)
                        else "teacher_top1_logprob=n/a"
                    ),
                    (
                        f"matches_sampled={not differs}"
                        if sampled_token_id is not None and teacher_top1_id is not None
                        else "matches_sampled=n/a"
                    ),
                ]
            ),
            quote=True,
        )
        chips.append(
            f"<span class='{classes}' title=\"{title}\">"
            f"<span class='chip-label'>t{idx}</span>"
            f"<span class='chip-text'>{html.escape(token_text) if token_text else '&nbsp;'}</span>"
            "</span>"
        )

    return (
        "<h3>Teacher Top-1 By Position</h3>"
        "<div class='token-strip'>"
        + "".join(chips)
        + "</div>"
    )
