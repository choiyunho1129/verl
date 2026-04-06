#!/usr/bin/env python3
"""Streamlit viewer for very large JSONL files.

This app is designed for files that are too large to load in memory.
It reuses a sidecar line-offset index to support random-access line reads.
"""

from __future__ import annotations

import json
import re
import time
from array import array
from pathlib import Path
from typing import Any

import streamlit as st

try:
    from scripts.large_jsonl_viewer import (
        clip,
        default_index_path,
        ensure_index,
        first_non_empty_str,
        human_bytes,
        message_preview,
        parse_json_line,
    )
except ImportError:
    # Fallback when executed from within scripts/ directly.
    from large_jsonl_viewer import (  # type: ignore[no-redef]
        clip,
        default_index_path,
        ensure_index,
        first_non_empty_str,
        human_bytes,
        message_preview,
        parse_json_line,
    )


def split_think_trace(text: str) -> tuple[str, str]:
    if not text:
        return "", ""
    start_tag = "<think>"
    end_tag = "</think>"
    start = text.find(start_tag)
    end = text.find(end_tag)
    if start >= 0 and end > start:
        think = text[start + len(start_tag) : end].strip()
        final = text[end + len(end_tag) :].strip()
        return think, final
    if start >= 0:
        think = text[start + len(start_tag) :].strip()
        return think, ""
    return "", text.strip()


def extract_chat_parts(obj: Any) -> tuple[str, str, str]:
    question = ""
    assistant = ""
    if isinstance(obj, dict):
        msgs = obj.get("messages")
        if isinstance(msgs, list):
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                role = str(m.get("role", "")).strip().lower()
                content = m.get("content")
                if not isinstance(content, str):
                    continue
                if not question and role == "user":
                    question = content.strip()
                if not assistant and role == "assistant":
                    assistant = content
        if not question:
            question = first_non_empty_str(obj, ["prompt", "question", "input"]).strip()
    think, final = split_think_trace(assistant)
    return question, think, final


def summarize_record(
    obj: Any,
    raw: str,
    preview_chars: int,
    think_friendly: bool,
) -> dict[str, Any]:
    if not isinstance(obj, dict):
        return {
            "id": "",
            "source": "PARSE_ERROR",
            "question": "",
            "think_preview": clip(raw, preview_chars),
            "think_chars": 0,
            "final_chars": 0,
            "preview": clip(raw, preview_chars),
        }

    rec_id = first_non_empty_str(obj, ["id", "sample_id", "uid"])
    source = first_non_empty_str(obj, ["dataset_source", "dataset", "source_repo", "source"])
    question, think, final = extract_chat_parts(obj)

    preview = question
    if not preview:
        preview = first_non_empty_str(obj, ["prompt", "question", "input", "ground_truth"])
    if not preview:
        preview = message_preview(obj.get("messages"))
    if not preview:
        preview = json.dumps(obj, ensure_ascii=False)

    if think_friendly:
        return {
            "id": clip(rec_id, 40),
            "source": clip(source, 56),
            "question": clip(question or preview, preview_chars),
            "think_preview": clip(think, preview_chars),
            "think_chars": len(think),
            "final_chars": len(final),
        }
    return {
        "id": clip(rec_id, 40),
        "source": clip(source, 56),
        "preview": clip(preview, preview_chars),
        "think_chars": len(think),
        "final_chars": len(final),
    }


@st.cache_resource(show_spinner=False)
def load_offsets_cached(file_path: str, index_path: str, file_size: int, mtime_ns: int) -> array:
    _ = file_size, mtime_ns  # cache key inputs
    return ensure_index(Path(file_path), Path(index_path), rebuild=False, verbose=False)


def read_record(file_path: Path, offsets: array, line_no: int) -> tuple[Any | None, str | None, str]:
    with file_path.open("rb") as f:
        f.seek(offsets[line_no - 1])
        raw = f.readline().decode("utf-8", errors="replace").rstrip("\n")
    obj, err = parse_json_line(raw)
    return obj, err, raw


def read_page(file_path: Path, offsets: array, start: int, count: int, preview_chars: int) -> list[dict[str, Any]]:
    return read_page_with_mode(
        file_path=file_path,
        offsets=offsets,
        start=start,
        count=count,
        preview_chars=preview_chars,
        think_friendly=False,
    )


def read_page_with_mode(
    file_path: Path,
    offsets: array,
    start: int,
    count: int,
    preview_chars: int,
    think_friendly: bool,
) -> list[dict[str, Any]]:
    end = min(len(offsets), start + count - 1)
    rows: list[dict[str, Any]] = []
    with file_path.open("rb") as f:
        for line_no in range(start, end + 1):
            f.seek(offsets[line_no - 1])
            raw = f.readline().decode("utf-8", errors="replace").rstrip("\n")
            obj, err = parse_json_line(raw)
            row = summarize_record(obj, raw, preview_chars, think_friendly=think_friendly)
            if err:
                row["source"] = "PARSE_ERROR"
            row["line"] = line_no
            rows.append(row)
    return rows


def find_next(
    file_path: Path,
    offsets: array,
    query: str,
    start_line: int,
    regex: bool,
    ignore_case: bool,
    max_scan: int,
) -> tuple[int | None, int, float]:
    if not query:
        return None, 0, 0.0

    n = len(offsets)
    start_line = max(1, min(start_line, n))
    end_line = n if max_scan <= 0 else min(n, start_line + max_scan - 1)

    pattern = None
    query_cmp = query
    if regex:
        flags = re.IGNORECASE if ignore_case else 0
        pattern = re.compile(query, flags)
    elif ignore_case:
        query_cmp = query.lower()

    scanned = 0
    t0 = time.time()
    with file_path.open("rb") as f:
        for line_no in range(start_line, end_line + 1):
            f.seek(offsets[line_no - 1])
            raw = f.readline().decode("utf-8", errors="replace")
            scanned += 1

            if pattern is not None:
                ok = pattern.search(raw) is not None
            else:
                hay = raw.lower() if ignore_case else raw
                ok = query_cmp in hay

            if ok:
                return line_no, scanned, time.time() - t0

    return None, scanned, time.time() - t0


def main() -> None:
    st.set_page_config(page_title="Large JSONL Viewer", layout="wide")
    st.title("Large JSONL Viewer (Streamlit)")
    st.caption("Memory-safe viewer for very large JSONL files using line-offset indexing.")

    default_file = "/home/jongwonlim/verl/data/Dolci-Think-SFT-7B.math_only.readable.jsonl"
    file_path_str = st.sidebar.text_input("JSONL file path", value=default_file)

    file_path = Path(file_path_str).expanduser()
    default_index = str(default_index_path(file_path))
    index_path_str = st.sidebar.text_input("Index path (.lineidx)", value=default_index)
    index_path = Path(index_path_str).expanduser()

    preview_chars = st.sidebar.slider("Preview chars", min_value=60, max_value=300, value=140, step=10)
    page_size = st.sidebar.slider("Page size", min_value=5, max_value=200, value=20, step=5)
    show_chars = st.sidebar.slider("Max chars in detail view", min_value=500, max_value=50000, value=8000, step=500)
    think_friendly = st.sidebar.checkbox("Think-trace friendly mode", value=True)
    show_question_final_tabs = st.sidebar.checkbox("Show question/final tabs", value=False)
    think_show_chars = st.sidebar.slider(
        "Max chars in think/final tabs",
        min_value=1000,
        max_value=200000,
        value=40000,
        step=1000,
    )

    if not file_path.exists() or not file_path.is_file():
        st.error(f"File not found: `{file_path}`")
        st.stop()

    file_stat = file_path.stat()
    rebuild_clicked = st.sidebar.button("Rebuild index")
    if rebuild_clicked:
        with st.spinner("Rebuilding index..."):
            ensure_index(file_path, index_path, rebuild=True, verbose=False)
            load_offsets_cached.clear()
        st.sidebar.success("Index rebuilt.")

    with st.spinner("Loading index..."):
        offsets = load_offsets_cached(
            str(file_path.resolve()),
            str(index_path.resolve()),
            file_stat.st_size,
            file_stat.st_mtime_ns,
        )

    n_lines = len(offsets)
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("File size", human_bytes(file_stat.st_size))
    col_b.metric("Lines", f"{n_lines:,}")
    col_c.metric("Index size", human_bytes(index_path.stat().st_size) if index_path.exists() else "N/A")

    if "cursor_line" not in st.session_state:
        st.session_state.cursor_line = 1

    st.subheader("Navigation")
    nav1, nav2, nav3, nav4 = st.columns([2, 1, 1, 1])
    line_input = nav1.number_input(
        "Current line (1-based)",
        min_value=1,
        max_value=max(1, n_lines),
        value=int(st.session_state.cursor_line),
        step=1,
    )
    if nav2.button("Go"):
        st.session_state.cursor_line = int(line_input)
    if nav3.button("Prev"):
        st.session_state.cursor_line = max(1, int(st.session_state.cursor_line) - 1)
    if nav4.button("Next"):
        st.session_state.cursor_line = min(n_lines, int(st.session_state.cursor_line) + 1)

    st.subheader("Page View")
    p1, p2, p3 = st.columns([2, 2, 1])
    page_start = p1.number_input(
        "Start line",
        min_value=1,
        max_value=max(1, n_lines),
        value=max(1, int(st.session_state.cursor_line)),
        step=1,
        key="page_start_input",
    )
    page_count = p2.number_input("Count", min_value=1, max_value=1000, value=page_size, step=1)
    if p3.button("Load page"):
        st.session_state.cursor_line = int(page_start)

    with st.spinner("Reading page..."):
        rows = read_page_with_mode(
            file_path=file_path,
            offsets=offsets,
            start=int(page_start),
            count=int(page_count),
            preview_chars=preview_chars,
            think_friendly=think_friendly,
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.subheader("Find")
    f1, f2, f3 = st.columns([4, 1, 1])
    query = f1.text_input("Query", value="", placeholder="e.g. bookshelf OR OpenThoughts3")
    regex = f2.checkbox("Regex", value=False)
    ignore_case = f3.checkbox("Ignore case", value=True)

    f4, f5, f6 = st.columns([2, 2, 1])
    search_start = f4.number_input(
        "Search start line",
        min_value=1,
        max_value=max(1, n_lines),
        value=min(n_lines, int(st.session_state.cursor_line) + 1),
        step=1,
    )
    max_scan = f5.number_input(
        "Max lines to scan (0=all)",
        min_value=0,
        max_value=max(0, n_lines),
        value=200000,
        step=1000,
    )
    find_clicked = f6.button("Find next")

    if find_clicked:
        with st.spinner("Searching..."):
            found_line, scanned, elapsed = find_next(
                file_path=file_path,
                offsets=offsets,
                query=query,
                start_line=int(search_start),
                regex=bool(regex),
                ignore_case=bool(ignore_case),
                max_scan=int(max_scan),
            )
        if found_line is None:
            st.warning(f"No match. Scanned {scanned:,} lines in {elapsed:.2f}s.")
        else:
            st.success(f"Found at line {found_line:,}. Scanned {scanned:,} lines in {elapsed:.2f}s.")
            st.session_state.cursor_line = found_line

    st.subheader("Line Detail")
    d1, d2 = st.columns([2, 1])
    detail_line = d1.number_input(
        "Line to inspect",
        min_value=1,
        max_value=max(1, n_lines),
        value=int(st.session_state.cursor_line),
        step=1,
    )
    _ = d2.button("Load detail")

    obj, err, raw = read_record(file_path=file_path, offsets=offsets, line_no=int(detail_line))
    st.caption(f"line {int(detail_line):,} / {n_lines:,}")
    question, think, final = extract_chat_parts(obj)
    m1, m2 = st.columns(2)
    m1.metric("Think chars", f"{len(think):,}")
    m2.metric("Final chars", f"{len(final):,}")

    if think_friendly and show_question_final_tabs:
        t0, t1, t2, t3, t4 = st.tabs(["Think Trace", "Question", "Final", "Pretty JSON", "Raw JSONL"])
    elif think_friendly:
        t0, t3, t4 = st.tabs(["Think Trace", "Pretty JSON", "Raw JSONL"])
        t1 = t2 = None
    else:
        t1, t2 = st.tabs(["Pretty JSON", "Raw JSONL"])
        t0 = t3 = t4 = None

    if think_friendly and t0 is not None and t3 is not None and t4 is not None:
        with t0:
            if think:
                think_text = think if len(think) <= think_show_chars else think[:think_show_chars]
                st.text_area("assistant think trace", think_text, height=520, disabled=True)
                if len(think) > think_show_chars:
                    st.caption(f"truncated {len(think) - think_show_chars:,} chars")
            else:
                st.info("No <think>...</think> block found.")
        if show_question_final_tabs and t1 is not None and t2 is not None:
            with t1:
                q = question or "(empty user message)"
                q_show = q if len(q) <= show_chars else q[:show_chars]
                st.text_area("user question", q_show, height=220, disabled=True)
                if len(q) > show_chars:
                    st.caption(f"truncated {len(q) - show_chars:,} chars")
            with t2:
                if final:
                    f_show = final if len(final) <= think_show_chars else final[:think_show_chars]
                    st.text_area("assistant final", f_show, height=380, disabled=True)
                    if len(final) > think_show_chars:
                        st.caption(f"truncated {len(final) - think_show_chars:,} chars")
                else:
                    st.info("No post-think final section found.")
        with t3:
            if err:
                st.error(err)
                out = raw
            else:
                out = json.dumps(obj, ensure_ascii=False, indent=2)
            if len(out) > show_chars:
                st.code(out[:show_chars], language="json")
                st.caption(f"truncated {len(out) - show_chars:,} chars")
            else:
                st.code(out, language="json")
        with t4:
            if len(raw) > show_chars:
                st.code(raw[:show_chars], language="json")
                st.caption(f"truncated {len(raw) - show_chars:,} chars")
            else:
                st.code(raw, language="json")
    else:
        with t1:
            if err:
                st.error(err)
                out = raw
            else:
                out = json.dumps(obj, ensure_ascii=False, indent=2)
            if len(out) > show_chars:
                st.code(out[:show_chars], language="json")
                st.caption(f"truncated {len(out) - show_chars:,} chars")
            else:
                st.code(out, language="json")
        with t2:
            if len(raw) > show_chars:
                st.code(raw[:show_chars], language="json")
                st.caption(f"truncated {len(raw) - show_chars:,} chars")
            else:
                st.code(raw, language="json")


if __name__ == "__main__":
    main()
