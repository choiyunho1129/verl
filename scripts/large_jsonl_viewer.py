#!/usr/bin/env python3
"""
Large JSONL viewer for very big files (tens of GB).

Key design goal:
- Never load the full dataset in memory.
- Build a compact line-offset index once, then support random access by line.

Usage examples:
  python scripts/large_jsonl_viewer.py /path/to/data.jsonl
  python scripts/large_jsonl_viewer.py /path/to/data.jsonl --show 120
  python scripts/large_jsonl_viewer.py /path/to/data.jsonl --page 500 20
  python scripts/large_jsonl_viewer.py /path/to/data.jsonl --index /tmp/data.lineidx
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import struct
import sys
import time
from array import array
from pathlib import Path
from typing import Any

INDEX_MAGIC = b"LIDX1\x00"
HEADER_STRUCT = struct.Struct("<QQQ")  # file_size, mtime_ns, num_lines


def clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(value, hi))


def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    x = float(n)
    for unit in units:
        if x < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(x)} {unit}"
            return f"{x:.2f} {unit}"
        x /= 1024
    return f"{n} B"


def default_index_path(src: Path) -> Path:
    return src.with_suffix(src.suffix + ".lineidx")


def save_index(index_path: Path, source_stat: os.stat_result, offsets: array) -> None:
    if offsets.typecode != "Q":
        raise RuntimeError(f"Expected offsets array typecode 'Q', got {offsets.typecode!r}")
    if offsets.itemsize != 8:
        raise RuntimeError(f"Expected 8-byte offsets, got itemsize={offsets.itemsize}")

    tmp_path = index_path.with_suffix(index_path.suffix + ".tmp")
    payload = offsets.tobytes()
    header = INDEX_MAGIC + HEADER_STRUCT.pack(
        source_stat.st_size,
        source_stat.st_mtime_ns,
        len(offsets),
    )

    with tmp_path.open("wb") as f:
        f.write(header)
        f.write(payload)
    os.replace(tmp_path, index_path)


def load_index_if_valid(index_path: Path, source_stat: os.stat_result) -> array | None:
    if not index_path.exists():
        return None

    try:
        with index_path.open("rb") as f:
            magic = f.read(len(INDEX_MAGIC))
            if magic != INDEX_MAGIC:
                return None

            header_bytes = f.read(HEADER_STRUCT.size)
            if len(header_bytes) != HEADER_STRUCT.size:
                return None

            idx_size, idx_mtime_ns, idx_num_lines = HEADER_STRUCT.unpack(header_bytes)
            if idx_size != source_stat.st_size or idx_mtime_ns != source_stat.st_mtime_ns:
                return None

            data = f.read()
            if len(data) != idx_num_lines * 8:
                return None

            offsets = array("Q")
            offsets.frombytes(data)
            if len(offsets) != idx_num_lines:
                return None
            return offsets
    except OSError:
        return None


def build_index(source_path: Path, index_path: Path, verbose: bool = True) -> array:
    start = time.time()
    offsets = array("Q")
    line_count = 0

    with source_path.open("rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            offsets.append(pos)
            line_count += 1
            if verbose and line_count % 100_000 == 0:
                elapsed = time.time() - start
                rate = int(line_count / elapsed) if elapsed > 0 else 0
                print(f"[index] lines={line_count:,} rate={rate:,}/s", file=sys.stderr)

    save_index(index_path, source_path.stat(), offsets)
    if verbose:
        elapsed = time.time() - start
        print(
            f"[index] done lines={line_count:,} elapsed={elapsed:.1f}s "
            f"index_size={human_bytes(index_path.stat().st_size)}",
            file=sys.stderr,
        )
    return offsets


def ensure_index(source_path: Path, index_path: Path, rebuild: bool = False, verbose: bool = True) -> array:
    source_stat = source_path.stat()
    if not rebuild:
        offsets = load_index_if_valid(index_path, source_stat)
        if offsets is not None:
            if verbose:
                print(
                    f"[index] using existing index {index_path} "
                    f"({len(offsets):,} lines, {human_bytes(index_path.stat().st_size)})",
                    file=sys.stderr,
                )
            return offsets
    return build_index(source_path, index_path=index_path, verbose=verbose)


def clip(text: str, max_chars: int) -> str:
    text = text.replace("\n", " ").replace("\r", " ")
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def parse_json_line(raw: str) -> tuple[Any | None, str | None]:
    try:
        return json.loads(raw), None
    except json.JSONDecodeError as e:
        return None, f"JSONDecodeError: {e}"


def first_non_empty_str(obj: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        v = obj.get(key)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def message_preview(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for m in messages:
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        if isinstance(content, str) and content.strip():
            role = m.get("role", "?")
            return f"[{role}] {content}"
    return ""


class JsonlViewer:
    def __init__(
        self,
        source_path: Path,
        offsets: array,
        preview_chars: int = 140,
        show_chars: int = 6_000,
    ):
        self.source_path = source_path
        self.offsets = offsets
        self.num_lines = len(offsets)
        self.preview_chars = preview_chars
        self.show_chars = show_chars
        self.cursor = 1 if self.num_lines > 0 else 0
        self.fh = source_path.open("rb")

    def close(self) -> None:
        self.fh.close()

    def _check_line_no(self, line_no: int) -> None:
        if line_no < 1 or line_no > self.num_lines:
            raise IndexError(f"line number out of range: {line_no} (valid: 1..{self.num_lines})")

    def raw_line(self, line_no: int) -> str:
        self._check_line_no(line_no)
        self.fh.seek(self.offsets[line_no - 1])
        return self.fh.readline().decode("utf-8", errors="replace").rstrip("\n")

    def read_record(self, line_no: int) -> tuple[Any | None, str | None, str]:
        raw = self.raw_line(line_no)
        obj, err = parse_json_line(raw)
        return obj, err, raw

    def _summary_fields(self, obj: Any, raw: str) -> tuple[str, str, str]:
        if not isinstance(obj, dict):
            return "", "", clip(raw, self.preview_chars)

        rec_id = first_non_empty_str(obj, ["id", "sample_id", "uid"])
        source = first_non_empty_str(obj, ["dataset_source", "dataset", "source_repo", "source"])

        preview = first_non_empty_str(obj, ["prompt", "question", "input", "ground_truth"])
        if not preview:
            preview = message_preview(obj.get("messages"))
        if not preview:
            preview = json.dumps(obj, ensure_ascii=False)
        return clip(rec_id, 28), clip(source, 36), clip(preview, self.preview_chars)

    def print_stats(self) -> None:
        s = self.source_path.stat()
        print(f"File:        {self.source_path}")
        print(f"File size:   {human_bytes(s.st_size)} ({s.st_size:,} bytes)")
        print(f"Line count:  {self.num_lines:,}")
        print(f"Cursor:      {self.cursor}")

    def show_line(self, line_no: int, raw: bool = False, max_chars: int | None = None) -> None:
        self._check_line_no(line_no)
        obj, err, line = self.read_record(line_no)
        self.cursor = line_no
        max_chars = self.show_chars if max_chars is None else max_chars

        print(f"# line {line_no:,}/{self.num_lines:,}")
        if raw:
            out = line
        else:
            if err:
                out = f"{err}\n\n{line}"
            else:
                out = json.dumps(obj, ensure_ascii=False, indent=2)

        if max_chars > 0 and len(out) > max_chars:
            print(out[:max_chars])
            print(f"... [truncated {len(out) - max_chars:,} chars]")
        else:
            print(out)

    def page(self, start: int, count: int) -> None:
        if self.num_lines == 0:
            print("No lines.")
            return
        count = max(1, count)
        start = clamp(start, 1, self.num_lines)
        end = min(self.num_lines, start + count - 1)
        self.cursor = start

        print(f"Showing lines {start:,}..{end:,} of {self.num_lines:,}")
        print("-" * 120)
        print(f"{'LINE':>9}  {'ID':<28}  {'SOURCE':<36}  PREVIEW")
        print("-" * 120)
        for i in range(start, end + 1):
            obj, err, raw = self.read_record(i)
            if err:
                rec_id, source, preview = "", "PARSE_ERROR", clip(raw, self.preview_chars)
            else:
                rec_id, source, preview = self._summary_fields(obj, raw)
            print(f"{i:>9}  {rec_id:<28}  {source:<36}  {preview}")
        print("-" * 120)

    def goto(self, line_no: int) -> None:
        self._check_line_no(line_no)
        self.cursor = line_no
        print(f"Cursor set to line {line_no:,}")

    def find(self, query: str, regex: bool = False, ignore_case: bool = True, start: int | None = None) -> int | None:
        if self.num_lines == 0:
            return None
        if start is None:
            start = self.cursor + 1
        start = clamp(start, 1, self.num_lines)

        pattern = None
        query_cmp = query
        if regex:
            flags = re.IGNORECASE if ignore_case else 0
            pattern = re.compile(query, flags)
        elif ignore_case:
            query_cmp = query.lower()

        for i in range(start, self.num_lines + 1):
            raw = self.raw_line(i)
            if pattern:
                ok = pattern.search(raw) is not None
            else:
                hay = raw.lower() if ignore_case else raw
                ok = query_cmp in hay
            if ok:
                self.cursor = i
                return i
        return None


HELP_TEXT = """
Commands:
  help
  stats
  page [start] [count]       Show compact rows (default: current line, 10)
  show <line> [max_chars]    Show parsed JSON for one line
  raw <line> [max_chars]     Show raw JSONL line
  goto <line>                Move cursor
  next [count]               Shortcut for page cursor+1
  prev [count]               Shortcut for page cursor-count
  find <text>                Case-insensitive substring search from cursor+1
  findre <regex>             Regex search from cursor+1
  q | quit | exit
"""


def run_repl(viewer: JsonlViewer) -> None:
    print(f"Loaded: {viewer.source_path}")
    viewer.print_stats()
    print("Type 'help' for commands.")
    viewer.page(viewer.cursor, 5)

    while True:
        try:
            cmdline = input("jsonl-viewer> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not cmdline:
            continue

        try:
            parts = shlex.split(cmdline)
        except ValueError as e:
            print(f"Parse error: {e}")
            continue
        if not parts:
            continue

        cmd = parts[0].lower()
        args = parts[1:]

        try:
            if cmd in {"q", "quit", "exit"}:
                return
            if cmd == "help":
                print(HELP_TEXT.strip())
            elif cmd == "stats":
                viewer.print_stats()
            elif cmd == "page":
                start = viewer.cursor
                count = 10
                if len(args) >= 1:
                    start = int(args[0])
                if len(args) >= 2:
                    count = int(args[1])
                viewer.page(start, count)
            elif cmd == "show":
                if not args:
                    print("Usage: show <line> [max_chars]")
                    continue
                line = int(args[0])
                max_chars = int(args[1]) if len(args) >= 2 else None
                viewer.show_line(line, raw=False, max_chars=max_chars)
            elif cmd == "raw":
                if not args:
                    print("Usage: raw <line> [max_chars]")
                    continue
                line = int(args[0])
                max_chars = int(args[1]) if len(args) >= 2 else None
                viewer.show_line(line, raw=True, max_chars=max_chars)
            elif cmd == "goto":
                if not args:
                    print("Usage: goto <line>")
                    continue
                viewer.goto(int(args[0]))
            elif cmd == "next":
                count = int(args[0]) if args else 10
                viewer.page(clamp(viewer.cursor + 1, 1, viewer.num_lines), count)
            elif cmd == "prev":
                count = int(args[0]) if args else 10
                viewer.page(clamp(viewer.cursor - count, 1, viewer.num_lines), count)
            elif cmd == "find":
                if not args:
                    print("Usage: find <text>")
                    continue
                query = " ".join(args)
                found = viewer.find(query, regex=False, ignore_case=True)
                if found is None:
                    print("No match.")
                else:
                    print(f"Found at line {found:,}")
                    viewer.page(found, 1)
            elif cmd == "findre":
                if not args:
                    print("Usage: findre <regex>")
                    continue
                query = " ".join(args)
                found = viewer.find(query, regex=True, ignore_case=True)
                if found is None:
                    print("No match.")
                else:
                    print(f"Found at line {found:,}")
                    viewer.page(found, 1)
            else:
                print(f"Unknown command: {cmd}. Type 'help'.")
        except Exception as e:
            print(f"Error: {e}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Memory-safe JSONL viewer for very large files.")
    p.add_argument("file", type=Path, help="Path to JSONL file")
    p.add_argument("--index", type=Path, default=None, help="Path to sidecar index file")
    p.add_argument("--rebuild-index", action="store_true", help="Force index rebuild")
    p.add_argument("--no-index-log", action="store_true", help="Suppress index progress logs")

    p.add_argument("--preview-chars", type=int, default=140, help="Preview width for page view")
    p.add_argument("--show-chars", type=int, default=6000, help="Default max chars for show/raw output")

    p.add_argument("--show", type=int, default=None, help="Non-interactive: show this line and exit")
    p.add_argument(
        "--page",
        nargs=2,
        type=int,
        metavar=("START", "COUNT"),
        default=None,
        help="Non-interactive: show compact page and exit",
    )
    p.add_argument("--raw", action="store_true", help="With --show: print raw line instead of pretty JSON")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    source_path = args.file.expanduser().resolve()
    if not source_path.exists():
        print(f"File not found: {source_path}", file=sys.stderr)
        return 1
    if not source_path.is_file():
        print(f"Not a file: {source_path}", file=sys.stderr)
        return 1

    index_path = args.index.expanduser().resolve() if args.index else default_index_path(source_path)
    try:
        offsets = ensure_index(
            source_path=source_path,
            index_path=index_path,
            rebuild=args.rebuild_index,
            verbose=not args.no_index_log,
        )
    except OSError as e:
        print(f"Failed to build/load index {index_path}: {e}", file=sys.stderr)
        return 2

    viewer = JsonlViewer(
        source_path=source_path,
        offsets=offsets,
        preview_chars=args.preview_chars,
        show_chars=args.show_chars,
    )
    try:
        if args.show is not None:
            viewer.show_line(args.show, raw=args.raw, max_chars=args.show_chars)
            return 0
        if args.page is not None:
            start, count = args.page
            viewer.page(start, count)
            return 0
        run_repl(viewer)
        return 0
    finally:
        viewer.close()


if __name__ == "__main__":
    raise SystemExit(main())
