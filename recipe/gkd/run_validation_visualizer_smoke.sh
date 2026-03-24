#!/usr/bin/env bash
set -euo pipefail

# Re-render an already executed GKD validation visualization.
# This uses the saved demo records produced by a prior run under outputs/.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INPUT_JSONL="${REPO_ROOT}/outputs/gkd_validation_viz_smoke/token_feedback/step_0/records.jsonl"
OUTPUT_DIR="${REPO_ROOT}/outputs/gkd_validation_viz_smoke_rerender"

export REPO_ROOT INPUT_JSONL OUTPUT_DIR

PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python3 - <<'PY'
from pathlib import Path
import json
import os

from recipe.gkd.validation_visualizer import dump_validation_feedback

repo_root = Path(os.environ["REPO_ROOT"])
input_jsonl = Path(os.environ["INPUT_JSONL"])
output_dir = Path(os.environ["OUTPUT_DIR"])

with input_jsonl.open("r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

dump_validation_feedback(
    dump_root=str(output_dir),
    step=0,
    records=records,
    metric="advantage",
    select="first",
    limit=1,
)

print(f"Wrote {output_dir / 'token_feedback' / 'step_0' / 'index.html'}")
PY
