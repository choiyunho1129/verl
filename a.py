import json, os
from pathlib import Path

old = "deepscaler_train5500_validation2000_seed1"
new = "deepscaler_train4096_validation2048_seed1"
root = Path(os.environ["NEW_RUN"]) / "train_runs"

for path in list(root.glob("train_shard*/all_experiments.jsonl")) + list(root.glob("train_shard*/evaluation_results.jsonl")):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        row["dataset_name"] = new
        rows.append(row)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")