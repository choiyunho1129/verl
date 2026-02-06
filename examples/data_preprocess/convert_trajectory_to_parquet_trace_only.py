import argparse
import json
from pathlib import Path

import pandas as pd
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig



def make_prompt(trajectory: str) -> tuple[str, str]:
    system_prompt = (
        "You are a math solution critic. Critique the solution trace based only on the trace itself. "
        "Identify specific logical errors or confirm the reasoning. Provide constructive feedback but do not give the direct answer."
    )
    user_prompt = f"Model Solution Trace:\n{trajectory}\n\n"
    return system_prompt, user_prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="/data1/home/yunhochoi/verl/data/llama_3b_instruct_trajectories_test.jsonl",
        help="Path to the trajectory JSONL input.",
    )
    parser.add_argument(
        "--output-parquet",
        type=str,
        default="/data1/home/yunhochoi/verl/data/test_critique_llama3b_trace_only.parquet",
        help="Destination parquet file with chat-formatted prompts.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default="/data1/home/yunhochoi/verl/data/test_critique_llama3b_trace_only.jsonl",
        help="Destination JSONL file with chat-formatted prompts.",
    )
    parser.add_argument("--data-source", type=str, default="critique_variants")
    args = parser.parse_args()

    rows = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            item = json.loads(line)

            system_prompt, user_prompt = make_prompt(trajectory=item["trajectory"])

            reward_meta = {
                "original_question": item["question"],
                "original_trajectory": item["trajectory"],
                "variants": item.get("variants", []),
            }
            ground_truth = json.dumps(reward_meta)

            rows.append(
                {
                    "data_source": args.data_source,
                    "prompt": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "response": "",
                    "reward_model": {"style": "rule", "ground_truth": ground_truth},
                    "reward_model_data": ground_truth,
                }
            )

    Path(args.output_parquet).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.output_parquet)

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as out_f:
        for row in rows:
            out_f.write(json.dumps(row) + "\n")

    print(f"Saved {len(rows)} samples to {args.output_parquet}")
    print(f"Saved {len(rows)} samples to {args.output_jsonl}")


if __name__ == "__main__":
    main()
