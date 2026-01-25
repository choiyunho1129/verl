import argparse
import json
from pathlib import Path

import pandas as pd
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig


def _verify_answer(solution: str, answer: str) -> float:
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ground_truth_boxed = "\\boxed{" + answer + "}"
    try:
        score, _ = verify_func([ground_truth_boxed], [solution])
        return float(score)
    except TimeoutException:
        return 0.0
    except Exception:
        return 0.0


def make_prompt(question: str, trajectory: str, verified_correct: bool) -> str:
    verification_text = "verified as correct" if verified_correct else "flagged as incorrect"
    return (
        "You are a math solution critic. Critique the solution trace based on the verification result.\n" 
        f"User Question: {question}\n\n"
        f"Model Solution Trace:\n{trajectory}\n\n"
        f"verification result: The solution was {verification_text}.\n"
        "Identify specific logical errors or confirm the solution trace. Provide constructive feedback but do not give the direct answer. "
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="/data01/yunhochoi/verl/data/llama3.2_trajectories_4.jsonl",
        help="Path to the trajectory JSONL input.",
    )
    parser.add_argument(
        "--output-parquet",
        type=str,
        default="/data01/yunhochoi/verl/data/train_critique_no_directanswer_3.2_4.parquet",
        help="Destination parquet file with chat-formatted prompts.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default="/data01/yunhochoi/verl/data/train_critique_no_directanswer_3.2_4.jsonl",
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

            verification_score = _verify_answer(item["trajectory"], item["answer"])
            verified_correct = bool(verification_score and verification_score > 0)

            prompt_text = make_prompt(
                question=item["question"],
                trajectory=item["trajectory"],
                verified_correct=verified_correct,
            )

            reward_meta = {
                "original_question": item["question"],
                "original_trajectory": item["trajectory"],
                "variants": item.get("variants", []),
                "verified_correct": verified_correct,
                "verification_score": float(verification_score or 0),
            }
            ground_truth = json.dumps(reward_meta)

            rows.append(
                {
                    "data_source": args.data_source,
                    "prompt": [{"role": "user", "content": prompt_text}],
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
