import json
import pandas as pd
import argparse
from pathlib import Path

def make_prompt(question, trajectory, answer):
    return (
        f"User Question: {question}\n"
        f"Correct Answer: {answer}\n\n"
        f"Model Solution Trace:\n{trajectory}\n\n"
        f"Task: Critique the solution trace above based on the correct answer. "
        f"Identify specific logical errors or confirm the reasoning. Provide constructive feedback."
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="/data1/home/yunhochoi/verl/data/llama_trajectories.jsonl")
    parser.add_argument("--output", type=str, default="/data1/home/yunhochoi/verl/data/train_critique.parquet")
    parser.add_argument("--data-source", type=str, default="critique_variants")
    args = parser.parse_args()

    data = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)

            prompt_text = make_prompt(
                question=item['question'],
                trajectory=item['trajectory'],
                answer=item['answer']
            )

            reward_meta = {
                "original_question": item['question'],
                "original_trajectory": item['trajectory'],
                "variants": item['variants']  # [{'question':..., 'answer':...}, ...]
            }
            ground_truth = json.dumps(reward_meta)

            data.append({
                "data_source": args.data_source,
                "prompt": [{"role": "user", "content": prompt_text}], # Chat Format
                "response": "",
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                # Optional duplicate for debugging/inspection
                "reward_model_data": ground_truth,
            })

    df = pd.DataFrame(data)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output)
    print(f"Saved {len(df)} samples to {args.output}")

if __name__ == "__main__":
    main()
