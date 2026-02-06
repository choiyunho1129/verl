import json
import pandas as pd
import argparse
from pathlib import Path

def make_prompt(question, trajectory, answer):
    system_prompt = (
        "You are a math solution critic. Critique the solution trace based on the correct answer. "
        "Identify specific logical errors or confirm the reasoning. Provide constructive feedback but do not give the direct answer."
    )
    user_prompt = (
        f"User Question: {question}\n"
        f"Model Solution Trace:\n{trajectory}\n\n"
        f"Correct Answer: {answer}\n\n"
    )
    return system_prompt, user_prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="/data1/home/yunhochoi/verl/data/llama_3b_instruct_trajectories_test.jsonl")
    parser.add_argument("--output", type=str, default="/data1/home/yunhochoi/verl/data/train_critique_llama3b_w_answer.parquet")
    parser.add_argument("--output-jsonl", type=str, default="/data1/home/yunhochoi/verl/data/train_critique_llama3b_w_answer.jsonl")
    parser.add_argument("--data-source", type=str, default="critique_variants")
    args = parser.parse_args()

    data = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)

            system_prompt, user_prompt = make_prompt(
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
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ], # Chat Format
                "response": "",
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                # Optional duplicate for debugging/inspection
                "reward_model_data": ground_truth,
            })

    df = pd.DataFrame(data)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output)
    print(f"Saved {len(df)} samples to {args.output}")

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as out_f:
        for row in data:
            out_f.write(json.dumps(row) + "\n")
    print(f"Saved {len(data)} samples to {args.output_jsonl}")

if __name__ == "__main__":
    main()
