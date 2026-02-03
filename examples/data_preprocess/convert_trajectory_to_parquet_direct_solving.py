import json
import pandas as pd
import argparse
from pathlib import Path


def make_prompt(question, trajectory, answer=None):
    system_prompt = (
        "Solve the problem step by step and put your final answer within \\boxed{}. "
        "You may use the student's solution as a hint."
    )
    user_prompt = (
        f"Problem: {question}\n\n"
        f"Student Solution:\n{trajectory}\n\n"
    )
    return system_prompt, user_prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="/data01/yunhochoi/verl/data/Qwen2.5_7b_instruct_trajectories_4.jsonl")
    parser.add_argument("--output", type=str, default="/data01/yunhochoi/verl/data/train_MATH3-5_w_student_trajectories_qwen7b.parquet")
    parser.add_argument("--data-source", type=str, default="HuggingFaceH4/MATH-500")
    args = parser.parse_args()

    data = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)

            # Prefer explicit question field; fall back to the first user message in prompt.
            question = item.get('question')
            if question is None:
                question = next(
                    (m.get('content') for m in item.get('prompt', []) if m.get('role') == 'user'),
                    None,
                )
            if question is None:
                raise KeyError("Neither 'question' field nor user message found in 'prompt'.")

            system_prompt, user_prompt = make_prompt(
                question=question,
                trajectory=item['trajectory'],
                answer=item.get('answer')
            )

            reward_meta = {
                "original_question": question,
                "original_trajectory": item['trajectory'],
                "variants": item.get('variants', [])  # [{'question':..., 'answer':...}, ...]
            }
            ground_truth = json.dumps(reward_meta)

            data.append({
                "data_source": args.data_source,
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],  # Chat Format
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
