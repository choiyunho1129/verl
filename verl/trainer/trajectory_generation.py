import argparse
import json
import sys
from pathlib import Path
from typing import List

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

def load_data(path: Path) -> List[dict]:
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return samples

def main(args):
    data_path = Path(args.data_path).expanduser()
    samples = load_data(data_path)
    print(f"Loaded {len(samples)} samples.")

    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    raw_prompts = []
    for s in samples:
        question = s.get("question") or s.get("problem") 
        messages = [{"role": "user", "content": question}]
        formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        raw_prompts.append(formatted_prompt)

    print("Generating trajectories...")
    outputs = llm.generate(raw_prompts, sampling_params)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for i, output in enumerate(outputs):
            item = samples[i]
            generated_text = output.outputs[0].text
            
            item["trajectory"] = generated_text
            
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Done. Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data-path", type=str, default="/data1/home/yunhochoi/verl/data/snapshots_variants_train/data.jsonl", help="Input JSONL path")
    parser.add_argument("--output-path", type=str, default="/data1/home/yunhochoi/verl/data/llama_trajectories.jsonl", help="Output JSONL path")
    
    parser.add_argument("--model-path", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=6144)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)

    parser.add_argument("--batch-size", type=int, default=64) 
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)

    args = parser.parse_args()
    main(args)