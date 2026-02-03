import argparse
import json
import os
import multiprocessing as mp
from types import SimpleNamespace
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

def generate(args):
    data_path = Path(args.data_path).expanduser()
    samples = load_data(data_path)
    print(f"Loaded {len(samples)} samples.")
    if args.num_shards > 1:
        if args.shard_id < 0 or args.shard_id >= args.num_shards:
            raise ValueError(f"Invalid shard_id {args.shard_id} for num_shards {args.num_shards}")
        samples = samples[args.shard_id::args.num_shards]
        print(f"Shard {args.shard_id}/{args.num_shards}: {len(samples)} samples.")

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
        n=args.num_trajectories,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    raw_prompts = []
    selected_samples = []  # Keep samples aligned with raw_prompts in case we skip any
    skipped = 0

    for s in samples:
        messages = None

        # Prefer explicit chat-style prompts when available
        if isinstance(s.get("prompt"), list):
            messages = s["prompt"]
        elif isinstance(s.get("messages"), list):
            messages = s["messages"]
        else:
            # Fall back to single-turn question style
            question = (
                s.get("question")
                or s.get("problem")
                or s.get("instruction")
                or s.get("input")
            )
            if question:
                # Include optional system prompt if present
                system_prompt = s.get("system") or s.get("system_prompt")
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": question})

        if not messages or any(m.get("content") is None for m in messages):
            skipped += 1
            continue

        formatted_prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        raw_prompts.append(formatted_prompt)
        selected_samples.append(s)

    if skipped:
        print(f"Skipped {skipped} samples missing prompt content.")

    print(f"Generating {args.num_trajectories} trajectories per question...")
    outputs = llm.generate(raw_prompts, sampling_params)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for i, output in enumerate(outputs):
            base_item = selected_samples[i]
            if not output.outputs:
                continue
            for gen in output.outputs:
                item = dict(base_item)
                item["trajectory"] = gen.text
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Done. Saved to {output_path}")

def run_worker(worker_args, device_id):
    # Restrict each worker to a single GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    generate(worker_args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data-path", type=str, default="/data01/yunhochoi/verl/data/MATH-500/train_MATH3-5_systemprompt.jsonl", help="Input JSONL path")
    parser.add_argument("--output-path", type=str, default="/data01/yunhochoi/verl/data/Qwen2.5_7b_instruct_trajectories_4.jsonl", help="Output JSONL path")
    
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    # Keep one full copy of the model per GPU (data-parallel)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=6144)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)

    parser.add_argument("--batch-size", type=int, default=64) 
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=1)
    parser.add_argument("--num-trajectories", type=int, default=4, help="Number of trajectories to generate per question")
    # Run a single process by default so all samples are processed unless the user opts into sharding
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards (processes)")
    parser.add_argument("--shard-id", type=int, default=0, help="Shard index for this process")
    parser.add_argument("--gpu-ids", type=str, default="2,3", help="Comma-separated GPU ids for data-parallel inference (each GPU loads full model). Overrides num_shards.")

    args = parser.parse_args()
    if args.gpu_ids:
        # Multiprocessing with CUDA must use spawn to avoid forked CUDA init errors
        mp.set_start_method("spawn", force=True)
        devices = [d.strip() for d in args.gpu_ids.split(",") if d.strip()]
        if not devices:
            raise ValueError("No valid GPU ids provided in --gpu-ids.")

        processes = []
        shard_paths = []
        for shard_id, dev in enumerate(devices):
            worker_args = SimpleNamespace(**vars(args))
            worker_args.tensor_parallel_size = 1  # force one GPU per worker
            worker_args.shard_id = shard_id
            worker_args.num_shards = len(devices)
            worker_args.output_path = f"{args.output_path}.shard{shard_id}"
            p = mp.Process(target=run_worker, args=(worker_args, dev))
            p.start()
            processes.append(p)
            shard_paths.append(worker_args.output_path)

        for p in processes:
            p.join()

        # Merge shard outputs into the final file
        final_path = Path(args.output_path)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        with final_path.open("w", encoding="utf-8") as fout:
            for shard_file in shard_paths:
                shard_path = Path(shard_file)
                if not shard_path.exists():
                    continue
                with shard_path.open("r", encoding="utf-8") as fin:
                    for line in fin:
                        fout.write(line)
        print(f"Merged {len(shard_paths)} shards into {args.output_path}")
    else:
        generate(args)
