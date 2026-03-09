import gc
import json
import os
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import torch
from math_verify import parse, verify
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from oat_math_grader import boxed_reward_fn as oat_evaluate

THOUGHT_DELIMITER_START = "<think>"
THOUGHT_DELIMITER_END = "</think>"

INITIAL_STAGE_MAX_MODEL_LEN = 4096
CRITIQUE_STAGE_MAX_MODEL_LEN = 6144
REVISE_STAGE_MAX_MODEL_LEN = 8192


def timeout(timeout_seconds: int = 10):
    if os.name == "posix":
        import signal

        def decorator(func):
            def handler(signum, frame):
                raise TimeoutError("verify timed out!")

            def wrapper(*args, **kwargs):
                old_handler = signal.getsignal(signal.SIGALRM)
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(timeout_seconds)
                try:
                    return func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)

            return wrapper

        return decorator


@timeout(timeout_seconds=10)
def labeling_responses(responses: list[str], golden_answer: str) -> list[bool]:
    predict_answers = list(map(parse, responses))
    golden_answers = list(map(parse, ["$" + golden_answer + "$"] * len(responses)))
    labels: list[bool] = []
    for golden, pred in zip(golden_answers, predict_answers):
        try:
            labels.append(bool(verify(golden, pred)))
        except Exception:
            labels.append(False)
    return labels


def normalize_bool(value: Any, default: bool | None = None) -> bool | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "t"}:
            return True
        if normalized in {"false", "0", "no", "n", "f"}:
            return False
        if normalized == "":
            return default
    raise ValueError(f"Cannot parse bool from value={value!r}")


def normalize_float(value: Any, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, str) and value.strip() == "":
        return default
    return float(value)


def normalize_int(value: Any, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, str) and value.strip() == "":
        return default
    return int(value)


def normalize_messages(messages: Any) -> list[dict]:
    if isinstance(messages, np.ndarray):
        messages = messages.tolist()
    elif hasattr(messages, "tolist") and not isinstance(messages, (list, str, bytes)):
        try:
            messages = messages.tolist()
        except Exception:
            pass

    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except Exception:
            messages = [{"role": "user", "content": messages}]

    if not isinstance(messages, list):
        raise ValueError(f"Prompt must be a list of messages, got: {type(messages)}")
    return messages


def extract_question(messages: list[dict]) -> str:
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if content:
                return content
    if messages:
        return messages[-1].get("content", "")
    return ""


def extract_answer(reward_model: Any) -> str:
    if isinstance(reward_model, str):
        try:
            reward_model = json.loads(reward_model)
        except Exception:
            return ""
    if isinstance(reward_model, dict):
        answer = reward_model.get("ground_truth")
        if answer is None:
            return ""
        return str(answer)
    return ""


def apply_qwen_math_template(question: str, tokenizer, enable_thinking=None) -> str:
    messages = [
        {
            "role": "system",
            "content": "Please reason step by step, and put your final answer within \\boxed{}.",
        },
        {"role": "user", "content": question},
    ]
    apply_kwargs = {}
    if enable_thinking is not None:
        apply_kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **apply_kwargs,
    )


def apply_qwen_revision_template(user_prompt: str, tokenizer, enable_thinking=None) -> str:
    system_prompt = (
        "Revise the solution by addressing the critique and regenerate the full answer.\n"
        "Put your final answer within \\boxed{}."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    apply_kwargs = {}
    if enable_thinking is not None:
        apply_kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **apply_kwargs,
    )


def make_prompt(question: str, trajectory: str) -> tuple[str, str]:
    system_prompt = (
        "You are a math teacher. Review the student's solution trace and provide critique that helps them solve similar "
        "variant problems well. Identify specific logical errors or confirm the reasoning. Provide constructive feedback "
        "but do not give the direct answer."
    )
    user_prompt = f"Question:\n{question}\n\nModel Solution Trace:\n{trajectory}\n\n"
    return system_prompt, user_prompt


def build_critique_prompt(question: str, trajectory: str, tokenizer) -> str:
    system_prompt, user_prompt = make_prompt(question=question, trajectory=trajectory)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_revision_user_prompt(question: str, trajectory: str, critique: str) -> str:
    return (
        f"Question:\n{question}\n\n"
        f"Your Previous Solution Trace:\n{trajectory}\n\n"
        f"Critique:\n{critique}\n\n"
    )


def write_jsonl(path: str, rows: list[dict]) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def flatten_vllm_outputs(raw_outputs, n: int) -> tuple[list[str], list[str]]:
    prompts: list[str] = []
    texts: list[str] = []
    for output in raw_outputs:
        prompt = output.prompt
        for j in range(n):
            text = output.outputs[j].text if j < len(output.outputs) else ""
            prompts.append(prompt)
            texts.append(text)
    return prompts, texts


def single_vllm_outputs(raw_outputs) -> tuple[list[str], list[str]]:
    prompts: list[str] = []
    texts: list[str] = []
    for output in raw_outputs:
        prompts.append(output.prompt)
        texts.append(output.outputs[0].text if output.outputs else "")
    return prompts, texts


def build_stage_llm(model_path: str, tensor_parallel_size: int, max_model_len: int) -> LLM:
    return LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.85,
        max_model_len=max_model_len,
    )


def main(
    input_file,
    output_file,
    model_path,
    debug=False,
    remove_system=True,
    template="qwen",
    temperature=0.6,
    top_p=1.0,
    max_tokens=8192,
    n=1,
    force_generate=True,
    add_think_before_answer=False,
    add_oat_evaluate=False,
    any_true=False,
    skip_scoring=False,
    output_eval=None,
    no_split_think=False,
    enable_thinking=None,
    critique_temperature=None,
    critique_top_p=None,
    critique_max_tokens=None,
    revise_temperature=None,
    revise_top_p=None,
    revise_max_tokens=None,
):
    del debug, output_eval

    if template != "qwen":
        raise ValueError("This script currently supports template='qwen' only.")

    remove_system = bool(normalize_bool(remove_system, default=True))
    force_generate = bool(normalize_bool(force_generate, default=True))
    add_think_before_answer = bool(normalize_bool(add_think_before_answer, default=False))
    add_oat_evaluate = bool(normalize_bool(add_oat_evaluate, default=False))
    any_true = bool(normalize_bool(any_true, default=False))
    skip_scoring = bool(normalize_bool(skip_scoring, default=False))
    no_split_think = bool(normalize_bool(no_split_think, default=False))
    enable_thinking = normalize_bool(enable_thinking, default=None)

    temperature = normalize_float(temperature, 0.6)
    top_p = normalize_float(top_p, 1.0)
    max_tokens = normalize_int(max_tokens, 8192)
    n = normalize_int(n, 1)
    critique_temperature = normalize_float(critique_temperature, temperature)
    critique_top_p = normalize_float(critique_top_p, top_p)
    critique_max_tokens = normalize_int(critique_max_tokens, max_tokens)
    revise_temperature = normalize_float(revise_temperature, temperature)
    revise_top_p = normalize_float(revise_top_p, top_p)
    revise_max_tokens = normalize_int(revise_max_tokens, max_tokens)

    output_root = output_file[:-6] if output_file.endswith(".jsonl") else output_file
    initial_decoded_path = output_root + ".initial.decoded.jsonl"
    critique_decoded_path = output_root + ".critique.decoded.jsonl"
    revised_decoded_path = output_root + ".revised.decoded.jsonl"

    need_generate = force_generate or not (
        os.path.exists(initial_decoded_path)
        and os.path.exists(critique_decoded_path)
        and os.path.exists(revised_decoded_path)
    )

    if need_generate:
        df = pd.read_parquet(input_file)

        messages = [normalize_messages(item) for item in df["prompt"].tolist()]
        if remove_system:
            cleaned_messages = []
            for message in messages:
                if message and message[0].get("role") == "system":
                    cleaned_messages.append(message[1:])
                else:
                    cleaned_messages.append(message)
            messages = cleaned_messages

        questions = [extract_question(message) for message in messages]
        answers = [extract_answer(item) for item in df["reward_model"].tolist()]
        data_sources = (
            df["data_source"].tolist() if "data_source" in df.columns else ["unknown"] * len(df)
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(torch.cuda.device_count())
        tensor_parallel_size = max(1, torch.cuda.device_count())

        solve_sampling = SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=max_tokens, n=n
        )
        critique_sampling = SamplingParams(
            temperature=critique_temperature,
            top_p=critique_top_p,
            max_tokens=critique_max_tokens,
            n=1,
        )
        revise_sampling = SamplingParams(
            temperature=revise_temperature,
            top_p=revise_top_p,
            max_tokens=revise_max_tokens,
            n=1,
        )

        print(
            f"[stage1] temperature={temperature}, top_p={top_p}, max_tokens={max_tokens}, "
            f"max_model_len={INITIAL_STAGE_MAX_MODEL_LEN}, n={n}, enable_thinking={enable_thinking}"
        )
        initial_prompts = [
            apply_qwen_math_template(question, tokenizer, enable_thinking=enable_thinking)
            for question in questions
        ]
        if initial_prompts:
            print("Example initial prompt:", initial_prompts[0])
        llm_stage1 = build_stage_llm(
            model_path=model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=INITIAL_STAGE_MAX_MODEL_LEN,
        )
        initial_raw_outputs = llm_stage1.generate(initial_prompts, solve_sampling)
        del llm_stage1
        gc.collect()
        torch.cuda.empty_cache()
        initial_prompt_texts, initial_texts = flatten_vllm_outputs(initial_raw_outputs, n=n)

        flat_questions = [q for q in questions for _ in range(n)]
        flat_answers = [a for a in answers for _ in range(n)]
        flat_data_sources = [d for d in data_sources for _ in range(n)]

        initial_records = []
        for prompt, question, text, answer, data_source in zip(
            initial_prompt_texts, flat_questions, initial_texts, flat_answers, flat_data_sources
        ):
            initial_records.append(
                {
                    "prompt": prompt,
                    "question": question,
                    "generated_text": text,
                    "answer": answer,
                    "data_source": data_source,
                }
            )
        write_jsonl(initial_decoded_path, initial_records)

        print(
            f"[stage2] temperature={critique_temperature}, top_p={critique_top_p}, max_tokens={critique_max_tokens}, "
            f"max_model_len={CRITIQUE_STAGE_MAX_MODEL_LEN}"
        )
        critique_prompts = [
            build_critique_prompt(question, trajectory, tokenizer)
            for question, trajectory in zip(flat_questions, initial_texts)
        ]
        if critique_prompts:
            print("Example critique prompt:", critique_prompts[0])
        llm_stage2 = build_stage_llm(
            model_path=model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=CRITIQUE_STAGE_MAX_MODEL_LEN,
        )
        critique_raw_outputs = llm_stage2.generate(critique_prompts, critique_sampling)
        del llm_stage2
        gc.collect()
        torch.cuda.empty_cache()
        critique_prompt_texts, critique_texts = single_vllm_outputs(critique_raw_outputs)

        critique_records = []
        for prompt, question, trajectory, critique, answer, data_source in zip(
            critique_prompt_texts,
            flat_questions,
            initial_texts,
            critique_texts,
            flat_answers,
            flat_data_sources,
        ):
            critique_records.append(
                {
                    "prompt": prompt,
                    "question": question,
                    "initial_generated_text": trajectory,
                    "generated_text": critique,
                    "answer": answer,
                    "data_source": data_source,
                }
            )
        write_jsonl(critique_decoded_path, critique_records)

        print(
            f"[stage3] temperature={revise_temperature}, top_p={revise_top_p}, max_tokens={revise_max_tokens}, "
            f"max_model_len={REVISE_STAGE_MAX_MODEL_LEN}"
        )
        revised_user_prompts = [
            build_revision_user_prompt(question, trajectory, critique)
            for question, trajectory, critique in zip(flat_questions, initial_texts, critique_texts)
        ]
        revised_prompts = [
            apply_qwen_revision_template(user_prompt, tokenizer, enable_thinking=enable_thinking)
            for user_prompt in revised_user_prompts
        ]
        if revised_prompts:
            print("Example revised prompt:", revised_prompts[0])
        llm_stage3 = build_stage_llm(
            model_path=model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=REVISE_STAGE_MAX_MODEL_LEN,
        )
        revised_raw_outputs = llm_stage3.generate(revised_prompts, revise_sampling)
        del llm_stage3
        gc.collect()
        torch.cuda.empty_cache()
        revised_prompt_texts, revised_texts = single_vllm_outputs(revised_raw_outputs)

        revised_records = []
        for prompt, question, initial_text, critique, revised_text, answer, data_source in zip(
            revised_prompt_texts,
            flat_questions,
            initial_texts,
            critique_texts,
            revised_texts,
            flat_answers,
            flat_data_sources,
        ):
            revised_records.append(
                {
                    "prompt": prompt,
                    "question": question,
                    "initial_generated_text": initial_text,
                    "critique_generated_text": critique,
                    "generated_text": revised_text,
                    "answer": answer,
                    "data_source": data_source,
                }
            )
        write_jsonl(revised_decoded_path, revised_records)

    else:
        print("Found decoded stage files. Skip generation.")

    initial_records = read_jsonl(initial_decoded_path)
    critique_records = read_jsonl(critique_decoded_path)
    revised_records = read_jsonl(revised_decoded_path)

    if not (len(initial_records) == len(critique_records) == len(revised_records)):
        raise ValueError(
            "Decoded file sizes do not match: "
            f"initial={len(initial_records)}, critique={len(critique_records)}, revised={len(revised_records)}"
        )

    outputs = [item["generated_text"] for item in revised_records]
    prompts = [item["prompt"] for item in revised_records]
    answers = [item["answer"] for item in revised_records]
    data_sources = [item.get("data_source", "unknown") for item in revised_records]

    if skip_scoring:
        return

    rets = defaultdict(list)
    save_data = []
    avg = 0
    diff_cnt = 0

    print("Scoring...")
    for i in tqdm(range(len(outputs)), total=len(outputs)):
        generated_text = outputs[i]
        prompt = prompts[i]
        answer = answers[i]

        think_format = False
        if prompt.endswith(THOUGHT_DELIMITER_START + "\n") or add_think_before_answer is True:
            generated_text = THOUGHT_DELIMITER_START + "\n" + generated_text
            think_format = True
        if no_split_think:
            think_format = False

        labels = None
        if think_format:
            try:
                generated_text = generated_text.split(THOUGHT_DELIMITER_END)[1]
            except Exception:
                labels = [False]

        if labels is None:
            try:
                labels = labeling_responses([generated_text], answer)
            except Exception:
                labels = [False]

        if add_oat_evaluate:
            try:
                new_label = oat_evaluate(generated_text, answer, fast=False)
                new_label = new_label[1] == 1.0
            except Exception:
                new_label = False

            if any_true is True:
                if labels[0] is False and new_label is True:
                    diff_cnt += 1
                labels = [labels[0] or new_label]
            else:
                labels = [new_label]

        rets[data_sources[i]].append(labels[0])
        save_data.append(
            {
                "prompt": prompt,
                "question": revised_records[i].get("question", ""),
                "initial_generated_text": initial_records[i].get("generated_text", ""),
                "critique_generated_text": critique_records[i].get("generated_text", ""),
                "generated_text": generated_text,
                "answer": answer,
                "correctness": labels[0],
            }
        )
        if labels[0]:
            avg += 1

    print("accuracy: ", avg / len(outputs) if outputs else 0.0)
    print("diff_cnt: ", diff_cnt)

    accs = []
    for data_source, labels in rets.items():
        acc = np.array(labels).mean() if labels else 0.0
        print(f"{data_source}: {acc}")
        accs.append(acc)

    print("avg acc: ", np.array(accs).mean() if accs else 0.0)

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for item in save_data:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
