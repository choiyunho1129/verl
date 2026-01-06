import asyncio
import json
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional

import aiohttp
from transformers import PreTrainedTokenizer

from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.math_verify import compute_score as mv_compute_score


def _get_env(*keys: str, default: str | None = None) -> str | None:
    for k in keys:
        v = os.getenv(k)
        if v not in (None, ""):
            return v
    return default


NUM_REPEATS = int(_get_env("REWARD_NUM_REPEATS", default="3"))
REWARD_MODEL_NAME = _get_env("REWARD_MODEL_PATH", default="meta-llama/Llama-3.2-3B-Instruct")
MAX_NEW_TOKENS = int(_get_env("REWARD_MAX_NEW_TOKENS", default="2048"))
TEMPERATURE = float(_get_env("REWARD_TEMPERATURE", default="0.6"))
TOP_P = float(_get_env("REWARD_TOP_P", default="0.9"))
MAX_CONCURRENCY = int(_get_env("REWARD_MAX_CONCURRENCY", default="2"))
HTTP_TIMEOUT_S = int(_get_env("REWARD_HTTP_TIMEOUT_S", default="60"))
MAX_RETRIES = int(_get_env("REWARD_HTTP_RETRIES", default="3"))
SCORING_PROCESSES = int(_get_env("REWARD_SCORING_PROCESSES", default="4"))
SCORE_MAX_CONCURRENCY = int(
    _get_env("REWARD_SCORE_MAX_CONCURRENCY", default=str(SCORING_PROCESSES))
)

DEBUG_MODE = (_get_env("REWARD_DEBUG", "DEBUG_REWARD", default="False") or "false").lower() == "true"
# -------------------------------------------------------


def _log_debug(msg: str) -> None:
    if DEBUG_MODE:
        print(f"[RewardDebug] {msg}", flush=True)


def _build_prompt(original_q: str, original_traj: str, critique: str, variant_q: str) -> str:
    return (
        f"Original Problem: {original_q}\n"
        f"Original Solution Trace: {original_traj}\n\n"
        f"Critique on the Original Solution: {critique}\n\n"
        f"Instruction: Using the critique above, solve the following variation problem. "
        f"Think step-by-step and put the final answer in \\boxed{{}}.\n\n"
        f"Variation Problem: {variant_q}"
    )


_PROCESS_POOL: ProcessPoolExecutor | None = None


def _get_process_pool() -> ProcessPoolExecutor:
    """
    Create a ProcessPoolExecutor using 'spawn' to avoid unsafe fork-from-thread.
    This is critical because compute_score may run in a worker thread.
    """
    global _PROCESS_POOL
    if _PROCESS_POOL is not None:
        return _PROCESS_POOL

    # Using spawn makes it safe to create processes even when called from a thread.
    ctx = mp.get_context("spawn")
    _PROCESS_POOL = ProcessPoolExecutor(max_workers=SCORING_PROCESSES, mp_context=ctx)
    return _PROCESS_POOL


async def _generate_single(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    router_address: str,
    prompt: str,
) -> Optional[str]:
    payload = {
        "model": REWARD_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "n": 1,
    }
    url = f"http://{router_address}/v1/chat/completions"

    async with semaphore:
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                async with session.post(url, json=payload) as resp:
                    resp.raise_for_status()
                    result = await resp.json(content_type=None)
                    choices = result.get("choices") or []
                    for choice in choices:
                        msg = choice.get("message") or {}
                        content = (msg.get("content") or "").strip()
                        if content:
                            return content

                    text = (result.get("text") or "").strip()
                    if text:
                        return text

            except Exception as exc:
                last_error = exc
                await asyncio.sleep(1.5 * (attempt + 1))

        _log_debug(f"GenRM call failed after retries: {last_error}")
        return None


async def _score_variant(
    router_address: str,
    session: aiohttp.ClientSession,
    http_semaphore: asyncio.Semaphore,
    score_semaphore: asyncio.Semaphore,
    prompt: str,
    ground_truth: str,
    loop: asyncio.AbstractEventLoop,
) -> tuple[float, List[str]]:
    gen_tasks = [
        asyncio.create_task(
            _generate_single(session, http_semaphore, router_address, prompt)
        )
        for _ in range(NUM_REPEATS)
    ]
    generations = [g for g in await asyncio.gather(*gen_tasks) if g]

    if not generations:
        return 0.0, []

    pool = _get_process_pool()

    async def _score_one(gen: str) -> float:
        # Limit the number of in-flight scoring jobs so we don't enqueue unbounded work.
        async with score_semaphore:
            try:
                return await loop.run_in_executor(pool, mv_compute_score, gen, ground_truth)
            except Exception:
                return 0.0

    score_tasks = [asyncio.create_task(_score_one(gen)) for gen in generations]
    scores = await asyncio.gather(*score_tasks)

    # Average over successful generations
    variant_score = float(sum(scores) / max(1, len(scores)))
    return variant_score, generations


# -------------------- Main entry --------------------
async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Any = None,
    reward_router_address: str | None = None,
    reward_model_tokenizer: PreTrainedTokenizer | None = None,
    **_: Any,
) -> dict[str, Any] | float:
    # Fast-path scoring for standard MATH datasets
    if data_source in {"HuggingFaceH4/MATH-500", "lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval"}:
        try:
            return {"score": float(default_compute_score(data_source, solution_str, ground_truth))} # need to be fixed.
        except Exception:
            return {"score": 0.0}

    if reward_router_address is None or reward_model_tokenizer is None:
        raise ValueError(
            "reward_router_address and reward_model_tokenizer are required. "
            "Ensure reward_model.enable=True and reward_model.use_reward_loop=True."
        )

    if not solution_str or len(solution_str.strip()) < 5:
        return {"score": 0.0, "acc": 0.0}

    # ground_truth is expected to be JSON containing original + variants
    try:
        meta = json.loads(ground_truth)
    except Exception:
        _log_debug("Failed to parse ground_truth JSON.")
        return {"score": 0.0, "acc": 0.0}

    original_q = meta.get("original_question") or meta.get("question", "") or ""
    original_traj = meta.get("original_trajectory") or meta.get("trajectory", "") or ""
    variants = meta.get("variants", []) or []
    if not variants:
        return {"score": 0.0, "acc": 0.0}

    timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT_S)
    http_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    score_semaphore = asyncio.Semaphore(SCORE_MAX_CONCURRENCY)
    loop = asyncio.get_running_loop()

    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks: List[asyncio.Task[tuple[float, List[str]]]] = []

        for idx, variant in enumerate(variants):
            var_q = variant.get("q") or variant.get("question")
            var_a = variant.get("a") or variant.get("answer")
            if not var_q or not var_a:
                continue

            prompt = _build_prompt(original_q, original_traj, solution_str, var_q)
            if idx == 0:
                _log_debug(f"Sample GenRM prompt (len={len(prompt)}): {prompt[:120]}...")

            tasks.append(
                asyncio.create_task(
                    _score_variant(
                        reward_router_address,
                        session,
                        http_semaphore,
                        score_semaphore,
                        prompt,
                        var_a,
                        loop,
                    )
                )
            )

        if not tasks:
            return {"score": 0.0, "acc": 0.0}

        results = await asyncio.gather(*tasks)

    variant_scores: List[float] = []
    all_generations: List[List[str]] = []
    for score, gens in results:
        variant_scores.append(float(score))
        all_generations.append(gens)

    if not variant_scores:
        return {"score": 0.0, "acc": 0.0}

    final_score = float(sum(variant_scores) / len(variant_scores))
    _log_debug(f"Final Reward Score: {final_score}")

    return {
        "score": final_score,
        "acc": final_score,
        "variant_scores": variant_scores,
        "num_variants": len(variant_scores),
        "num_generations": sum(len(g) for g in all_generations),
    }
