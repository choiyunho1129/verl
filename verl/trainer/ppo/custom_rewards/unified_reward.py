import asyncio
import json
import os
import multiprocessing as mp
import re
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from transformers import PreTrainedTokenizer

def _get_env(*keys: str, default: str | None = None) -> str | None:
    for k in keys:
        v = os.getenv(k)
        if v not in (None, ""):
            return v
    return default


# --- Configuration ---
NUM_REPEATS = int(_get_env("REWARD_NUM_REPEATS", default="3"))
REWARD_MODEL_NAME = _get_env("REWARD_MODEL_PATH", default="meta-llama/Llama-3.2-3B-Instruct")
MAX_NEW_TOKENS = int(_get_env("REWARD_MAX_NEW_TOKENS", default="2048"))
TEMPERATURE = float(_get_env("REWARD_TEMPERATURE", default="0.6"))
TOP_P = float(_get_env("REWARD_TOP_P", default="1.0"))
MAX_CONCURRENCY = int(_get_env("REWARD_MAX_CONCURRENCY", default="6")) 
HTTP_TIMEOUT_S = int(_get_env("REWARD_HTTP_TIMEOUT_S", default="120"))
MAX_RETRIES = int(_get_env("REWARD_HTTP_RETRIES", default="6")) 
SCORING_PROCESSES = int(_get_env("REWARD_SCORING_PROCESSES", default="8"))
SCORE_MAX_CONCURRENCY = int(
    _get_env("REWARD_SCORE_MAX_CONCURRENCY", default=str(SCORING_PROCESSES))
)
DEBUG_MODE = (_get_env("REWARD_DEBUG", "DEBUG_REWARD", default="False") or "false").lower() == "true"
# ---------------------
_MATH_VERIFY_UNAVAILABLE_LOGGED = False


def _log_debug(msg: str) -> None:
    if DEBUG_MODE:
        print(f"[RewardDebug] {msg}", flush=True)


def _try_math_verify_score(model_output: str, ground_truth: str) -> Optional[float]:
    global _MATH_VERIFY_UNAVAILABLE_LOGGED
    try:
        from verl.utils.reward_score.math_verify import compute_score as mv_compute_score
    except Exception:
        if not _MATH_VERIFY_UNAVAILABLE_LOGGED:
            print(
                "[RewardWarn] math_verify is unavailable. "
                "Solve/verification rewards will return 0 when labels are not precomputed.",
                flush=True,
            )
            _MATH_VERIFY_UNAVAILABLE_LOGGED = True
        return None

    try:
        return float(mv_compute_score(model_output, ground_truth))
    except Exception:
        return 0.0


_VERDICT_PATTERN = re.compile(r"final\s*verdict\s*[:\-]\s*(correct|incorrect)\b", re.IGNORECASE)
_VERDICT_TOKEN_PATTERN = re.compile(r"\b(correct|incorrect)\b", re.IGNORECASE)

_SOLVE_DATA_SOURCES = {
    "huggingfaceh4/math-500",
    "lighteval/math",
    "digitallearninggmbh/math-lighteval",
    "deepmath",
    "qwen-math",
    "deepscaler",
    "math",
    "math_dapo",
    "math_dapo_reasoning",
}
_CRITIQUE_DATA_SOURCES = {
    "critique",
    "critique_variants",
}


def _normalize_data_source(data_source: str) -> str:
    return (data_source or "").strip().lower()


def _is_verification_data_source(data_source: str) -> bool:
    normalized = _normalize_data_source(data_source)
    return "verification" in normalized


def _is_critique_data_source(data_source: str) -> bool:
    normalized = _normalize_data_source(data_source)
    return normalized in _CRITIQUE_DATA_SOURCES or "critique" in normalized


def _is_solve_data_source(data_source: str) -> bool:
    normalized = _normalize_data_source(data_source)
    return normalized in _SOLVE_DATA_SOURCES or "solve" in normalized or normalized.startswith("aime")


def _parse_ground_truth_dict(ground_truth: Any) -> Dict[str, Any]:
    if isinstance(ground_truth, dict):
        return ground_truth
    if isinstance(ground_truth, str):
        try:
            payload = json.loads(ground_truth)
            if isinstance(payload, dict):
                return payload
        except Exception:
            return {}
    return {}


def _extract_solve_ground_truth(ground_truth: Any, meta: Dict[str, Any]) -> str:
    for key in ("answer", "ground_truth", "label", "target"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return str(ground_truth) if ground_truth is not None else ""


def _to_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        if value == 1:
            return True
        if value == 0:
            return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "correct"}:
            return True
        if normalized in {"false", "0", "no", "n", "incorrect"}:
            return False
    return None


def _parse_verification_verdict(solution_str: str) -> Optional[bool]:
    if not solution_str:
        return None

    verdict_matches = _VERDICT_PATTERN.findall(solution_str)
    if verdict_matches:
        return verdict_matches[-1].lower() == "correct"

    token_matches = _VERDICT_TOKEN_PATTERN.findall(solution_str)
    if token_matches:
        return token_matches[-1].lower() == "correct"

    return None


def _resolve_actual_correct(meta: Dict[str, Any], extra_info: Any) -> Optional[bool]:
    for key in ("actual_correct", "verified_correct", "trajectory_is_correct"):
        resolved = _to_bool(meta.get(key))
        if resolved is not None:
            return resolved

    if isinstance(extra_info, dict):
        for key in ("actual_correct", "verified_correct", "trajectory_is_correct"):
            resolved = _to_bool(extra_info.get(key))
            if resolved is not None:
                return resolved

    trajectory = meta.get("trajectory")
    answer = meta.get("answer")
    if trajectory is not None and answer is not None:
        score = _try_math_verify_score(str(trajectory), str(answer))
        if score is None:
            return None
        return score > 0.0

    verification_score = meta.get("verification_score")
    if isinstance(verification_score, int | float):
        return float(verification_score) > 0.0

    return None


def _resolve_task_type(data_source: str, meta: Dict[str, Any]) -> str:
    if _is_verification_data_source(data_source):
        return "verification"
    if _is_critique_data_source(data_source):
        return "critique"
    if _is_solve_data_source(data_source):
        return "solve"

    if meta:
        if any(k in meta for k in ("actual_correct", "verified_correct", "trajectory_is_correct")):
            return "verification"
        variants = meta.get("variants")
        if isinstance(variants, list) and variants:
            return "critique"
        if meta.get("trajectory") is not None and meta.get("answer") is not None:
            return "verification"

    return "solve"


def _compute_verification_reward(solution_str: str, ground_truth: Any, extra_info: Any) -> dict[str, Any]:
    meta = _parse_ground_truth_dict(ground_truth)
    actual_correct = _resolve_actual_correct(meta, extra_info)
    model_pred_correct = _parse_verification_verdict(solution_str or "")
    parsed = model_pred_correct is not None

    if actual_correct is None:
        return {
            "score": 0.0,
            "acc": 0.0,
            "verification_accuracy": 0.0,
            "parsed": 1.0 if parsed else 0.0,
            "model_pred_correct": -1.0 if model_pred_correct is None else (1.0 if model_pred_correct else 0.0),
            "actual_correct": -1.0,
            "task_type_id": 1.0,
        }

    match = parsed and (model_pred_correct == actual_correct)
    score = 1.0 if match else 0.0
    return {
        "score": score,
        "acc": score,
        "verification_accuracy": score,
        "parsed": 1.0 if parsed else 0.0,
        "model_pred_correct": -1.0 if model_pred_correct is None else (1.0 if model_pred_correct else 0.0),
        "actual_correct": 1.0 if actual_correct else 0.0,
        "task_type_id": 1.0,
    }


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
    This is critical for stability in PyTorch/CUDA environments.
    """
    global _PROCESS_POOL
    if _PROCESS_POOL is not None:
        return _PROCESS_POOL

    ctx = mp.get_context("spawn")
    _PROCESS_POOL = ProcessPoolExecutor(max_workers=SCORING_PROCESSES, mp_context=ctx)
    return _PROCESS_POOL


async def _generate_single(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    router_address: str,
    prompt: str,
) -> Optional[str]:
    """
    Sends a request to the vLLM server with aggressive retry logic.
    """
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
        for attempt in range(MAX_RETRIES + 1):
            try:
                async with session.post(url, json=payload) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise RuntimeError(f"HTTP {resp.status}: {error_text}")
                    
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
                    
                    # If empty response, treat as error to trigger retry
                    raise ValueError("Empty response from model")

            except Exception as exc:
                last_error = exc
                # Aggressive Backoff: 1s, 2s, 4s, 8s...
                sleep_time = 1.0 * (2 ** attempt)
                if attempt < MAX_RETRIES:
                    _log_debug(f"Generation attempt {attempt+1} failed. Retrying in {sleep_time}s. Error: {exc}")
                    await asyncio.sleep(sleep_time)

        _log_debug(f"GenRM critical failure after {MAX_RETRIES+1} attempts: {last_error}")
        return None


async def _score_variant(
    router_address: str,
    session: aiohttp.ClientSession,
    http_semaphore: asyncio.Semaphore,
    score_semaphore: asyncio.Semaphore,
    prompt: str,
    ground_truth: str,
    loop: asyncio.AbstractEventLoop,
) -> Tuple[Optional[float], List[str]]:
    """
    Generates solutions and scores them.
    Returns: (score, generations_list)
    If generation completely fails, score is None (to be skipped).
    """
    gen_tasks = [
        asyncio.create_task(
            _generate_single(session, http_semaphore, router_address, prompt)
        )
        for _ in range(NUM_REPEATS)
    ]
    
    # Wait for all generations
    raw_generations = await asyncio.gather(*gen_tasks)
    generations = [g for g in raw_generations if g]

    # If all generations failed due to network/server issues, return None score
    # so we don't penalize the model for infrastructure failures.
    if not generations:
        return None, []

    pool = _get_process_pool()

    async def _score_one(gen: str) -> float:
        async with score_semaphore:
            try:
                # Running CPU-bound scoring in a separate process (spawn context)
                score = await loop.run_in_executor(pool, _try_math_verify_score, gen, ground_truth)
                if score is None:
                    return 0.0
                return float(score)
            except Exception:
                return 0.0

    score_tasks = [asyncio.create_task(_score_one(gen)) for gen in generations]
    scores = await asyncio.gather(*score_tasks)

    # Average over successful generations
    if not scores:
        return 0.0, generations
        
    variant_score = float(sum(scores) / len(scores))
    return variant_score, generations


async def _score_math_verify(solution_str: str, ground_truth: str) -> float:
    loop = asyncio.get_running_loop()
    pool = _get_process_pool()
    try:
        score = await loop.run_in_executor(pool, _try_math_verify_score, solution_str, ground_truth)
        if score is None:
            return 0.0
        return float(score)
    except Exception as exc:
        _log_debug(f"Math-verify scoring failed: {exc}")
        return 0.0


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
    meta = _parse_ground_truth_dict(ground_truth)
    task_type = _resolve_task_type(data_source=data_source, meta=meta)

    if task_type == "verification":
        result = _compute_verification_reward(solution_str=solution_str, ground_truth=ground_truth, extra_info=extra_info)
        result["task_type"] = "verification"
        return result

    if task_type == "solve":
        solve_ground_truth = _extract_solve_ground_truth(ground_truth, meta)
        score = await _score_math_verify(solution_str, solve_ground_truth)
        return {
            "score": score,
            "acc": score,
            "solve_accuracy": score,
            "num_variants": 1 if score > 0.0 else 0,
            "num_generations": 0,
            "task_type_id": 0.0,
            "task_type": "solve",
        }

    if reward_router_address is None or reward_model_tokenizer is None:
        raise ValueError(
            "Critique reward requires reward_router_address and reward_model_tokenizer. "
            f"Current data_source={data_source!r} was resolved as critique."
        )

    if not solution_str or len(solution_str.strip()) < 5:
        return {"score": 0.0, "acc": 0.0, "num_variants": 0, "num_generations": 0, "task_type_id": 2.0, "task_type": "critique"}

    original_q = meta.get("original_question") or meta.get("question", "") or ""
    original_traj = meta.get("original_trajectory") or meta.get("trajectory", "") or ""
    variants = meta.get("variants", []) or []
    
    if not variants:
        return {"score": 0.0, "acc": 0.0, "num_variants": 0, "num_generations": 0, "task_type_id": 2.0, "task_type": "critique"}

    timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT_S)
    http_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    score_semaphore = asyncio.Semaphore(SCORE_MAX_CONCURRENCY)
    loop = asyncio.get_running_loop()

    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks: List[asyncio.Task[Tuple[Optional[float], List[str]]]] = []

        for idx, variant in enumerate(variants):
            var_q = variant.get("q") or variant.get("question")
            var_a = variant.get("a") or variant.get("answer")
            if not var_q or not var_a:
                continue

            prompt = _build_prompt(original_q, original_traj, solution_str, var_q)
            if idx == 0:
                _log_debug(f"Sample second generation prompt (len={len(prompt)}): {prompt[:100]}...")

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
            return {"score": 0.0, "acc": 0.0, "task_type_id": 2.0, "task_type": "critique"}

        results = await asyncio.gather(*tasks)

    # --- UPDATED AGGREGATION LOGIC ---
    valid_scores: List[float] = []
    all_generations: List[List[str]] = []
    
    for score, gens in results:
        all_generations.append(gens)
        # Only include variants that successfully returned a score (not None)
        # None implies network failure or model timeout for all repeats.
        if score is not None:
            valid_scores.append(score)

    if not valid_scores:
        # All variants failed to generate. Return 0.0 or handled as failure.
        _log_debug("All variants failed to generate valid responses.")
        return {"score": 0.0, "acc": 0.0, "num_variants": 0, "num_generations": 0, "task_type_id": 2.0, "task_type": "critique"}

    # Calculate average only on valid scores
    final_score = float(sum(valid_scores) / len(valid_scores))
    
    if DEBUG_MODE:
        _log_debug(f"Final Reward Score: {final_score} (Valid Variants: {len(valid_scores)}/{len(variants)})")

    return {
        "score": final_score,
        "acc": final_score,
        "num_variants": len(valid_scores),
        "num_generations": sum(len(g) for g in all_generations),
        "task_type_id": 2.0,
        "task_type": "critique",
    }
