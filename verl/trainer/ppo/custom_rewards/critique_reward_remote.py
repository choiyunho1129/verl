import json
import os
import asyncio
import aiohttp
import threading
import traceback
from typing import Any, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor 
from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.math_verify import compute_score as mv_compute_score


def _get_env(*keys: str, default: str | None = None) -> str | None:
    for k in keys:
        v = os.getenv(k)
        if v not in (None, ""):
            return v
    return default


# --- Configuration ---
DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
REWARD_PORT = _get_env("REWARD_PORT", default="8000")
REWARD_API_URL = _get_env("REWARD_API_URL", default=f"http://localhost:{REWARD_PORT}/v1")
REWARD_MODEL_NAME = _get_env("REWARD_MODEL_NAME", "REWARD_MODEL_PATH", default=DEFAULT_MODEL)
NUM_REPEATS = int(_get_env("REWARD_NUM_REPEATS", default="3"))
MAX_TOKENS = int(_get_env("REWARD_MAX_NEW_TOKENS", default="2048"))
TEMPERATURE = float(_get_env("REWARD_TEMPERATURE", default="0.6"))
TOP_P = float(_get_env("REWARD_TOP_P", default="0.9"))
MAX_CONCURRENCY = int(_get_env("REWARD_MAX_CONCURRENCY", default="8"))
HTTP_TIMEOUT_S = int(_get_env("REWARD_HTTP_TIMEOUT_S", default="60"))
MAX_RETRIES = int(_get_env("REWARD_HTTP_RETRIES", default="2"))
MAX_PROMPT_LEN_CHARS = int(_get_env("REWARD_MAX_PROMPT_CHARS", default="8192"))
SCORING_PROCESSERS = int(_get_env("REWARD_SCORING_PROCESSES", default="4")) 
DEBUG_MODE = (_get_env("REWARD_DEBUG", "DEBUG_REWARD", default="False") or "false").lower() == "true"


def _score_worker_func(args):
    pred, gt = args
    try:
        return mv_compute_score(pred, gt)
    except Exception:
        return 0.0

_PROCESS_POOL: Optional[ProcessPoolExecutor] = None

def _get_process_pool():
    global _PROCESS_POOL
    if _PROCESS_POOL is None:
        _PROCESS_POOL = ProcessPoolExecutor(max_workers=SCORING_PROCESSERS)
    return _PROCESS_POOL
# ---------------------------------------------


class _AsyncRunner:
    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        
        fut = asyncio.run_coroutine_threadsafe(self._init_session(), self._loop)
        fut.result()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _init_session(self):
        connector = aiohttp.TCPConnector(
            limit=MAX_CONCURRENCY * 2,
            ttl_dns_cache=300,
            keepalive_timeout=30,
            enable_cleanup_closed=True,
        )
        timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT_S)
        self._session = aiohttp.ClientSession(connector=connector, timeout=timeout)

    @property
    def session(self) -> aiohttp.ClientSession:
        return self._session

    def run(self, coro):
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result()


_RUNNER: Optional[_AsyncRunner] = None


def _get_runner() -> _AsyncRunner:
    global _RUNNER
    if _RUNNER is None:
        _RUNNER = _AsyncRunner()
    return _RUNNER


def _log_debug(msg: str):
    if DEBUG_MODE:
        print(f"[RewardDebug] {msg}", flush=True)


def _build_prompt(original_q: str, original_traj: str, critique: str, variant_q: str) -> str:
    prompt = (
        f"Original Problem: {original_q}\n"
        f"Original Solution Trace: {original_traj}\n\n"
        f"Critique on the Original Solution: {critique}\n\n"
        f"Instruction: Using the insights from the critique above, solve the following variation problem.\n"
        f"Think step-by-step and strictly put the final answer in \\boxed{{}} format.\n\n"
        f"Variation Problem: {variant_q}"
    )
    return prompt[:MAX_PROMPT_LEN_CHARS]


async def _call_llama(session: aiohttp.ClientSession, sem: asyncio.Semaphore, prompt: str, n: int) -> List[str]:
    url = f"{REWARD_API_URL}/chat/completions"
    payload = {
        "model": REWARD_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "n": n,
    }

    last_error = None
    
    async with sem:
        for attempt in range(MAX_RETRIES + 1):
            try:
                async with session.post(url, json=payload) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise RuntimeError(f"HTTP {resp.status}: {error_text}")

                    result = await resp.json()
                    choices = result.get("choices", []) or []
                    outs = []
                    for c in choices:
                        msg = c.get("message", {}) or {}
                        content = (msg.get("content", "") or "").strip()
                        if content:
                            outs.append(content)
                    
                    if not outs and n > 0:
                         _log_debug("Model returned empty responses.")
                         return []

                    return outs

            except aiohttp.ClientError as e:
                last_error = e
                await asyncio.sleep(1)
            except Exception as e:
                last_error = e
                await asyncio.sleep(0.5)

    if isinstance(last_error, aiohttp.ClientConnectorError):
        raise ConnectionError(
            f"CRITICAL: Cannot connect to vLLM server at {REWARD_API_URL}. Is it running? Error: {last_error}"
        )
    
    _log_debug(f"API call failed after retries: {last_error}")
    return []


async def _async_collect_predictions(solution_str: str, ground_truth: str) -> Tuple[List[List[str]], List[str]]:
    try:
        meta = json.loads(ground_truth)
    except Exception:
        _log_debug("JSON parsing failed for ground_truth.")
        return [], []

    original_q = meta.get("original_question") or meta.get("question", "") or ""
    original_traj = meta.get("original_trajectory") or meta.get("trajectory", "") or ""
    variants = meta.get("variants", []) or []

    if not variants:
        return [], []

    if not solution_str or len(solution_str.strip()) < 5:
        return [], []

    runner = _get_runner()
    session = runner.session
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    tasks = []
    gts = []
    prompts: List[str] = []

    for i, var in enumerate(variants):
        var_q = var.get("q") or var.get("question")
        var_a = var.get("a") or var.get("answer")
        if not var_q or not var_a:
            continue
        
        prompt = _build_prompt(original_q, original_traj, solution_str, var_q)
        
        if i == 0:
            _log_debug(f"Sample Prompt (len={len(prompt)}): {prompt[:100]}...")

        tasks.append(asyncio.create_task(_call_llama(session, sem, prompt, NUM_REPEATS)))
        gts.append(var_a)
        prompts.append(prompt)

    if not tasks:
        return [], []

    preds_per_variant: List[List[str]] = await asyncio.gather(*tasks)

    for idx, preds in enumerate(preds_per_variant):
        if preds:
            continue
        prompt = prompts[idx]
        try:
            retry_timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT_S * 2)
            async with aiohttp.ClientSession(timeout=retry_timeout) as tmp_session:
                tmp_preds = await _call_llama(tmp_session, asyncio.Semaphore(1), prompt, NUM_REPEATS)
            if tmp_preds:
                preds_per_variant[idx] = tmp_preds
                _log_debug(f"Fallback retry succeeded for variant {idx}.")
        except Exception:
            pass

    return preds_per_variant, gts


def compute_score(
    data_source: str, 
    solution_str: str,
    ground_truth: str,
    extra_info: Any = None,
    **kwargs,
) -> float:
    if data_source in {"HuggingFaceH4/MATH-500", "lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval"}:
        try:
            return float(default_compute_score(data_source, solution_str, ground_truth))
        except Exception:
            return 0.0

    runner = _get_runner()
    
    try:
        preds_per_variant, gts = runner.run(_async_collect_predictions(solution_str, ground_truth))
    except ConnectionError as e:
        print(f"\n[Reward Error] CRITICAL: vLLM Server Connection Failed!\n{e}\n")
        raise e
    except Exception as e:
        if DEBUG_MODE:
            traceback.print_exc()
        return 0.0

    if not preds_per_variant:
        return 0.0

    scoring_tasks = []
    for preds, gt in zip(preds_per_variant, gts):
        if not preds: continue
        for p in preds:
            scoring_tasks.append((p, gt))

    if not scoring_tasks:
        return 0.0

    try:
        pool = _get_process_pool()
        flat_scores = list(pool.map(_score_worker_func, scoring_tasks))
    except Exception as e:
        if DEBUG_MODE:
            print(f"[Reward Error] Process pool scoring failed: {e}")
        return 0.0

    total_score = 0.0
    denom = 0
    
    score_idx = 0  

    for i, (preds, gt) in enumerate(zip(preds_per_variant, gts)):
        if not preds:
            continue

        denom += 1
        s = 0.0
        k = 0
        for _ in preds:
            score = flat_scores[score_idx]
            s += score
            score_idx += 1
            k += 1

        variant_score = (s / k) if k > 0 else 0.0
        total_score += variant_score

        if i == 0 and DEBUG_MODE:
            _log_debug(f"Variant 0 Score: {variant_score} (GT: {gt}, Preds len: {k})")

    final_score = total_score / denom if denom > 0 else 0.0
    _log_debug(f"Final Reward Score: {final_score}")

    return final_score