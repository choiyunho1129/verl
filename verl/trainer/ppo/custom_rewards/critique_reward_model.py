import json
import os
import traceback
from concurrent.futures import ProcessPoolExecutor
from typing import Any, List, Optional, Tuple

from vllm import LLM, SamplingParams

from verl.utils.reward_score.math_verify import compute_score as mv_compute_score


def _get_env(*keys: str, default: str | None = None) -> str | None:
    for k in keys:
        v = os.getenv(k)
        if v not in (None, ""):
            return v
    return default


# --- Configuration ---
DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
REWARD_MODEL_NAME = _get_env("REWARD_MODEL_NAME", "REWARD_MODEL_PATH", default=DEFAULT_MODEL)
NUM_REPEATS = int(_get_env("REWARD_NUM_REPEATS", default="3"))
MAX_TOKENS = int(_get_env("REWARD_MAX_NEW_TOKENS", default="2048"))
TEMPERATURE = float(_get_env("REWARD_TEMPERATURE", default="0.6"))
TOP_P = float(_get_env("REWARD_TOP_P", default="0.9"))
MAX_PROMPT_LEN_CHARS = int(_get_env("REWARD_MAX_PROMPT_CHARS", default="8192"))
SCORING_PROCESSERS = int(_get_env("REWARD_SCORING_PROCESSES", default="4"))
DEBUG_MODE = (_get_env("REWARD_DEBUG", "DEBUG_REWARD", default="False") or "false").lower() == "true"
REWARD_TP = int(_get_env("REWARD_TENSOR_PARALLEL_SIZE", default="1"))


def _score_worker_func(args):
    pred, gt = args
    try:
        return mv_compute_score(pred, gt)
    except Exception:
        return 0.0


_PROCESS_POOL: Optional[ProcessPoolExecutor] = None
_LLM: Optional[LLM] = None


def _get_process_pool():
    global _PROCESS_POOL
    if _PROCESS_POOL is None:
        _PROCESS_POOL = ProcessPoolExecutor(max_workers=SCORING_PROCESSERS)
    return _PROCESS_POOL


def _get_llm() -> LLM:
    global _LLM
    if _LLM is None:
        _LLM = LLM(model=REWARD_MODEL_NAME, tensor_parallel_size=REWARD_TP)
    return _LLM

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


def _collect_predictions(solution_str: str, ground_truth: str) -> Tuple[List[List[str]], List[str]]:
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

    prompts: List[str] = []
    gts: List[str] = []
    for i, var in enumerate(variants):
        var_q = var.get("q") or var.get("question")
        var_a = var.get("a") or var.get("answer")
        if not var_q or not var_a:
            continue
        prompt = _build_prompt(original_q, original_traj, solution_str, var_q)
        if i == 0:
            _log_debug(f"Sample Prompt (len={len(prompt)}): {prompt[:100]}...")
        prompts.append(prompt)
        gts.append(var_a)

    if not prompts:
        return [], []

    llm = _get_llm()
    sampling_params = SamplingParams(
        n=NUM_REPEATS,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )

    try:
        outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
    except Exception as e:
        if DEBUG_MODE:
            traceback.print_exc()
        _log_debug(f"vLLM generate failed: {e}")
        return [], []

    preds_per_variant: List[List[str]] = []
    for out in outputs:
        candidates = []
        for cand in out.outputs:
            text = (cand.text or "").strip()
            if text:
                candidates.append(text)
        preds_per_variant.append(candidates)

    return preds_per_variant, gts


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Any = None,
    reward_router_address: str | None = None,  # API 호환용, 내부 추론에서는 사용하지 않음
    **kwargs,
) -> float:
    if data_source in {"HuggingFaceH4/MATH-500", "lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval"}:
        try:
            return float(mv_compute_score(data_source, solution_str, ground_truth, extra_info=extra_info))
        except Exception:
            return 0.0

    try:
        preds_per_variant, gts = _collect_predictions(solution_str, ground_truth)
    except Exception as e:
        if DEBUG_MODE:
            traceback.print_exc()
        return 0.0

    if not preds_per_variant:
        return 0.0

    scoring_tasks = []
    for preds, gt in zip(preds_per_variant, gts):
        if not preds:
            continue
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
