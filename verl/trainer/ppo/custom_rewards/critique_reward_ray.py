import json
import math
import os
import ray
import traceback
from concurrent.futures import ProcessPoolExecutor
from typing import Any, List, Optional, Tuple

from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.math_verify import compute_score as mv_compute_score


REWARD_MODEL_NAME = os.getenv("REWARD_MODEL_PATH", "meta-llama/Llama-3.2-3B-Instruct")
NUM_REPEATS = int(os.getenv("REWARD_NUM_REPEATS", "3"))
MAX_TOKENS = int(os.getenv("REWARD_MAX_NEW_TOKENS", "2048"))
MAX_PROMPT_TOKENS = int(os.getenv("REWARD_MAX_PROMPT_TOKENS", "8192"))
TEMPERATURE = float(os.getenv("REWARD_TEMPERATURE", "0.6"))
TOP_P = float(os.getenv("REWARD_TOP_P", "0.9"))
DEBUG_MODE = (os.getenv("DEBUG_REWARD", "False").lower() == "true")
SCORING_PROCESSES = int(os.getenv("REWARD_SCORING_PROCESSES", "4"))
REWARD_NUM_ACTORS = max(1, int(os.getenv("REWARD_NUM_ACTORS", "1")))
REWARD_ACTOR_GPUS_PER_WORKER = max(1, int(os.getenv("REWARD_ACTOR_GPUS_PER_WORKER", "1")))
REWARD_MODEL_TENSOR_PARALLEL_SIZE = max(
    1, int(os.getenv("REWARD_MODEL_TENSOR_PARALLEL_SIZE", str(REWARD_ACTOR_GPUS_PER_WORKER)))
)
_SCORING_POOL: Optional[ProcessPoolExecutor] = None
_ACTOR_POOL: Optional[List[Any]] = None


def _score_worker_func(args: Tuple[str, str]) -> float:
    pred, gt = args
    try:
        return float(mv_compute_score(pred, gt))
    except Exception as e:
        if DEBUG_MODE:
            print(f"[Reward Debug] Scoring worker error: {e}", flush=True)
        return 0.0


@ray.remote(num_gpus=REWARD_ACTOR_GPUS_PER_WORKER)
class VLLMRewardActor:
    def __init__(self, model_path: str, tensor_parallel_size: int):
        print(f"[RewardActor] Initializing vLLM with model: {model_path}...", flush=True)
        try:
            from vllm import LLM, SamplingParams

            self.llm = LLM(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=0.9,
                trust_remote_code=True,
                max_model_len=10240, 
            )
            self.tokenizer = self.llm.get_tokenizer()

            self.sampling_params = SamplingParams(
                n=NUM_REPEATS,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            print("[RewardActor] Model loaded successfully!", flush=True)
        except Exception as e:
            print(f"[RewardActor] CRITICAL ERROR during init: {e}", flush=True)
            traceback.print_exc()
            raise e

    def generate_rewards(self, prompts: List[Any]) -> List[List[str]]:
        """
        prompts: List[messages], where messages = List[{"role":..., "content":...}, ...]
        """
        if not prompts:
            return []
        
        str_prompts: List[str] = []
        for messages in prompts:
            try:
                prompt_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                prompt_text = json.dumps(messages, ensure_ascii=False)

            str_prompts.append(prompt_text)

        outputs = self.llm.generate(
            str_prompts, sampling_params=self.sampling_params, use_tqdm=False
        )

        batch_results: List[List[str]] = []
        for output in outputs:
            candidates = [o.text.strip() for o in output.outputs if o.text and o.text.strip()]
            batch_results.append(candidates)

        return batch_results


ACTOR_NAME_PREFIX = "Verl_Shared_Reward_Model"

def _get_or_create_actor_pool() -> List[Any]:
    global _ACTOR_POOL
    if _ACTOR_POOL:
        return _ACTOR_POOL

    actors: List[Any] = []
    for idx in range(REWARD_NUM_ACTORS):
        actor_name = ACTOR_NAME_PREFIX if REWARD_NUM_ACTORS == 1 else f"{ACTOR_NAME_PREFIX}_{idx}"
        try:
            actor = ray.get_actor(actor_name)
        except ValueError:
            try:
                actor = VLLMRewardActor.options(
                    name=actor_name,
                    lifetime="detached",
                    num_gpus=REWARD_ACTOR_GPUS_PER_WORKER,
                ).remote(
                    model_path=REWARD_MODEL_NAME,
                    tensor_parallel_size=REWARD_MODEL_TENSOR_PARALLEL_SIZE,
                )

                ray.get(actor.__ray_ready__.remote())
            except ValueError:
                actor = ray.get_actor(actor_name)

        actors.append(actor)

    _ACTOR_POOL = actors
    return _ACTOR_POOL


def _generate_rewards_parallel(prompts: List[Any]) -> List[List[str]]:
    actors = _get_or_create_actor_pool()
    if len(actors) == 1 or len(prompts) <= 1:
        return ray.get(actors[0].generate_rewards.remote(prompts))

    shard_size = max(1, math.ceil(len(prompts) / len(actors)))
    pending: List[Tuple[int, Any]] = []
    for i, actor in enumerate(actors):
        start = i * shard_size
        end = min(len(prompts), start + shard_size)
        if start >= len(prompts):
            break
        pending.append((start, actor.generate_rewards.remote(prompts[start:end])))

    ordered_outputs: List[List[str]] = [[] for _ in range(len(prompts))]
    for start, ref in pending:
        shard_outputs = ray.get(ref)
        ordered_outputs[start : start + len(shard_outputs)] = shard_outputs

    return ordered_outputs


def _build_prompt(original_q: str, original_traj: str, critique: str, variant_q: str) -> List[dict]:
    user = (
        f"Original Problem: {original_q}\n"
        f"Original Solution Trace: {original_traj}\n\n"
        f"Critique on the Original Solution: {critique}\n\n"
        f"Instruction: Using the insights from the critique above, solve the following problem.\n"
        f"Think step-by-step and strictly put the final answer in \\boxed{{}} format.\n\n"
        f"Variation Problem: {variant_q}"
    )
    return [{"role": "user", "content": user}]



def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Any = None,
    **kwargs,
) -> float:
    if data_source in {"HuggingFaceH4/MATH-500", "lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval"}:
        return float(default_compute_score(data_source, solution_str, ground_truth))

    try:
        meta = json.loads(ground_truth)
    except Exception:
        return 0.0

    original_q = meta.get("original_question") or meta.get("question", "") or ""
    original_traj = meta.get("original_trajectory") or meta.get("trajectory", "") or ""
    variants = meta.get("variants", []) or []

    if not variants or not solution_str or len(solution_str.strip()) < 5:
        return 0.0

    prompts: List[Any] = []
    gts: List[str] = []
    for var in variants:
        var_q = var.get("q") or var.get("question")
        var_a = var.get("a") or var.get("answer")
        if var_q and var_a:
            prompts.append(_build_prompt(original_q, original_traj, solution_str, var_q))
            gts.append(var_a)

    if not prompts:
        return 0.0

    try:
        preds_per_variant = _generate_rewards_parallel(prompts)
    except Exception as e:
        if DEBUG_MODE:
            print(f"[Reward Error] Ray Actor Call Failed: {e}", flush=True)
            traceback.print_exc()
        return 0.0

    scoring_tasks: List[Tuple[str, str]] = []
    for preds, gt in zip(preds_per_variant, gts):
        if not preds:
            continue
        for p in preds:
            scoring_tasks.append((p, gt))

    if not scoring_tasks:
        return 0.0

    try:
        global _SCORING_POOL
        if _SCORING_POOL is None:
            _SCORING_POOL = ProcessPoolExecutor(max_workers=SCORING_PROCESSES)
        flat_scores = list(_SCORING_POOL.map(_score_worker_func, scoring_tasks))
    except Exception as e:
        if DEBUG_MODE:
            print(f"[Reward Error] Process pool scoring failed: {e}", flush=True)
        return 0.0

    total_score = 0.0
    valid_variants = 0
    score_idx = 0

    for i, (preds, gt) in enumerate(zip(preds_per_variant, gts)):
        if not preds:
            continue

        s = 0.0
        k = 0
        for _ in preds:
            if score_idx >= len(flat_scores):
                break
            s += flat_scores[score_idx]
            score_idx += 1
            k += 1

        if k > 0:
            variant_score = s / k
            total_score += variant_score
            valid_variants += 1

            if DEBUG_MODE and i == 0:
                print(f"[Reward Debug] Variant 0 Score: {variant_score} (Preds: {k})", flush=True)

    final_score = (total_score / valid_variants) if valid_variants > 0 else 0.0

    if DEBUG_MODE:
        print(f"[Reward] Score: {final_score:.2f} (Variants: {valid_variants})", flush=True)

    return final_score
