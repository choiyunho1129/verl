from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ACECODER_ROOT = REPO_ROOT / "classifer_training" / "external" / "AceCoder"


def _resolve_acecoder_root(acecoder_root: Path | str | None = None) -> Path | None:
    if acecoder_root is not None:
        return Path(acecoder_root).expanduser().resolve()
    env_root = os.environ.get("ACECODER_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    if DEFAULT_ACECODER_ROOT.exists():
        return DEFAULT_ACECODER_ROOT.resolve()
    return None


def _load_eval_module(acecoder_root: Path | str | None = None):
    root = _resolve_acecoder_root(acecoder_root)
    if root is not None:
        src_dir = root / "src"
        if src_dir.exists():
            src_text = str(src_dir)
            if src_text not in sys.path:
                sys.path.insert(0, src_text)

    try:
        from acecoder import eval_test_cases  # type: ignore
    except Exception as exc:
        raise ImportError(
            "Could not import AceCoder's official test-case evaluator. "
            "Install AceCoder or clone it under classifer_training/external/AceCoder, "
            "and install its runtime dependencies such as evalplus and termcolor."
        ) from exc
    return eval_test_cases


def normalize_test_cases(test_cases: Any) -> list[str]:
    if isinstance(test_cases, str):
        stripped = test_cases.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return [stripped]
        return normalize_test_cases(parsed)
    if isinstance(test_cases, Sequence) and not isinstance(test_cases, (bytes, bytearray)):
        return [str(item) for item in test_cases if str(item).strip()]
    return []


def evaluate_acecode_response(
    generated_text: str,
    test_cases: Any,
    *,
    acecoder_root: Path | str | None = None,
    min_time_limit: float = 1.0,
    gt_time_limit_factor: float = 4.0,
) -> dict[str, Any]:
    tests = normalize_test_cases(test_cases)
    if not tests:
        return {
            "score": 0.0,
            "pass_rate": 0.0,
            "passed_all": False,
            "num_passed": 0,
            "num_tests": 0,
            "status": "missing_tests",
            "per_test": [],
        }

    eval_test_cases = _load_eval_module(acecoder_root)
    try:
        entry_point = eval_test_cases.get_entry_point_from_test_case(tests[0])
        result = eval_test_cases.check_correctness_assert(
            task_id=0,
            completion_id=0,
            entry_point=entry_point,
            solution=str(generated_text or ""),
            assert_tests=tests,
            dataset="acecode",
            fast_check=False,
            min_time_limit=float(min_time_limit),
            gt_time_limit_factor=float(gt_time_limit_factor),
            extract_solution=True,
        )
        eval_results = dict(result.get("eval_results") or {})
    except Exception as exc:
        return {
            "score": 0.0,
            "pass_rate": 0.0,
            "passed_all": False,
            "num_passed": 0,
            "num_tests": len(tests),
            "status": "evaluator_error",
            "error": repr(exc),
            "per_test": [],
        }

    details = list(eval_results.get("details") or [])
    per_test = []
    num_passed = 0
    for item in details:
        if isinstance(item, dict):
            did_pass = bool(item.get("pass"))
            num_passed += int(did_pass)
            per_test.append(
                {
                    "pass": did_pass,
                    "reason": item.get("reason"),
                    "error_message": item.get("error_message"),
                    "time_limit": float(item["time_limit"]) if item.get("time_limit") is not None else None,
                }
            )
    num_tests = len(tests)
    pass_rate = float(eval_results.get("pass_rate", num_passed / num_tests if num_tests else 0.0))
    return {
        "score": pass_rate,
        "pass_rate": pass_rate,
        "passed_all": bool(num_tests > 0 and num_passed == num_tests),
        "num_passed": int(num_passed),
        "num_tests": int(num_tests),
        "status": str(eval_results.get("status", "")),
        "code_error": eval_results.get("code_error"),
        "per_test": per_test,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test AceCoder official test-case verification.")
    parser.add_argument("--acecoder-root", type=Path, default=None)
    parser.add_argument("--generated-text", default="def add(a, b):\n    return a + b")
    parser.add_argument("--test-case", action="append", default=["assert add(1, 2) == 3"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_acecode_response(
        args.generated_text,
        args.test_case,
        acecoder_root=args.acecoder_root,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
