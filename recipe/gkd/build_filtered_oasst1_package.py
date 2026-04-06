#!/usr/bin/env python3

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset, load_dataset


CODE_PATTERNS = [
    re.compile(r"```"),
    re.compile(r"\b(def|class|import|from|return|print|SELECT|INSERT|UPDATE|DELETE)\b"),
    re.compile(r"\b(curl|flask|django|fastapi|javascript|typescript|sql|api endpoint)\b", re.IGNORECASE),
    re.compile(r"</?[a-zA-Z][^>]*>"),
]

MATH_PATTERNS = [
    re.compile(r"\\(frac|sum|int|sqrt|alpha|beta|gamma|theta|pi)"),
    re.compile(r"\\[\(\)\[\]]"),
    re.compile(r"\$\$?.+?\$\$?"),
    re.compile(r"\b(integral|derivative|equation|theorem|lemma|proof|matrix|eigenvalue|solve for)\b", re.IGNORECASE),
]

CITATION_PATTERNS = [
    re.compile(r"\b(cite|citation|citations|reference|references|bibliography)\b", re.IGNORECASE),
    re.compile(r"\b(arxiv|doi|journal|conference paper|peer-reviewed)\b", re.IGNORECASE),
    re.compile(r"\bet al\.", re.IGNORECASE),
]


def is_valid_oasst_message(message: Dict[str, Any]) -> bool:
    return (
        message.get("role") in {"prompter", "assistant"}
        and message.get("lang") == "en"
        and bool(message.get("review_result"))
        and not bool(message.get("deleted"))
        and not bool(message.get("synthetic"))
        and message.get("tree_state") == "ready_for_export"
        and isinstance(message.get("text"), str)
        and message["text"].strip() != ""
    )


def get_label_value(message: Dict[str, Any], label_name: str) -> Optional[float]:
    labels = message.get("labels")
    if not isinstance(labels, dict):
        return None

    for name, value in zip(labels.get("name", []), labels.get("value", [])):
        if name == label_name:
            return float(value)

    return None


def detect_domain_flags(path: List[Dict[str, Any]]) -> List[str]:
    flags = set()

    for message in path:
        text = message.get("text")
        if not isinstance(text, str):
            continue

        if any(pattern.search(text) for pattern in CODE_PATTERNS):
            flags.add("code")
        if any(pattern.search(text) for pattern in MATH_PATTERNS):
            flags.add("math")
        if any(pattern.search(text) for pattern in CITATION_PATTERNS):
            flags.add("citation")

    return sorted(flags)


def load_filtered_oasst1_dataset(
    dataset_name: str,
    split: str,
    target_examples: Optional[int],
    top_k: int,
    min_quality: Optional[float],
    min_helpfulness: Optional[float],
    max_examples_per_tree: Optional[int],
    seed: int,
) -> Tuple[Dataset, Dict[str, Any]]:
    raw_dialogs = load_dataset(dataset_name, split=split)
    records = list(raw_dialogs)
    valid_records = {
        record["message_id"]: record
        for record in records
        if is_valid_oasst_message(record)
    }

    candidates = []
    excluded_domain_candidates = 0
    excluded_by_domain = {"code": 0, "math": 0, "citation": 0}
    for message in valid_records.values():
        if message.get("role") != "assistant":
            continue

        rank = message.get("rank")
        if rank is None or rank >= top_k:
            continue

        quality = get_label_value(message, "quality")
        if min_quality is not None and (quality is None or quality < min_quality):
            continue

        helpfulness = get_label_value(message, "helpfulness")
        if min_helpfulness is not None and (
            helpfulness is None or helpfulness < min_helpfulness
        ):
            continue

        current = message
        path = []
        seen = set()
        valid_path = True
        while current is not None:
            message_id = current["message_id"]
            if message_id in seen:
                valid_path = False
                break
            seen.add(message_id)
            path.append(current)

            parent_id = current.get("parent_id")
            if parent_id is None:
                break

            current = valid_records.get(parent_id)
            if current is None:
                valid_path = False
                break

        if not valid_path:
            continue

        path.reverse()
        roles = [node["role"] for node in path]
        if len(path) < 2 or roles[0] != "prompter" or roles[-1] != "assistant":
            continue
        if any(left == right for left, right in zip(roles, roles[1:])):
            continue

        domain_flags = detect_domain_flags(path)
        if domain_flags:
            excluded_domain_candidates += 1
            for flag in domain_flags:
                excluded_by_domain[flag] += 1
            continue

        candidates.append(message)

    rng = random.Random(seed)
    rng.shuffle(candidates)

    selected_assistant_ids = set()
    per_tree_counts = {}
    required_message_ids = set()
    for message in candidates:
        tree_id = message.get("message_tree_id")
        current_count = per_tree_counts.get(tree_id, 0)
        if max_examples_per_tree is not None and current_count >= max_examples_per_tree:
            continue

        selected_assistant_ids.add(message["message_id"])
        per_tree_counts[tree_id] = current_count + 1

        current = valid_records.get(message["message_id"])
        while current is not None:
            required_message_ids.add(current["message_id"])
            parent_id = current.get("parent_id")
            if parent_id is None:
                break
            current = valid_records.get(parent_id)

        if target_examples is not None and len(selected_assistant_ids) >= target_examples:
            break

    filtered_records = []
    for record in records:
        message_id = record.get("message_id")
        if message_id not in required_message_ids:
            continue

        filtered_record = dict(record)
        filtered_record["is_training_target"] = message_id in selected_assistant_ids
        filtered_records.append(filtered_record)

    filtered_dialogs = Dataset.from_list(filtered_records)
    stats = {
        "dataset_name": dataset_name,
        "split": split,
        "raw_messages": len(records),
        "filtered_messages": len(filtered_records),
        "candidate_assistant_messages": len(candidates),
        "excluded_domain_candidates": excluded_domain_candidates,
        "excluded_code_candidates": excluded_by_domain["code"],
        "excluded_math_candidates": excluded_by_domain["math"],
        "excluded_citation_candidates": excluded_by_domain["citation"],
        "selected_training_targets": len(selected_assistant_ids),
        "unique_trees": len(per_tree_counts),
        "top_k": top_k,
        "min_quality": min_quality,
        "min_helpfulness": min_helpfulness,
        "max_examples_per_tree": max_examples_per_tree,
        "seed": seed,
    }
    return filtered_dialogs, stats


def write_json(
    dialogs: Dataset,
    stats: Dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "dataset_name": stats["dataset_name"],
        "split": stats["split"],
        "filter_stats": stats,
        "records": list(dialogs),
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a JSON file containing a prefiltered OASST1 dataset."
    )
    parser.add_argument(
        "--dataset-name",
        default="OpenAssistant/oasst1",
        help="Hugging Face dataset name.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to filter.",
    )
    parser.add_argument(
        "--target-examples",
        type=int,
        default=5000,
        help="Maximum number of assistant targets to keep.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Keep assistant messages with rank < top_k.",
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=None,
        help="Minimum quality label threshold.",
    )
    parser.add_argument(
        "--min-helpfulness",
        type=float,
        default=None,
        help="Optional helpfulness label threshold.",
    )
    parser.add_argument(
        "--max-examples-per-tree",
        type=int,
        default=1,
        help="Maximum number of targets to keep from each message tree.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for candidate shuffling.",
    )
    parser.add_argument(
        "--output-path",
        default="artifacts/filtered_oasst1.json",
        help="Path for the final JSON file that should be uploaded to Dropbox.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dialogs, stats = load_filtered_oasst1_dataset(
        dataset_name=args.dataset_name,
        split=args.split,
        target_examples=args.target_examples,
        top_k=args.top_k,
        min_quality=args.min_quality,
        min_helpfulness=args.min_helpfulness,
        max_examples_per_tree=args.max_examples_per_tree,
        seed=args.seed,
    )

    output_path = Path(args.output_path).resolve()
    write_json(dialogs, stats, output_path)

    print(json.dumps(stats, indent=2))
    print()
    print(f"JSON path: {output_path}")
    print("Upload the JSON file to Dropbox, then paste the shared link into hw1_260323.ipynb.")


if __name__ == "__main__":
    main()
