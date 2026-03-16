#!/usr/bin/env python3
"""Upload selected VERL LoRA checkpoints to Hugging Face Hub."""

from __future__ import annotations

import argparse
import os
import posixpath
from pathlib import Path

WEIGHT_PATTERNS = (
    "model*.safetensors",
    "pytorch_model*.bin",
    "model*.index.json",
    "pytorch_model*.index.json",
)
TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def _resolve_token(token: str | None) -> str | None:
    if token:
        return token
    for env_name in TOKEN_ENV_VARS:
        value = os.environ.get(env_name)
        if value:
            return value
    return None


def _glob_has_file(root: Path, pattern: str) -> bool:
    return any(path.is_file() for path in root.glob(pattern))


def _has_model_weights(model_dir: Path) -> bool:
    return any(_glob_has_file(model_dir, pattern) for pattern in WEIGHT_PATTERNS)


def _validate_upload_dir(upload_dir: Path, source_subdir: str) -> None:
    if not upload_dir.is_dir():
        raise FileNotFoundError(f"Upload source directory not found: {upload_dir}")

    normalized = source_subdir.strip("/").replace("\\", "/")
    if normalized == "actor/huggingface":
        if not (upload_dir / "config.json").is_file():
            raise FileNotFoundError(f"Missing config.json in {upload_dir}")
        if not _has_model_weights(upload_dir):
            raise FileNotFoundError(
                "Missing model weights in actor/huggingface. "
                "Expected files like model*.safetensors or model*.index.json."
            )


def _collect_step_sources(
    checkpoint_root: Path,
    steps: list[int],
    source_subdir: str,
    skip_missing: bool,
) -> list[tuple[int, Path]]:
    sources: list[tuple[int, Path]] = []

    for step in steps:
        step_dir = checkpoint_root / f"global_step_{step}"
        upload_dir = step_dir / source_subdir
        if not upload_dir.is_dir():
            message = f"Missing step checkpoint directory for step {step}: {upload_dir}"
            if skip_missing:
                print(f"[WARN] {message}")
                continue
            raise FileNotFoundError(message)
        _validate_upload_dir(upload_dir, source_subdir)
        sources.append((step, upload_dir))

    if not sources:
        raise FileNotFoundError("No valid checkpoint directories found to upload.")

    return sources


def _build_repo_path(path_prefix: str, step: int) -> str:
    prefix = path_prefix.strip("/")
    step_name = f"global_step_{step}"
    return posixpath.join(prefix, step_name) if prefix else step_name


def upload_steps(args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi

    checkpoint_root = Path(args.checkpoint_root).expanduser().resolve()
    if not checkpoint_root.is_dir():
        raise FileNotFoundError(f"Checkpoint root not found: {checkpoint_root}")

    source_subdir = args.source_subdir.strip("/")
    sources = _collect_step_sources(
        checkpoint_root=checkpoint_root,
        steps=args.steps,
        source_subdir=source_subdir,
        skip_missing=args.skip_missing,
    )

    token = _resolve_token(args.token)
    api = HfApi(token=token)
    if not args.dry_run:
        api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    print(f"repo_id={args.repo_id}")
    print(f"revision={args.revision}")
    print(f"source_subdir={source_subdir}")
    print(f"steps={[step for step, _ in sources]}")

    for step, folder_path in sources:
        path_in_repo = _build_repo_path(args.path_prefix, step)
        commit_message = args.commit_message.format(step=step, subdir=source_subdir)

        print(f"[UPLOAD] step={step} folder={folder_path} -> {path_in_repo}")
        if args.dry_run:
            continue

        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=str(folder_path),
            path_in_repo=path_in_repo,
            revision=args.revision,
            commit_message=commit_message,
        )

    print("Completed checkpoint upload.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upload selected VERL LoRA checkpoints to Hugging Face Hub.")
    parser.add_argument(
        "--checkpoint-root",
        required=True,
        type=str,
        help="Root checkpoint directory containing global_step_* folders.",
    )
    parser.add_argument("--repo-id", required=True, type=str, help="Target Hugging Face model repo id.")
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=[50, 100],
        help="Checkpoint steps to upload (e.g. --steps 50 100).",
    )
    parser.add_argument(
        "--source-subdir",
        type=str,
        default="actor/huggingface",
        help="Subdirectory inside each global_step_* folder to upload.",
    )
    parser.add_argument(
        "--path-prefix",
        type=str,
        default="checkpoints",
        help="Remote folder prefix in repo. Empty string uploads to repo root.",
    )
    parser.add_argument("--revision", type=str, default="main", help="Target branch or revision.")
    parser.add_argument("--private", action="store_true", help="Create/use private model repo.")
    parser.add_argument("--token", type=str, default=None, help="HF token. Defaults to HF_TOKEN env vars.")
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload LoRA checkpoint global_step_{step} ({subdir})",
        help="Commit message template. Supports {step} and {subdir}.",
    )
    parser.add_argument("--skip-missing", action="store_true", help="Skip missing step folders.")
    parser.add_argument("--dry-run", action="store_true", help="Print what will be uploaded without pushing.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    upload_steps(args)


if __name__ == "__main__":
    main()
