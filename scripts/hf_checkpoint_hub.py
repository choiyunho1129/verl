#!/usr/bin/env python3
"""Upload and download eval checkpoints via Hugging Face Hub."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

WEIGHT_PATTERNS = (
    "model*.safetensors",
    "pytorch_model*.bin",
    "model*.index.json",
    "pytorch_model*.index.json",
)
TOKENIZER_PATTERNS = (
    "tokenizer*",
    "vocab.json",
    "merges.txt",
    "spiece.model",
    "sentencepiece.bpe.model",
)


def _eprint(message: str) -> None:
    print(message, file=sys.stderr)


def _resolve_token(token: str | None) -> str | None:
    if token:
        return token
    for env_name in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(env_name)
        if value:
            return value
    return None


def _glob_has_file(root: Path, pattern: str) -> bool:
    return any(path.is_file() for path in root.glob(pattern))


def _has_model_weights(model_dir: Path) -> bool:
    return any(_glob_has_file(model_dir, pattern) for pattern in WEIGHT_PATTERNS)


def validate_eval5_model_dir(model_dir: Path) -> None:
    if not model_dir.exists() or not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"Missing config.json in {model_dir}")
    if not _has_model_weights(model_dir):
        raise FileNotFoundError(
            "Missing model weights. Expected one of: "
            "model*.safetensors, pytorch_model*.bin, model*.index.json, pytorch_model*.index.json"
        )
    if not any(_glob_has_file(model_dir, pattern) for pattern in TOKENIZER_PATTERNS):
        raise FileNotFoundError(
            "Missing tokenizer files. Expected files like tokenizer.json/tokenizer.model/vocab.json."
        )


def maybe_merge_checkpoint(args: argparse.Namespace, merged_dir: Path) -> None:
    if args.skip_merge:
        _eprint("Skipping merge due to --skip-merge.")
        return

    if _has_model_weights(merged_dir):
        _eprint(f"Merged weights already found: {merged_dir}")
        return

    if not args.actor_dir:
        raise ValueError("--actor-dir is required when merged weights are not present and --skip-merge is not set.")

    actor_dir = Path(args.actor_dir).expanduser().resolve()
    if not actor_dir.exists() or not actor_dir.is_dir():
        raise FileNotFoundError(f"Actor checkpoint directory not found: {actor_dir}")

    merge_cmd = [
        sys.executable,
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        args.backend,
        "--local_dir",
        str(actor_dir),
        "--target_dir",
        str(merged_dir),
    ]
    if args.tie_word_embedding:
        merge_cmd.append("--tie-word-embedding")
    if args.is_value_model:
        merge_cmd.append("--is-value-model")
    if args.trust_remote_code:
        merge_cmd.append("--trust-remote-code")
    if args.use_cpu_initialization:
        merge_cmd.append("--use_cpu_initialization")

    repo_root = Path(__file__).resolve().parents[1]
    _eprint(f"Running merge command: {' '.join(merge_cmd)}")
    subprocess.run(merge_cmd, check=True, cwd=str(repo_root))


def write_manifest(
    merged_dir: Path,
    repo_id: str,
    backend: str,
    actor_dir: str | None,
    revision: str | None = None,
) -> Path:
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repo_id": repo_id,
        "backend": backend,
        "source_actor_dir": actor_dir,
        "recommended_eval_script": "eval/eval_5.sh",
        "recommended_env": {
            "HF_MODEL_REPO": repo_id,
            "HF_MODEL_REVISION": revision or "main",
        },
    }
    manifest_path = merged_dir / "eval_5_hub_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return manifest_path


def upload_checkpoint(args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi

    merged_dir = Path(args.merged_dir).expanduser().resolve()
    merged_dir.mkdir(parents=True, exist_ok=True)

    maybe_merge_checkpoint(args, merged_dir)
    validate_eval5_model_dir(merged_dir)
    write_manifest(
        merged_dir=merged_dir,
        repo_id=args.repo_id,
        backend=args.backend,
        actor_dir=args.actor_dir,
        revision=args.revision,
    )

    token = _resolve_token(args.token)
    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(merged_dir),
        revision=args.revision,
        commit_message=args.commit_message,
    )
    info = api.model_info(repo_id=args.repo_id, revision=args.revision, token=token)
    revision_sha = info.sha

    print(f"Uploaded model to: {args.repo_id}")
    print(f"Resolved revision: {revision_sha}")
    print("Run eval/eval_5.sh on another server with:")
    print(f"HF_MODEL_REPO={args.repo_id} HF_MODEL_REVISION={revision_sha} bash eval/eval_5.sh")


def download_checkpoint(args: argparse.Namespace) -> None:
    from huggingface_hub import snapshot_download

    token = _resolve_token(args.token)
    download_kwargs: dict[str, object] = {
        "repo_id": args.repo_id,
        "repo_type": "model",
        "token": token,
        "local_files_only": args.local_files_only,
    }
    if args.revision:
        download_kwargs["revision"] = args.revision
    if args.cache_dir:
        download_kwargs["cache_dir"] = str(Path(args.cache_dir).expanduser())
    if args.local_dir:
        download_kwargs["local_dir"] = str(Path(args.local_dir).expanduser())

    resolved_path = Path(snapshot_download(**download_kwargs)).resolve()
    if not args.skip_validation:
        validate_eval5_model_dir(resolved_path)

    if args.print_path_only:
        print(str(resolved_path))
        return

    print(f"Downloaded model path: {resolved_path}")
    if args.revision:
        print(f"Requested revision: {args.revision}")
    print("You can run:")
    print(f"MODEL_PATH={resolved_path} bash eval/eval_5.sh")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Utilities for eval checkpoint publish/download on Hugging Face Hub.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    upload_parser = subparsers.add_parser("upload", help="Merge (optional) and upload a checkpoint folder.")
    upload_parser.add_argument("--repo-id", required=True, type=str, help="Target Hugging Face model repo id.")
    upload_parser.add_argument(
        "--merged-dir",
        required=True,
        type=str,
        help="Directory containing merged Hugging Face weights (or target dir to create them).",
    )
    upload_parser.add_argument(
        "--actor-dir",
        type=str,
        default=None,
        help="Actor checkpoint directory for merge source. Required if merged-dir has no HF weights.",
    )
    upload_parser.add_argument("--backend", choices=("fsdp", "megatron"), default="fsdp")
    upload_parser.add_argument("--skip-merge", action="store_true", help="Skip merge and upload merged-dir as-is.")
    upload_parser.add_argument("--private", action="store_true", help="Create/use private model repo.")
    upload_parser.add_argument("--token", type=str, default=None, help="HF token. Defaults to HF_TOKEN env.")
    upload_parser.add_argument("--revision", type=str, default="main", help="Target branch or revision.")
    upload_parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload merged eval checkpoint",
        help="Commit message for upload.",
    )
    upload_parser.add_argument("--tie-word-embedding", action="store_true")
    upload_parser.add_argument("--is-value-model", action="store_true")
    upload_parser.add_argument("--trust-remote-code", action="store_true")
    upload_parser.add_argument("--use-cpu-initialization", action="store_true")

    download_parser = subparsers.add_parser("download", help="Download a model snapshot to local path.")
    download_parser.add_argument("--repo-id", required=True, type=str, help="Hugging Face model repo id.")
    download_parser.add_argument("--revision", type=str, default=None, help="Commit SHA/tag/branch.")
    download_parser.add_argument("--cache-dir", type=str, default=None, help="HF cache directory.")
    download_parser.add_argument("--local-dir", type=str, default=None, help="Explicit destination directory.")
    download_parser.add_argument("--token", type=str, default=None, help="HF token. Defaults to HF_TOKEN env.")
    download_parser.add_argument("--skip-validation", action="store_true", help="Skip eval_5 compatibility check.")
    download_parser.add_argument(
        "--print-path-only",
        action="store_true",
        help="Print only resolved local model path. Useful for shell command substitution.",
    )
    download_parser.add_argument("--local-files-only", action="store_true", help="Use local cache only.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "upload":
        upload_checkpoint(args)
    elif args.command == "download":
        download_checkpoint(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
