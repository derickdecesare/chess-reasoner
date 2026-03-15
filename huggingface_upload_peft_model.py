"""
Upload a trained LoRA adapter (+ tokenizer) to Hugging Face Hub.

Usage:
  HF_HOME=/workspace/hf_cache python3 huggingface_upload_peft_model.py
  HF_HOME=/workspace/hf_cache python3 huggingface_upload_peft_model.py --checkpoint 1000
  HF_HOME=/workspace/hf_cache python3 huggingface_upload_peft_model.py --checkpoint 1750 --repo derickio/my-chess-model
"""

import os
os.environ["HF_HOME"] = "/workspace/hf_cache"
os.environ["TMPDIR"] = "/workspace/tmp"
os.environ["XET_CACHE_DIR"] = "/workspace/tmp/xet"
os.makedirs("/workspace/tmp/xet", exist_ok=True)

import argparse
import sys
from huggingface_hub import login, HfApi, upload_folder

DEFAULT_ADAPTER_DIR = "/workspace/models/qwen-14B-GRPO-unsloth"
DEFAULT_CHECKPOINT = 1750
DEFAULT_REPO = "derickio/qwen-14B-chess-reasoner-GRPO"


def main():
    parser = argparse.ArgumentParser(description="Upload LoRA adapter to Hugging Face Hub")
    parser.add_argument("--checkpoint", type=int, default=DEFAULT_CHECKPOINT,
                        help=f"Checkpoint step number (default: {DEFAULT_CHECKPOINT})")
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="Full path to adapter dir (overrides --checkpoint)")
    parser.add_argument("--repo", type=str, default=DEFAULT_REPO,
                        help=f"HuggingFace repo name (default: {DEFAULT_REPO})")
    parser.add_argument("--private", action="store_true",
                        help="Make the repo private")
    args = parser.parse_args()

    if args.adapter_path:
        adapter_path = args.adapter_path
    else:
        adapter_path = f"{DEFAULT_ADAPTER_DIR}/checkpoint-{args.checkpoint}"

    if not os.path.exists(adapter_path):
        print(f"ERROR: Adapter path does not exist: {adapter_path}")
        available = sorted(
            [d for d in os.listdir(DEFAULT_ADAPTER_DIR) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1])
        )
        print(f"Available checkpoints: {', '.join(available)}")
        sys.exit(1)

    print(f"Adapter path: {adapter_path}")
    print(f"Contents: {os.listdir(adapter_path)}")
    print(f"Target repo: {args.repo}")

    token = os.environ.get("HF_TOKEN")
    if token:
        print("Using HF_TOKEN from environment variable")
    else:
        login()
        token = None

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo, exist_ok=True, private=args.private)

    print(f"\nUploading {adapter_path} → {args.repo} ...")
    upload_folder(
        folder_path=adapter_path,
        repo_id=args.repo,
        repo_type="model",
        token=token,
        commit_message=f"Upload LoRA adapter from {os.path.basename(adapter_path)}",
        ignore_patterns=["optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin"],
    )

    print(f"\nDone! Model available at: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()