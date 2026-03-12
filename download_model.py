"""Download Qwen2.5-14B-Instruct to /workspace/hf_cache"""
import os
os.environ["HF_HOME"] = "/workspace/hf_cache"

from huggingface_hub import snapshot_download

print("Downloading Qwen/Qwen2.5-14B-Instruct to /workspace/hf_cache ...")
path = snapshot_download("Qwen/Qwen2.5-14B-Instruct")
print(f"Download complete! Model cached at: {path}")
