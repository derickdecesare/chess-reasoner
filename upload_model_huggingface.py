"""
upload_model_huggingface.py

This module provides a reusable function to upload fine-tuned models (full parameter,
not using LoRA) to the Hugging Face Hub.

Modifiable variables:
- MODEL_PATH: The local directory path for your fine-tuned model.
- REPO_NAME: The Hugging Face repository name to which your model will be uploaded.
- MODEL_CLASS: The class used to load the model (default: AutoModelForCausalLM).
- TOKENIZER_CLASS: The class used to load the tokenizer (default: AutoTokenizer).
- DEVICE_MAP: The device configuration for model loading (default: "auto").
- TORCH_DTYPE: The torch data type to use (default: torch.float16).
- TRUST_REMOTE_CODE: Whether to trust remote code when loading the model (default: True).

Usage:
    1. Change the variables below (MODEL_PATH, REPO_NAME, etc.) to match your model.
    2. Run this file: `python upload_model.py`
     
Make sure that you have a Hugging Face token ready, as you might be prompted to login.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login

def upload_model_to_hub(
    model_path,
    repo_name,
    model_class=AutoModelForCausalLM,
    tokenizer_class=AutoTokenizer,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
):
    """
    Uploads a full parameter fine-tuned model and its tokenizer from the local directory to
    a Hugging Face Hub repository.

    Parameters:
        model_path (str): Local directory of your fine-tuned model.
                          Change this to point to your model's directory.
        repo_name (str): Hugging Face repository name (e.g., "username/model-name").
                         Change this to your desired repo name.
        model_class (class): The model class used for loading (default: AutoModelForCausalLM).
                             Modify if you are using another model type.
        tokenizer_class (class): The tokenizer class used for loading (default: AutoTokenizer).
                                 Modify if needed.
        device_map (str or dict): Device configuration for model loading (default: "auto").
        torch_dtype: The data type for model weights (default: torch.float16).
        trust_remote_code (bool): Whether to trust remote code during loading (default: True).

    Note:
        This function is designed for models that have been fully fine-tuned
        (i.e., no use of LoRA adapters).

        Required files in the model directory (model_path):
        - A model configuration file (typically "config.json").
        - The model weights file (e.g., "pytorch_model.bin" or "model.safetensors").
        - Tokenizer files, such as:
            - "tokenizer_config.json"
            - Additional files as required by your tokenizer (e.g., "vocab.txt", "merges.txt", "tokenizer.json").
    """
    # Load tokenizer from the specified model directory.
    print("Loading tokenizer from:", model_path)
    tokenizer = tokenizer_class.from_pretrained(model_path, trust_remote_code=trust_remote_code)

    # Load the full fine-tuned model.
    print("Loading model from:", model_path)
    model = model_class.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        torch_dtype=torch_dtype
    )

    # Upload model and tokenizer to the Hugging Face Hub.
    print(f"Pushing model and tokenizer to Hugging Face repository: {repo_name}")
    model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)
    print("Upload complete!")

if __name__ == "__main__":
    # Log in to Hugging Face. This will prompt you to enter your token.
    login()

    # Change these variables to match your local model and desired repository name.
    MODEL_PATH = "models/llama-3.2-1b-instruct-finetune_png_10k_cot_1k"
    REPO_NAME = "derickio/llama-3.2-1b-instruct-finetune_png_10k_cot_1k"

    upload_model_to_hub(
        model_path=MODEL_PATH,
        repo_name=REPO_NAME,
        # You can modify additional parameters here if needed:
        # model_class=YourModelClass,
        # tokenizer_class=YourTokenizerClass,
        # device_map="auto",
        # torch_dtype=torch.float16,
        # trust_remote_code=True,
    )