from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from huggingface_hub import login, HfApi

# Login to HuggingFace
login()  # Will prompt for your token

# Load base model and LoRA adapter (same as in chat.py)
base_model_id = "Qwen/Qwen2.5-3B"
lora_path = "models/qwen_instruct_20250123_041812_final"

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16
)

# Load our fine-tuned LoRA model
model = PeftModel.from_pretrained(base_model, lora_path)

# Push to hub
model_name = "derickio/qwen-instruct-lora"
print(f"Pushing model to {model_name}...")
model.push_to_hub(model_name)
tokenizer.push_to_hub(model_name)  # Also push tokenizer config

print("Upload complete!")