import os
from transformers import AutoTokenizer

def download_and_save_tokenizer():
    # The original model identifier from which you've fine-tuned.
    # This should be the base model used in fine-tuning.
    original_model = "meta-llama/Llama-3.2-1B-Instruct"  
    print(f"Downloading tokenizer from {original_model}...")
    tokenizer = AutoTokenizer.from_pretrained(
        original_model,
        trust_remote_code=True
    )
    
    # Compute the local model directory (where your fine-tuned model is saved)
    rel_path = "models/llama-3.2-1b-instruct-finetune_png_4k"
    local_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", rel_path))
    print(f"Saving tokenizer to {local_model_path} ...")
    
    # Save the tokenizer files into the same directory as the fine-tuned model.
    tokenizer.save_pretrained(local_model_path)
    print("Tokenizer saved successfully.")

if __name__ == "__main__":
    download_and_save_tokenizer() 