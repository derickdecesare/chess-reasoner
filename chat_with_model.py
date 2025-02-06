import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_path, local_files_only=True):
    """
    Load the fine-tuned model and its tokenizer from the given model_path.
    
    Parameters:
        model_path (str): Either the local directory path for your model or a hub identifier.
        local_files_only (bool): If True, ensures that only local files are used.
                                 Set to False to allow downloading from the Hugging Face Hub.
    
    Returns:
        model, tokenizer, device: The loaded model, tokenizer, and the device used.
    """
    # Setup device for any accelerator (CUDA, MPS, or CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load the model and tokenizer from local files
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        local_files_only=local_files_only
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only)
    
    # Only move the model to MPS if applicable -- for other devices, device_map should handle allocation
    if device.type == "mps":
        model = model.to(device)
    return model, tokenizer, device

def chat_with_model(path="models/llama-3.2-1b-instruct-finetune_png_100k", local_files_only=True):
    """
    Provides an interactive command-line chat interface with the fine-tuned model.
    Every new input is treated as a fresh prompt.
    """
    if local_files_only:
        model_path = os.path.abspath(path)
        print(f"Loading model from local path: {model_path}")
    else:
        model_path = path
        print(f"Loading model from Hugging Face Hub: {model_path}")
    
    model, tokenizer, device = load_model_and_tokenizer(model_path, local_files_only=local_files_only)

    print("\nInteractive chat mode. Type 'quit' or 'exit' to stop.\n")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ["quit", "exit"]:
            break
        
        # Tokenize the prompt
        inputs = tokenizer(user_input, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate a response: adjust max_new_tokens and sampling parameters as needed
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # Decode the generated tokens. Remove the input prompt, if echoed.
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(user_input):].strip()
        
        print("Model:", response)
        
if __name__ == "__main__":
    # Set the source for the model. This can be either:
    # a) A local path (e.g., "models/llama-3.2-1b-instruct-finetune_png_100k")
    # b) A Hugging Face Hub repository identifier (e.g., "derickio/llama-3.2-1b-instruct-finetune_png_10k")
    
    # Example for loading from the Hugging Face Hub:
    model_source = "derickio/llama-3.2-1b-instruct-finetune_png_10k"
    use_local = False  # Set to False to pull directly from the Hub
    
    # Example for loading a local model:
    # model_source = "models/llama-3.2-1b-instruct-finetune_png_100k"
    # use_local = True
    
    chat_with_model(path=model_source, local_files_only=use_local)