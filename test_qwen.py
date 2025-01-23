import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_id="Qwen/Qwen2.5-3B"):
    print(f"Loading model {model_id}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16  # Using float16 for efficiency
    )
    
    return model, tokenizer

def test_generation(model, tokenizer, prompt="Give me a simple chess opening move:"):
    print(f"\nTesting with prompt: {prompt}")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nModel response:\n{response}")

def main():
    # Load model and tokenizer
    model, tokenizer = load_model()
    
    # Test with a few different prompts
    test_prompts = [
        "Give me a simple chess opening move:",
        # "Analyze this chess position: 1. e4 e5 2. Nf3",
        # "What's a good chess strategy for beginners?",

    ]
    
    for prompt in test_prompts:
        test_generation(model, tokenizer, prompt)

if __name__ == "__main__":
    main()