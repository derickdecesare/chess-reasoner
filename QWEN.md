# Getting Started with Qwen2.5-3B for Local Fine-Tuning or RL

1. **Environment Setup**

   - Python 3.8+
   - Recommended packages:
     ```bash
     pip install --upgrade transformers accelerate safetensors
     ```
   - Ensure `transformers` is at least `4.38.0` (or install from source).

2. **Download & Load Model**

   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer

   model_id = "Qwen/Qwen2.5-3B"
   tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
   model = AutoModelForCausalLM.from_pretrained(
       model_id,
       trust_remote_code=True,
       device_map="auto",   # e.g. {"": "cuda:0"} for single-GPU
       torch_dtype="auto"
   )

   prompt = "Hello, my name is"
   inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
   output = model.generate(**inputs, max_new_tokens=30, temperature=0.3)
   print(tokenizer.decode(output[0], skip_special_tokens=True))
   ```
