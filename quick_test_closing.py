"""Quick test: can the model ever close </think> and produce an answer?"""
import os
os.environ["HF_HOME"] = "/workspace/hf_cache"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    attn_implementation="flash_attention_2",
    device_map="auto",
    trust_remote_code=True,
)
print(f"Loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

# Test 1: Simple non-chess question (does it EVER close </think>?)
print("\n" + "="*60)
print("TEST 1: Simple math question (baseline — does it close </think>?)")
print("="*60)
messages = [{"role": "user", "content": "What is 2+2?"}]
formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=512, temperature=0.6, do_sample=True)
resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(resp)
print(f"\nHas </think>: {'</think>' in resp}")

# Test 2: Simple chess with minimal prompt (no FEN parsing burden)
print("\n" + "="*60)
print("TEST 2: Simple chess — minimal prompt, no FEN")
print("="*60)
messages = [{"role": "user", "content": "In chess, after 1. e4 d5 2. exd5, what should Black play? Give your move in <answer> tags."}]
formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=1024, temperature=0.6, do_sample=True)
resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(resp)
print(f"\nHas </think>: {'</think>' in resp}")
print(f"Has <answer>: {'<answer>' in resp}")

# Test 3: Our actual prompt format but with chat template
print("\n" + "="*60)
print("TEST 3: Our chess prompt format via chat template")
print("="*60)
chess_prompt = """You are a chess grandmaster. The game so far: 1. e4 d5 2. exd5

It is Black's turn. Analyze briefly, then give your move.
After your reasoning, close with </think> and put your move in <answer>Nf6</answer> format."""

messages = [{"role": "user", "content": chess_prompt}]
formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=1024, temperature=0.6, do_sample=True)
resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(resp)
print(f"\nHas </think>: {'</think>' in resp}")
print(f"Has <answer>: {'<answer>' in resp}")

# Test 4: Our full dataset prompt via chat template, lower temp
print("\n" + "="*60)
print("TEST 4: Full dataset prompt via chat template, temp=0.3")
print("="*60)
import pandas as pd
df = pd.read_parquet("src/data/chess/chess_rl_fen_pgn_prompt_10k.parquet")
sample = df.iloc[2000]
messages = [{"role": "user", "content": sample["prompt"]}]
formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=2048, temperature=0.3, do_sample=True)
resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(resp[:3000])
if len(resp) > 3000:
    print(f"\n... ({len(resp)} total chars)")
print(f"\nTokens generated: {out.shape[1] - inputs['input_ids'].shape[1]}")
print(f"Has </think>: {'</think>' in resp}")
print(f"Has <answer>: {'<answer>' in resp}")
