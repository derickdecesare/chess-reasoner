"""
Sanity-check inference script for Qwen2.5-14B-Instruct on chess prompts.

Tests:
  1. Model loads correctly in 4-bit quantization (same config as training)
  2. Tokenizer chat template works with our prompt format
  3. Model generates a response with <think>...</think> and <answer>...</answer> tags
  4. Generated move is parseable and legal via python-chess

Usage:
  HF_HOME=/workspace/hf_cache python3 sanity_check_inference.py
"""

import os
os.environ["HF_HOME"] = "/workspace/hf_cache"

import torch
import pandas as pd
import chess
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
DATASET_PATH = "src/data/chess/chess_rl_fen_pgn_prompt_10k.parquet"


def extract_move(response_text: str) -> str | None:
    """Extract move from <answer>...</answer> tags."""
    match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL)
    if match:
        move = match.group(1).strip().replace("+", "").replace("#", "")
        return move
    return None


def check_move_legal(fen: str, move_san: str) -> bool:
    """Check if a move is legal in the given position."""
    board = chess.Board(fen)
    try:
        parsed = board.parse_san(move_san)
        return parsed in board.legal_moves
    except Exception:
        return False


def main():
    print("=" * 60)
    print("SANITY CHECK: DeepSeek-R1-Distill-Qwen-14B Chess Inference")
    print("=" * 60)

    # --- Step 1: Load tokenizer and check chat template ---
    print("\n[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Show the chat template format
    test_msg = [{"role": "user", "content": "Test"}]
    formatted = tokenizer.apply_chat_template(test_msg, tokenize=False, add_generation_prompt=True)
    print(f"  Chat template format: {repr(formatted)}")
    print(f"  NOTE: Template ends with '<think>\\n' — model is primed to start reasoning")

    # --- Step 2: Load model in 4-bit (same as training) ---
    print("\n[2/4] Loading model in 4-bit quantization...")
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
    print(f"  Model loaded. Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'N/A'}")
    print(f"  GPU memory used: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # --- Step 3: Load dataset and pick test positions ---
    print("\n[3/4] Loading dataset and testing prompts...")
    df = pd.read_parquet(DATASET_PATH)

    # Test 3 diverse positions: simple opening, midgame, complex
    test_indices = [1, 500, 2000]

    for idx in test_indices:
        row = df.iloc[idx]
        prompt = row["prompt"]
        fen = row["fen"]

        print(f"\n  --- Test position idx={idx} ---")
        print(f"  FEN: {fen}")
        print(f"  PGN (truncated): {row['pgn'][:80]}...")

        # Format as chat message (DeepSeek R1 expects this)
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]
        print(f"  Prompt tokens: {prompt_len}")

        # Generate
        print(f"  Generating response...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )

        # Decode only the generated part (not the prompt)
        generated_tokens = outputs[0][prompt_len:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        print(f"  Generated tokens: {len(generated_tokens)}")
        print(f"\n  === FULL RESPONSE ({len(response)} chars) ===")
        print(response)

        # Check structure
        has_think = "</think>" in response
        has_answer = "<answer>" in response and "</answer>" in response
        print(f"\n  Has </think> tag: {has_think}")
        print(f"  Has <answer>...</answer> tags: {has_answer}")

        # Extract and validate move
        move = extract_move(response)
        if move:
            legal = check_move_legal(fen, move)
            print(f"  Extracted move: '{move}' — Legal: {legal}")
        else:
            print(f"  Could not extract a move from response")

    # --- Step 4: Summary ---
    print("\n" + "=" * 60)
    print("SANITY CHECK COMPLETE")
    print(f"  GPU memory used: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    print(f"  GPU memory peak: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    print("=" * 60)


if __name__ == "__main__":
    main()
