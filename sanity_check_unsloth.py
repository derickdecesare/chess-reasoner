"""
Sanity-check script for the Unsloth + vLLM inference path.

Tests (in order):
  1. All imports resolve (torch, unsloth, vllm, chess, trl, etc.)
  2. Model loads via FastLanguageModel with 4-bit + vLLM backend
  3. LoRA adapters apply correctly
  4. Generation works on 3 chess positions from the training dataset
  5. Reward function imports from rl_training_loop_trl work
  6. Precomputed eval table loads

Usage:
  HF_HOME=/workspace/hf_cache python3 sanity_check_unsloth.py
"""

import os
import sys
import re

os.environ["HF_HOME"] = "/workspace/hf_cache"

def step(num, total, msg):
    print(f"\n[{num}/{total}] {msg}")

def main():
    TOTAL = 6
    print("=" * 60)
    print("SANITY CHECK: Unsloth + vLLM Inference Path")
    print("=" * 60)

    # --- 1. Imports ---
    step(1, TOTAL, "Checking imports...")
    import torch
    print(f"  torch {torch.__version__}  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    import transformers; print(f"  transformers {transformers.__version__}")
    import trl; print(f"  trl {trl.__version__}")
    import peft; print(f"  peft {peft.__version__}")
    import datasets; print(f"  datasets {datasets.__version__}")
    import chess; print(f"  python-chess {chess.__version__}")
    import pandas; print(f"  pandas {pandas.__version__}")

    from unsloth import FastLanguageModel
    print(f"  unsloth OK")
    import vllm; print(f"  vllm {vllm.__version__}")

    if not torch.cuda.is_available():
        print("\n  FATAL: No GPU detected. Cannot continue.")
        sys.exit(1)

    # --- 2. Load model via Unsloth ---
    step(2, TOTAL, "Loading model via FastLanguageModel (4-bit + vLLM)...")
    MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=1280,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=64,
        gpu_memory_utilization=0.6,
    )
    print(f"  Model loaded: {MODEL_NAME}")
    print(f"  GPU memory used: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # --- 3. Apply LoRA adapters ---
    step(3, TOTAL, "Applying LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        lora_alpha=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        use_gradient_checkpointing="unsloth",
    )
    print(f"  LoRA applied (r=64, alpha=16)")
    print(f"  GPU memory used: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # --- 4. Test generation on chess positions ---
    step(4, TOTAL, "Testing generation on 3 chess positions...")
    DATASET_PATH = "src/data/chess/chess_rl_fen_pgn_prompt_10k.parquet"
    df = pandas.read_parquet(DATASET_PATH)
    print(f"  Dataset loaded: {len(df)} positions")

    test_indices = [1, 500, 2000]
    for idx in test_indices:
        row = df.iloc[idx]
        prompt = row["prompt"]
        fen = row["fen"]

        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
        prompt_len = inputs["input_ids"].shape[1]

        print(f"\n  --- Position idx={idx} (prompt: {prompt_len} tokens) ---")
        print(f"  FEN: {fen}")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
        response = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
        print(f"  Generated {len(outputs[0]) - prompt_len} tokens")

        has_think = "</think>" in response
        has_answer = "<answer>" in response and "</answer>" in response
        move_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        move = move_match.group(1).strip() if move_match else None

        legal = False
        if move:
            try:
                board = chess.Board(fen)
                parsed = board.parse_san(move)
                legal = parsed in board.legal_moves
            except Exception:
                pass

        print(f"  Format OK: think={has_think} answer={has_answer}")
        print(f"  Move: {move or 'NONE'}  Legal: {legal}")
        if len(response) < 500:
            print(f"  Response: {response}")
        else:
            print(f"  Response (truncated): {response[:300]}...")

    # --- 5. Test reward function imports ---
    step(5, TOTAL, "Testing reward function imports from rl_training_loop_trl...")
    sys.path.insert(0, "src")
    from rl_training_loop_trl import (
        move_analysis_log_func,
        legal_move_reward_func,
        good_move_reward_func,
        checkmate_reward_func,
        strict_format_reward_func,
        soft_format_reward_func,
    )
    print(f"  All 6 reward functions imported OK")

    # --- 6. Verify precomputed eval table ---
    step(6, TOTAL, "Checking precomputed eval table...")
    from rl_training_loop_trl import EVAL_TABLE
    print(f"  Eval table: {len(EVAL_TABLE):,} entries")
    if len(EVAL_TABLE) == 0:
        print("  WARNING: Eval table is empty — good_move_reward_func will always return 0.0")
    else:
        sample_key = next(iter(EVAL_TABLE))
        print(f"  Sample entry: {sample_key} -> {EVAL_TABLE[sample_key]}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SANITY CHECK COMPLETE — all systems go")
    print(f"  GPU memory used: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    print(f"  GPU memory peak: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    print("=" * 60)


if __name__ == "__main__":
    main()
