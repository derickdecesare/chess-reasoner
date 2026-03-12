"""
Quick evaluation of the trained LoRA model on chess positions.
Loads the base model + LoRA adapter and runs inference on test positions,
showing the full <think>/<answer> output and checking move legality + quality.

Usage:
  HF_HOME=/workspace/hf_cache python3 eval_trained_model.py
"""

import os
os.environ["HF_HOME"] = "/workspace/hf_cache"

import re
import sys
import torch
import pandas as pd
import chess

from unsloth import FastLanguageModel

MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
ADAPTER_PATH = "/workspace/models/qwen-14B-GRPO-unsloth/final"
DATASET_PATH = "src/data/chess/chess_rl_fen_pgn_prompt_10k.parquet"
EVAL_TABLE_PATH = "src/data/chess/precomputed_move_evals_10k.parquet"


def load_eval_table(path):
    try:
        df = pd.read_parquet(path)
        return {(row.fen, row.move): row.eval_diff for row in df.itertuples(index=False)}
    except Exception:
        return {}


def check_move(fen, move_san, eval_table):
    board = chess.Board(fen)
    try:
        parsed = board.parse_san(move_san)
        legal = parsed in board.legal_moves
    except Exception:
        return False, None, False

    canonical = board.san(parsed) if legal else move_san
    eval_diff = eval_table.get((fen, canonical))

    temp = board.copy()
    temp.push(parsed)
    is_mate = temp.is_checkmate()

    return legal, eval_diff, is_mate


def main():
    print("=" * 70)
    print("EVAL: Trained LoRA Model on Chess Positions")
    print("=" * 70)

    eval_table = load_eval_table(EVAL_TABLE_PATH)
    print(f"Loaded {len(eval_table):,} precomputed evals")

    print(f"\nLoading base model + LoRA adapter from {ADAPTER_PATH}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_PATH,
        max_seq_length=2560,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    print(f"Model loaded. GPU: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    df = pd.read_parquet(DATASET_PATH)
    print(f"Dataset: {len(df)} positions\n")

    test_indices = [0, 1, 10, 50, 100, 200, 500, 1000, 2000, 5000]
    legal_count = 0
    total_count = 0

    for idx in test_indices:
        row = df.iloc[idx]
        fen = row["fen"]
        pgn = row["pgn"]

        board = chess.Board(fen)
        legal_moves = [board.san(m) for m in board.legal_moves]

        messages = [{"role": "user", "content": row["prompt"]}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
        response = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)

        move_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        move = move_match.group(1).strip() if move_match else None

        legal, eval_diff, is_mate = (False, None, False)
        if move:
            legal, eval_diff, is_mate = check_move(fen, move, eval_table)

        total_count += 1
        if legal:
            legal_count += 1

        print(f"{'='*70}")
        print(f"Position {idx} | FEN: {fen}")
        print(f"PGN: {pgn[:100]}...")
        print(f"Legal moves ({len(legal_moves)}): {', '.join(legal_moves[:15])}{'...' if len(legal_moves) > 15 else ''}")
        print(f"\n--- MODEL OUTPUT ---")
        print(response[:1500])
        if len(response) > 1500:
            print(f"... ({len(response)} chars total)")
        print(f"\n--- VERDICT ---")
        if move:
            eval_str = f"{eval_diff:+.0f}cp" if eval_diff is not None else "N/A"
            mate_str = " *** CHECKMATE! ***" if is_mate else ""
            status = "LEGAL" if legal else "ILLEGAL"
            print(f"Move: {move} | {status} | Eval: {eval_str}{mate_str}")
        else:
            print(f"Move: NONE (no <answer> tag found)")
        print()

    print(f"{'='*70}")
    print(f"SUMMARY: {legal_count}/{total_count} legal moves ({legal_count/total_count*100:.0f}%)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
