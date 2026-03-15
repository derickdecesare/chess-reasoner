"""
Quick evaluation of the trained LoRA model on chess positions.
Loads the base model + LoRA adapter and runs inference on test positions,
showing the full <think>/<answer> output and checking move legality + quality.

Usage:
  HF_HOME=/workspace/hf_cache python3 eval_trained_model.py
  HF_HOME=/workspace/hf_cache python3 eval_trained_model.py --checkpoint 1000
  HF_HOME=/workspace/hf_cache python3 eval_trained_model.py --adapter-path /workspace/models/qwen-14B-GRPO-unsloth/checkpoint-500
  HF_HOME=/workspace/hf_cache python3 eval_trained_model.py --num-positions 50
"""

import os
os.environ["HF_HOME"] = "/workspace/hf_cache"

import argparse
import re
import sys
import torch
import pandas as pd
import chess

from unsloth import FastLanguageModel

DEFAULT_ADAPTER_DIR = "/workspace/models/qwen-14B-GRPO-unsloth"
DEFAULT_CHECKPOINT = 1750
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
    parser = argparse.ArgumentParser(description="Evaluate a trained LoRA chess model")
    parser.add_argument("--checkpoint", type=int, default=DEFAULT_CHECKPOINT,
                        help=f"Checkpoint step number (default: {DEFAULT_CHECKPOINT})")
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="Full path to adapter dir (overrides --checkpoint)")
    parser.add_argument("--num-positions", type=int, default=10,
                        help="Number of positions to evaluate (default: 10)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling test positions")
    args = parser.parse_args()

    if args.adapter_path:
        adapter_path = args.adapter_path
    else:
        adapter_path = f"{DEFAULT_ADAPTER_DIR}/checkpoint-{args.checkpoint}"

    is_hf_repo = "/" in adapter_path and not os.path.exists(adapter_path)
    if not is_hf_repo and not os.path.exists(adapter_path):
        print(f"ERROR: Adapter path does not exist: {adapter_path}")
        available = sorted(
            [d for d in os.listdir(DEFAULT_ADAPTER_DIR) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1])
        )
        print(f"Available checkpoints: {', '.join(available)}")
        sys.exit(1)

    print("=" * 70)
    print(f"EVAL: LoRA Checkpoint — {adapter_path}")
    print("=" * 70)

    eval_table = load_eval_table(EVAL_TABLE_PATH)
    print(f"Loaded {len(eval_table):,} precomputed evals")

    print(f"\nLoading base model + LoRA adapter...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=2560,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    print(f"Model loaded. GPU: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    df = pd.read_parquet(DATASET_PATH)
    print(f"Dataset: {len(df)} positions")

    test_df = df.sample(n=min(args.num_positions, len(df)), random_state=args.seed)
    print(f"Evaluating {len(test_df)} randomly sampled positions (seed={args.seed})\n")

    legal_count = 0
    good_count = 0
    total_count = 0
    eval_diffs = []

    for i, (idx, row) in enumerate(test_df.iterrows()):
        fen = row["fen"]
        pgn = row["pgn"]

        board = chess.Board(fen)
        side = "White" if board.turn else "Black"
        legal_moves = [board.san(m) for m in board.legal_moves]

        messages = [{"role": "user", "content": row["prompt"]}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=args.temperature,
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
        if eval_diff is not None and eval_diff >= 0:
            good_count += 1
        if eval_diff is not None:
            eval_diffs.append(eval_diff)

        print(f"{'='*70}")
        print(f"[{i+1}/{len(test_df)}] FEN: {fen}")
        print(f"Side to move: {side} | Legal moves: {len(legal_moves)}")
        print(f"PGN: {pgn[:100]}...")
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
    print(f"SUMMARY — {adapter_path}")
    print(f"{'='*70}")
    print(f"  Legal moves:  {legal_count}/{total_count} ({legal_count/total_count*100:.0f}%)")
    print(f"  Good moves:   {good_count}/{total_count} ({good_count/total_count*100:.0f}%) [eval_diff >= 0]")
    if eval_diffs:
        avg_diff = sum(eval_diffs) / len(eval_diffs)
        print(f"  Avg eval_diff: {avg_diff:+.0f}cp (over {len(eval_diffs)} moves with eval data)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
