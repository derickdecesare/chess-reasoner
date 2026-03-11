"""
Precompute Stockfish evaluations for all legal moves from each position in the 10k dataset.

Output: src/data/chess/precomputed_move_evals_10k.parquet
Schema:
  fen       (str)  - the board position
  move      (str)  - the legal move in SAN
  eval_diff (float)- centipawn improvement from playing this move (positive = better for side to move)
  init_eval (float)- Stockfish eval of the position before the move
  new_eval  (float)- Stockfish eval after the move

Usage:
  python src/precompute_move_evals.py
  python src/precompute_move_evals.py --depth 15 --workers 4
"""

import argparse
import chess
import chess.engine
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import sys
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "src" / "data" / "chess"
INPUT_FILE = DATA_DIR / "chess_truncated_pgns_with_fen_10k.parquet"
OUTPUT_FILE = DATA_DIR / "precomputed_move_evals_10k.parquet"

# Add src to path so we can import get_stockfish_path
sys.path.insert(0, str(REPO_ROOT / "src"))
from utils.get_stockfish_path import get_stockfish_path

MATE_SCORE = 10_000  # centipawns assigned to forced mate


# ---------------------------------------------------------------------------
# Core evaluation logic (runs in worker processes)
# ---------------------------------------------------------------------------

def evaluate_fen(args: tuple) -> list[dict]:
    """
    Given a FEN string, evaluate every legal move with Stockfish.
    Returns a list of dicts: {fen, move, eval_diff, init_eval, new_eval}

    This function is called in a subprocess, so it creates its own engine instance.
    """
    fen, stockfish_path, depth = args
    rows = []

    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        limit = chess.engine.Limit(depth=depth)

        board = chess.Board(fen)
        if board.is_game_over():
            engine.quit()
            return rows

        # Evaluate the position before any move
        try:
            init_result = engine.analyse(board, limit)
            init_eval = init_result["score"].relative.score(mate_score=MATE_SCORE)
        except Exception:
            engine.quit()
            return rows

        # Evaluate each legal move
        for move in board.legal_moves:
            move_san = board.san(move)
            board.push(move)
            try:
                new_result = engine.analyse(board, limit)
                # Score is from the perspective of the side to move AFTER the push,
                # so negate it to get the perspective of the original side to move.
                new_eval_opponent = new_result["score"].relative.score(mate_score=MATE_SCORE)
                new_eval = -new_eval_opponent  # flip back to original side's perspective
            except Exception:
                board.pop()
                continue
            board.pop()

            eval_diff = new_eval - init_eval
            rows.append({
                "fen": fen,
                "move": move_san,
                "eval_diff": float(eval_diff),
                "init_eval": float(init_eval),
                "new_eval": float(new_eval),
            })

        engine.quit()
    except Exception as e:
        print(f"  [worker error] FEN={fen[:30]}... : {e}", flush=True)

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Precompute Stockfish move evaluations")
    parser.add_argument("--depth", type=int, default=12,
                        help="Stockfish search depth per move (default: 12)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel worker processes (default: 4)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N positions (for testing)")
    args = parser.parse_args()

    print(f"Loading dataset from {INPUT_FILE}")
    df = pd.read_parquet(INPUT_FILE)
    fens = df["fen"].dropna().unique().tolist()

    if args.limit:
        fens = fens[: args.limit]
        print(f"  (limited to {args.limit} positions)")

    print(f"  {len(fens)} unique FEN positions to evaluate")
    print(f"  Stockfish depth={args.depth}, workers={args.workers}")

    stockfish_path = get_stockfish_path()
    print(f"  Stockfish binary: {stockfish_path}")

    work_items = [(fen, stockfish_path, args.depth) for fen in fens]

    all_rows = []
    errors = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(evaluate_fen, item): item[0] for item in work_items}
        with tqdm(total=len(futures), desc="Evaluating positions", unit="pos") as pbar:
            for future in as_completed(futures):
                try:
                    rows = future.result()
                    all_rows.extend(rows)
                except Exception as e:
                    errors += 1
                    print(f"  [error] {e}", flush=True)
                pbar.update(1)

    print(f"\nDone. {len(all_rows)} move evaluations ({errors} position errors).")

    if not all_rows:
        print("No data to save — check Stockfish installation.")
        return

    result_df = pd.DataFrame(all_rows)
    result_df = result_df.sort_values(["fen", "eval_diff"], ascending=[True, False])

    print(f"Saving to {OUTPUT_FILE}")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(OUTPUT_FILE, index=False)

    # Summary stats
    print(f"\nSample output:")
    print(result_df.head(10).to_string())
    print(f"\nPositions covered: {result_df['fen'].nunique()}")
    print(f"Moves per position (avg): {len(result_df) / result_df['fen'].nunique():.1f}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
