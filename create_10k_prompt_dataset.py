"""
Create a 10k training dataset with prompts that include both PGN and FEN.
This matches the positions covered by precomputed_move_evals_10k.parquet
so that chess quality rewards (good_move, checkmate) actually fire during training.

Input:  src/data/chess/chess_truncated_pgns_with_fen_10k.parquet
Output: src/data/chess/chess_rl_fen_pgn_prompt_10k.parquet
"""

import pandas as pd


def create_prompt(pgn: str, fen: str) -> str:
    """
    Generate a prompt that gives the model both the PGN move history and the FEN.
    The PGN gives move context (what happened in the game so far).
    The FEN gives an exact board snapshot (piece positions, whose turn, castling rights, etc.).
    Together they give the model maximum information to reason about the position.
    """
    return f"""You are a chess grandmaster. Please analyze this chess position and provide your reasoning and next move.

    Current game (PGN):
    {pgn}

    Current position (FEN):
    {fen}

    Provide your analysis and move in the following format:

    <think>
    Your detailed reasoning, outlining key threats, piece positions, and any plans.
    </think>
    <answer>
    Your chosen move in standard algebraic notation (SAN)
    </answer>
     """


def main():
    # Load the 10k dataset (has truncated_pgn, fen, but no prompt column)
    df = pd.read_parquet("src/data/chess/chess_truncated_pgns_with_fen_10k.parquet")
    print(f"Loaded {len(df)} rows from 10k dataset")
    print(f"Columns: {df.columns.tolist()}")

    # Generate prompts with both PGN and FEN
    df["prompt"] = df.apply(lambda row: create_prompt(row["truncated_pgn"], row["fen"]), axis=1)

    # Rename truncated_pgn -> pgn to match what the training loop expects
    df = df.rename(columns={"truncated_pgn": "pgn"})

    # Drop original_pgn — not needed for training
    df = df.drop(columns=["original_pgn"])

    # Verify overlap with precomputed evals
    evals = pd.read_parquet("src/data/chess/precomputed_move_evals_10k.parquet")
    fens_evals = set(evals["fen"].unique())
    overlap = df["fen"].isin(fens_evals).sum()

    print(f"\nFinal dataset:")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Unique FENs: {df['fen'].nunique()}")
    print(f"  Rows with precomputed evals: {overlap} / {len(df)} ({100*overlap/len(df):.1f}%)")

    print(f"\n=== Sample prompt ===")
    print(df["prompt"].iloc[0])

    # Save
    output_path = "src/data/chess/chess_rl_fen_pgn_prompt_10k.parquet"
    df.to_parquet(output_path, index=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
