import os
import pandas as pd
import chess
import chess.pgn
import io

# full_moves is what we should use to filter by length if we want to

def verify_position(truncated_pgn: str, fen: str) -> bool:
    """Verify that the FEN string matches the position after playing through the truncated PGN."""
    pgn_io = io.StringIO(truncated_pgn)
    game = chess.pgn.read_game(pgn_io)
    if game is None:
        return False
        
    board = game.board()
    for move in game.mainline_moves():
        board.push(move)
        
    return board.fen() == fen


def main():
    # Path to the truncated dataset with prompt
    dataset_rel_path = os.path.join("src","data", "chess", "chess_rl_fen_pgn_prompt_100k.parquet")
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), dataset_rel_path))
    print(f"Loading dataset from: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print("ERROR: Dataset file not found!")
        return

    try:
        df = pd.read_parquet(dataset_path)
        print("Dataset loaded successfully!")
    except Exception as e:
        print(f"ERROR: Failed to load the dataset: {e}")
        return

    # Basic dataset information
    print(f"\nDataset shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    
    # Analyze specific columns
    print("\nStatistics for number_of_moves:")
    print(df['number_of_moves'].describe())
    
    print("\nStatistics for splice_point:")
    print(df['splice_point'].describe())
    
    print("\nStatistics for full_moves:")
    print(df['full_moves'].describe())

    print("\nStatistics for last_number:")
    print(df['last_number'].describe())

    print("\nStatistics for prompt:")
    print(df['prompt'].describe())
    
    # Verify a sample of positions
    print("\nVerifying a sample of positions...")
    sample_size = min(100, len(df))
    sample_df = df.sample(n=sample_size)
    verification_results = []
    
    for idx, row in sample_df.iterrows():
        is_valid = verify_position(row['pgn'], row['fen'])
        verification_results.append(is_valid)
    
    valid_count = sum(verification_results)
    print(f"Position verification results: {valid_count}/{sample_size} positions verified correctly")
    
    # Show some example truncated games
    print("\nExample truncated games (first 3):")
    for idx, row in df.head(3).iterrows():
        print(f"\nExample {idx + 1}:")
        print(f"PGN: {row['pgn']}")
        print(f"FEN: {row['fen']}")
        print(f"Prompt: {row['prompt']}")
        print("-" * 80)
    
    # Calculate distribution of game lengths
    print("\nDistribution of truncation points (full_moves):")
    move_distribution = df['full_moves'].value_counts().sort_index()
    print(move_distribution)


    # calculate distribution of prompt lengths
    print("\nDistribution of prompt lengths:")
    prompt_distribution = df['prompt'].apply(len).value_counts().sort_index()
    print(prompt_distribution)


    # calculate distribution of pgn lengths
    print("\nDistribution of pgn lengths:")
    pgn_distribution = df['pgn'].apply(len).value_counts().sort_index()
    print(pgn_distribution)

if __name__ == "__main__":
    main()