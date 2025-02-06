import chess.pgn
import pandas as pd
from tqdm import tqdm
import zstandard as zstd
import io
import os
from chess.pgn import StringExporter
import re

def count_games(input_file: str, log_interval: int = 10000) -> int:
    """
    Counts the total number of games in a compressed PGN file.
    
    Parameters:
      input_file (str): Path to the compressed PGN file.
      log_interval (int): How often to log the count (every N games).
      
    Returns:
      total_games (int): The total number of games in the file.
    """
    total_games = 0
    dctx = zstd.ZstdDecompressor()
    with open(input_file, 'rb') as f:
        with dctx.stream_reader(f) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            while True:
                game = chess.pgn.read_game(text_stream)
                if game is None:
                    break
                total_games += 1
                if total_games % log_interval == 0:
                    print(f"Counted {total_games} games so far...")
    return total_games


def clean_pgn(pgn: str) -> str:
    """
    Fallback cleaning: Remove any stray clock information and game result tokens.
    (Typically, clock info will be in comments, which are already removed by the exporter.)
    Also we remove the game result tokens at the end of the PGN string.
    """
    # Remove stray clock information
    pattern = r"\s*\{\s*\[%clk\s+[^\]]+\]\s*\}"
    cleaned = re.sub(pattern, "", pgn)
    cleaned = " ".join(cleaned.split())
    
    # Remove game result tokens like "1-0", "0-1", or "1/2-1/2" at the end of the PGN string
    cleaned = re.sub(r'\s*(1-0|0-1|1/2-1/2)\s*$', '', cleaned)
    
    return cleaned

def process_game(game, elo_threshold=1800):
    """
    Process a single PGN game:
      - Extract White and Black Elo from headers, calculate average Elo.
      - Filter out games with average Elo below the given threshold.
      - Remove headers from the game so that only moves starting at '1.' remain.
    
    Returns:
      A dictionary with keys 'pgn' (the moves only) and 'elo' (the average Elo),
      or None if the game does not meet the criteria.
    """
    try:
        white_elo = int(game.headers.get("WhiteElo", 0))
        black_elo = int(game.headers.get("BlackElo", 0))
    except Exception as e:
        return None

    avg_elo = (white_elo + black_elo) // 2
    if avg_elo < elo_threshold:
        return None
    
    # Export the game moves in clean format (like "1. e4 b6 2. Nf3 ...")
    exporter = StringExporter(headers=False, variations=False, comments=False)
    moves = game.accept(exporter)
    moves = clean_pgn(moves) # remove game outcome tokens and fallback cleaning


    if not moves.strip():
        return None

    return {"pgn": moves, "elo": avg_elo}

def create_pgn_dataset(input_file="lichess_db_standard_rated_2024-10.pgn.zst",
                       output_file="src/data/chess/chess_pgn_dataset_100k.parquet",
                       num_games=100000,  # You can adjust this number (e.g. 100000 for a larger dataset)
                       elo_threshold=1800):
    """
    Process a large Lichess PGN file and produce a dataset of processed PGN strings.

    Parameters:
      input_file (str): Path to the compressed PGN file (.pgn.zst)
      output_file (str): Where to save the processed dataset (in Parquet format)
      num_games (int): The target number of games (that satisfy the Elo criteria) to process.
      elo_threshold (int): Minimum average Elo for a game to be included.
    """


    # print("Counting total number of games in the PGN file...")
    # total_games_in_file = count_games(input_file)
    # print(f"Total games in file: {total_games_in_file}")
    games_data = []
    processed_games = 0
    errors = 0

    print(f"Processing up to {num_games} games from {input_file} with average Elo >= {elo_threshold} ...")
    
    # Create a Zstandard decompressor and open the file.
    dctx = zstd.ZstdDecompressor()
    with open(input_file, 'rb') as f:
        with dctx.stream_reader(f) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            
            # Use tqdm to keep track of processing progress.
            pbar = tqdm(total=num_games, desc="Processing games")
            while processed_games < num_games:
                try:
                    game = chess.pgn.read_game(text_stream)
                    if game is None:
                        break  # End of file reached.
                    processed = process_game(game, elo_threshold=elo_threshold)
                    if processed is not None:
                        games_data.append(processed)
                        processed_games += 1
                        pbar.update(1)
                except Exception as e:
                    errors += 1
                    print(f"Error processing game: {e}")
            pbar.close()

    print(f"Processed {processed_games} games (skipped {errors} with errors).")
    
    # Create a DataFrame from the list of processed games.
    df = pd.DataFrame(games_data)

    # Print a few samples for verification before saving
    print("\nSample of processed games (first few):")
    for i in range(min(5, len(df))):
        print(f"Game {i+1} PGN: {df.iloc[i]['pgn']}")

    
    # Ensure the output directory exists.
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save the DataFrame as a Parquet file.
    df.to_parquet(output_file, index=False)
    print(f"Dataset saved to {output_file}. Total examples: {len(df)}")

if __name__ == "__main__":
    create_pgn_dataset()