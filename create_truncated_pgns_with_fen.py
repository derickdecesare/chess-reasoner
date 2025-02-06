"""
create_truncated_pgns_with_fen.py

This script creates a new dataset of truncated chess games from a cleaned PGN dataset.
For each game in the input dataset, it splices the game at a random point in the
opening (before the fullmove number exceeds a given threshold, default=24) and records:
  - The truncated PGN up to that move.
  - The FEN string of the board state at that point.

The output dataset is saved as a Parquet file for further processing.

Requirements:
  - A cleaned PGN dataset in Parquet format (e.g. "src/data/chess/chess_pgn_dataset_10k.parquet" or similar)
  - pandas, chess, tqdm, and pyarrow installed.
"""

import os
import io
import random
import pandas as pd
import chess
import chess.pgn
from tqdm import tqdm

# Configuration
# Path to the input cleaned PGN dataset (change this as needed)
INPUT_DATASET = "src/data/chess/chess_pgn_dataset_100k.parquet"
# Path to the output truncated dataset
OUTPUT_DATASET = "src/data/chess/chess_truncated_pgns_with_fen_100k.parquet"
# Maximum fullmove number at which we allow splicing (default: 24)
MAX_FULLMOVE = 24


def create_prompt(pgn: str) -> str:
    """
    Given a PGN string, generate a prompt for the model.
    The prompt instructs the model to analyze the position and output its reasoning and move.
    The PGN string will be in standard format like:
    1. e4 e5 2. Nf3 Nc6 3. Bb5 a6
    """
   
    return f"""You are a chess grandmaster. Please analyze this chess position and provide your reasoning and next move.

    Current game (PGN):
    {pgn}

    Provide your analysis and move in the following format:

    <think>
    Your detailed reasoning, outlining key threats, piece positions, and any plans.
    </think>
    <answer>
    Your chosen move in standard algebraic notation (SAN)
    </answer>\n
     """


def truncate_game(pgn_text: str, max_fullmove: int = MAX_FULLMOVE) -> dict:
    """
    Truncates a game at a random valid point somewhere in the game.

    Args:
        pgn_text (str): The full PGN string of the game.
        max_fullmove (int): The maximum board.fullmove_number allowed for splicing.

    Returns:
        dict: A dictionary with:
          - "truncated_pgn": PGN string up to the splice point.
          - "fen": The FEN string of the corresponding board position.
          - "original_pgn": The original PGN string of the game.
        Returns None if unable to create a valid truncated game.
    """
    try:
        # Parse the PGN from the loaded string
        pgn_io = io.StringIO(pgn_text)
        game = chess.pgn.read_game(pgn_io)
    except Exception as e:
        print(f"Error parsing PGN: {e}")
        return None

    if game is None:
        return None

    # Prepare to iterate over moves, while recording valid splice indices.
    board = game.board()
    valid_indices = []  # list of indices (number of moves played) eligible for splicing
    moves = list(game.mainline_moves()) # each move( white or black)

    # Extract the number before the last period in the PGN string
    # First split by periods and get all but the last element
    parts = pgn_text.split(".")[:-1]
    if not parts:
        return None
    # Get the last number before the final period
    try:
        last_number = int(parts[-1].split()[-1])
   
    except (IndexError, ValueError):
        return None
    
    # Traverse moves and record indices as long as fullmove_number is within the threshold.
    for idx, move in enumerate(moves):
        board.push(move)
        # board.fullmove_number starts at 1 and increments after Black's move.
        # if board.fullmove_number <= max_fullmove:
        #     # We store idx+1, representing the number of moves played,
        #     # so that we always include at least one move.
        #     valid_indices.append(idx + 1)

        current_full_move = (idx // 2) + 1
        if current_full_move <= max_fullmove:
            valid_indices.append(idx + 1)
        else:
            break

    # print("length valid indexes" ,len(valid_indices))
    
    # If no valid splice point was found (e.g. game ended quickly),
    # then skip this game.
    if not valid_indices:
        return None

    # Select a random splice index from valid candidates.
    splice_point = random.choice(valid_indices)
    full_moves = splice_point // 2

    # Verify that we're not exceeding our move limit
    assert full_moves <= max_fullmove, f"Splice point {splice_point} exceeds max moves {max_fullmove}"

    # Rebuild the truncated game from the original game up to the splice point.
    board = game.board()
    truncated_game = chess.pgn.Game()
    truncated_game.headers = game.headers  # optionally keep headers if needed, or leave empty. --> we want to leave headers empty since we don't be including them when we pass it in the prompt
    node = truncated_game

    # Replay moves up to the splice point.
    for move in moves[:splice_point]:
        board.push(move)
        node = node.add_variation(move)
        
    # Use a StringExporter to get a clean PGN string with only moves.
    exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
    truncated_pgn = truncated_game.accept(exporter).strip()
    # Get the FEN of the current board state.
    current_fen = board.fen()

    if not verify_position(truncated_pgn, current_fen):
        print(f"Position verification failed!")
        print(f"Truncated PGN: {truncated_pgn}")
        print(f"Stored FEN: {current_fen}")
        return None

  
    prompt = create_prompt(truncated_pgn) # important to use the truncated pgn so it matches the fen

    return {
        "pgn": truncated_pgn, 
        "fen": current_fen, 
        "number_of_moves": len(moves),
          "splice_point": splice_point, 
          "last_number": last_number, 
          "full_moves": full_moves, 
          "prompt": prompt
          }


    # return {"truncated_pgn": truncated_pgn, "fen": current_fen, "original_pgn": pgn_text, "number_of_moves": len(moves), "splice_point": splice_point, "last_number": last_number, "full_moves": full_moves, "prompt": prompt}

def verify_position(truncated_pgn: str, fen: str) -> bool:
    """
    Verify that the FEN string matches the position reached after playing through the truncated PGN.
    
    Args:
        truncated_pgn (str): The truncated PGN string
        fen (str): The FEN string to verify
        
    Returns:
        bool: True if positions match, False otherwise
    """
    # Create a new game from the truncated PGN
    pgn_io = io.StringIO(truncated_pgn)
    game = chess.pgn.read_game(pgn_io)
    if game is None:
        return False
        
    # Play through the moves to reach the final position
    board = game.board()
    for move in game.mainline_moves():
        board.push(move)
        
    # Compare the resulting position's FEN with our stored FEN
    return board.fen() == fen




def create_truncated_dataset(input_dataset: str = INPUT_DATASET,
                             output_dataset: str = OUTPUT_DATASET,
                             max_fullmove: int = MAX_FULLMOVE) -> None:
    """
    Loads a cleaned PGN dataset from a Parquet file, truncates each game at a random
    valid point in the opening phase, and saves the resulting dataset as a new Parquet file.
    
    Args:
        input_dataset (str): Path to the input cleaned PGN dataset.
        output_dataset (str): Path where the truncated dataset will be saved.
        max_fullmove (int): Maximum board.fullmove_number allowed for splicing.
    """
    # Load the cleaned PGN dataset
    try:
        df = pd.read_parquet(input_dataset)
    except Exception as e:
        print(f"Error loading input dataset from {input_dataset}: {e}")
        return

    if 'pgn' not in df.columns:
        print("Input dataset does not contain a 'pgn' column.")
        return

    truncated_examples = []

    print("Processing games to create truncated positions...")
    # Process each PGN in the dataset
    for idx, row in tqdm(df.iterrows(), total=len(df)):
    # for idx, row in tqdm(df.head(5).iterrows(), total=5): # for testing
        pgn_text = row["pgn"]
        truncated = truncate_game(pgn_text, max_fullmove=max_fullmove)
        if truncated is not None:
            truncated_examples.append(truncated)

    if not truncated_examples:
        print("No valid truncated games were found.")
        return

    # Create a new DataFrame from the truncated examples.
    truncated_df = pd.DataFrame(truncated_examples)

    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_rows', None)       # Optional: if you want to display all rows
    pd.set_option('display.max_columns', None)    # Optional: if you have many columns
    pd.set_option('display.width', 200)           # Adjust the total width as needed, or use None for unlimited

    print(truncated_df.head())
    

    
    # Ensure the output directory exists.
    output_dir = os.path.dirname(output_dataset)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the new dataset as a Parquet file.
    try:
        truncated_df.to_parquet(output_dataset, index=False)
        print(f"Truncated dataset saved to {output_dataset}.")
        print(f"Total examples: {len(truncated_df)}")
    except Exception as e:
        print(f"Error saving output dataset to {output_dataset}: {e}")

def main():
    create_truncated_dataset()

if __name__ == "__main__":
    main()