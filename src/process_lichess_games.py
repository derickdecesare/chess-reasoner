import chess.pgn
import pandas as pd
from typing import List, Dict
from tqdm import tqdm
import zstandard as zstd
from stockfish import Stockfish
import io

def process_game(game: chess.pgn.Game, stockfish: Stockfish) -> List[Dict]:
    """Process a single game into positions with evaluations"""
    positions = []
    board = chess.Board()  # Start with fresh board
    
    # Get headers we care about
    white_elo = int(game.headers.get("WhiteElo", 0))
    black_elo = int(game.headers.get("BlackElo", 0))
    avg_elo = (white_elo + black_elo) // 2
    
    # Create new game for current position
    current_game = chess.pgn.Game()
    current_game.headers = game.headers
    current_node = current_game
    
    # Iterate through moves
    node = game
    while node.variations:
        node = node.variations[0]
        
        # Make move on our board
        board.push(node.move)
        
        # Create fresh game up to this position
        current_game = chess.pgn.Game()
        current_node = current_game
        
        # Rebuild game up to current position
        for move in board.move_stack:
            current_node = current_node.add_variation(move)
        
        # Get position evaluation
        stockfish.set_fen_position(board.fen())
        eval_dict = stockfish.get_evaluation()
        
        if eval_dict['type'] == 'mate':
            eval_value = 10000 if eval_dict['value'] > 0 else -10000
        else:
            eval_value = eval_dict['value']
        
        # Only store if game isn't over
        if not board.is_game_over():
            positions.append({
                'pgn': str(current_game),  # PGN up to current position only
                'elo': avg_elo,
                'move_number': board.fullmove_number,
                'position_eval': eval_value,
                'phase': 'opening' if board.fullmove_number <= 15 else 
                        'middlegame' if board.fullmove_number <= 40 else 'endgame',
                'game_result': None
            })
    
    return positions

def generate_dataset(
    input_file: str = 'lichess_db_standard_rated_2024-10.pgn.zst',
    num_games: int = 400,
    output_file: str = 'src/data/chess/chess_positions_lichess.parquet'
) -> None:
    """Process Lichess games into training positions"""
    
    # Initialize Stockfish
    stockfish = Stockfish(parameters={
        "Threads": 1,
        "Hash": 16,
    })
    
    all_positions = []
    
    # Open compressed PGN file
    with open(input_file, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            
            # Process games
            for _ in tqdm(range(num_games)):
                game = chess.pgn.read_game(text_stream)
                if game is None:
                    break
                    
                try:
                    positions = process_game(game, stockfish)
                    all_positions.extend(positions)
                except Exception as e:
                    print(f"Error processing game: {e}")
                    continue
    
    # Convert FENs back to PGNs and create DataFrame
    df = pd.DataFrame(all_positions)
    
    # Save dataset
    df.to_parquet(output_file)
    
    # Print statistics
    print(f"\nGenerated {len(df)} positions")
    print("\nPositions per phase:")
    print(df.groupby('phase').size())
    print("\nELO distribution:")
    print(df['elo'].describe())

if __name__ == "__main__":
    generate_dataset()