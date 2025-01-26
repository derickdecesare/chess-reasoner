import chess
import chess.pgn
from stockfish import Stockfish
import pandas as pd
from typing import List, Dict
from tqdm import tqdm
from datetime import datetime
import os

def create_game_positions(elo_level: int = 2000) -> List[Dict]:
    """Play a full game and collect positions
    Args:
        elo_level: ELO rating for Stockfish (1200-3000)
    Returns:
        List of positions with metadata
    """
    positions = []
    
    # Initialize game and engine with direct ELO setting
    game = chess.pgn.Game()
    game.headers["Event"] = "Training Game"
    game.headers["White"] = f"Stockfish {elo_level}"
    game.headers["Black"] = f"Stockfish {elo_level}"
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    stockfish = Stockfish(parameters={
       "UCI_LimitStrength": True,
        "UCI_Elo": elo_level,
        "Threads": 1,
        "Move Overhead": 10,
    })
    board = game.board()
    node = game
    
    # Play until game is over
    while not board.is_game_over():
        # Store current position before making move
        stockfish.set_fen_position(board.fen())
        current_eval = stockfish.get_evaluation()
        
        # Handle mate scores
        if current_eval['type'] == 'mate':
            eval_value = 100.0 if current_eval['value'] > 0 else -100.0  # Use large value for mate
        else:
            eval_value = current_eval['value'] / 100  # Convert centipawns to pawns

        # Only store position if we have some moves in the game
        if board.fullmove_number > 1:  # or whatever minimum number of moves you want
            positions.append({
                'pgn': str(game),
                'elo': elo_level,
                'move_number': board.fullmove_number,
                'position_eval': eval_value,
                'phase': 'opening' if board.fullmove_number <= 15 else 
                        'middlegame' if board.fullmove_number <= 40 else 'endgame',
                'game_result': board.result() if board.is_game_over() else None
            })
        
        # positions.append({
        #     'pgn': str(game),
        #     'elo': elo_level,
        #     'move_number': board.fullmove_number,
        #     'position_eval': current_eval['value'] / 100,  # Convert centipawns to pawns
        #     'phase': 'opening' if board.fullmove_number <= 15 else 
        #             'middlegame' if board.fullmove_number <= 40 else 'endgame',
        #     'game_result': board.result() if board.is_game_over() else None
        # })
        
        # Get and make move
        move = chess.Move.from_uci(stockfish.get_best_move())
        board.push(move)
        node = node.add_variation(move)
    
    return positions

def generate_dataset(
    num_games: int = 20,
    elo_levels: List[int] = [1320 ,1500, 2000, 2500, 3000],  # More reasonable ELO spread

    output_file: str = 'src/data/chess/chess_positions.parquet',
) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    """Generate full dataset with multiple games at different levels"""
    all_positions = []
    
    # Generate games
    for elo in tqdm(elo_levels, desc="ELO levels"):
        for _ in tqdm(range(num_games // len(elo_levels)), desc=f"Games at ELO {elo}"):
            try:
                positions = create_game_positions(elo)
                all_positions.extend(positions)
            except Exception as e:
                print(f"Error generating game at ELO {elo}: {e}")
                continue
    
    # Convert to DataFrame and save
    df = pd.DataFrame(all_positions)
    df.to_parquet(output_file)
    
    # Print statistics
    print(f"\nGenerated {len(df)} positions")
    print("\nPositions per ELO/phase:")
    print(df.groupby(['elo', 'phase']).size().unstack(fill_value=0))
    
    # Print average eval by phase
    print("\nAverage evaluation by phase:")
    print(df.groupby('phase')['position_eval'].mean())

if __name__ == "__main__":
    generate_dataset()
