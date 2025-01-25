import chess
import chess.pgn
from stockfish import Stockfish
import io

# Create a new game
game = chess.pgn.Game()
game.headers["Event"] = "Test Game"
game.headers["White"] = "Stockfish"
game.headers["Black"] = "Stockfish"

# Get the board from the game
board = game.board()
node = game  # Track current position in game tree

# Initialize Stockfish
stockfish = Stockfish()

print("Initial board state:")
print(board)

# Let's verify what methods we actually have available
print("\nAvailable board methods:")
print([method for method in dir(board) if not method.startswith('_')])

# verify what's in the png module
print("\nchess.pgn module contents:")
print([item for item in dir(chess.pgn) if not item.startswith('_')])

print("\nLet's play 3 moves and track everything:")
for i in range(3):
    print(f"\n--- Move {i+1} ---")
    
    # Get current game state as PGN
    print("Current PGN:")
    print(str(game))
    
    # Get Stockfish move
    stockfish.set_fen_position(board.fen())
    move_uci = stockfish.get_best_move()
    print(f"Stockfish suggests move: {move_uci}")
    
    # Convert UCI move and make it
    move = chess.Move.from_uci(move_uci)
    board.push(move)
    node = node.add_variation(move)  # Add to game tree
    
    print("\nPosition Status:")
    print(f"Check: {board.is_check()}")
    print(f"Checkmate: {board.is_checkmate()}")
    print(f"Stalemate: {board.is_stalemate()}")
    print(f"Insufficient material: {board.is_insufficient_material()}")
    print(f"Can claim fifty moves: {board.can_claim_fifty_moves()}")
    print(f"Can claim threefold repetition: {board.can_claim_threefold_repetition()}")
    
    # Print final PGN after move
    print("\nUpdated PGN:")
    print(str(game))

print("\nFinal game PGN:")
print(str(game))
