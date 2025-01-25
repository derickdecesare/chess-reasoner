# Python-Chess Library Reference

## Board State Checks

- `board.is_game_over()` - Check if game is finished
- `board.is_check()` - Is current position check?
- `board.is_checkmate()` - Is current position checkmate?
- `board.is_stalemate()` - Is current position stalemate?
- `board.is_insufficient_material()` - Not enough pieces to mate?
- `board.can_claim_fifty_moves()` - 50 move rule claimable?
- `board.can_claim_threefold_repetition()` - Position repeated 3 times?

## Move Generation & Validation

- `board.legal_moves` - Iterator of legal moves
- `board.push(move)` - Make a move on the board
- `board.pop()` - Undo last move
- `board.parse_san("e4")` - Convert SAN to move object
- `chess.Move.from_uci("e2e4")` - Convert UCI to move object

## PGN Handling

- `game = chess.pgn.Game()` - Create new game
- `game.headers["Event"] = "Test"` - Set PGN headers
- `board = game.board()` - Get board from game
- `node = game.add_variation(move)` - Add move to game tree
- `str(game)` - Convert entire game to PGN string

## Position Information

- `board.fen()` - Get FEN string of position
- `board.turn` - Whose turn (True=White, False=Black)
- `board.fullmove_number` - Current move number
- `board.result()` - Get game result ("1-0", "0-1", "1/2-1/2", "\*")

## Stockfish Integration

- `stockfish = Stockfish()` - Initialize engine
- `stockfish.set_fen_position(board.fen())` - Set position
- `stockfish.get_best_move()` - Get engine's best move (UCI format)
- `stockfish.get_parameters()` - Get engine settings
- `stockfish.set_skill_level(20)` - Set engine strength

## Common Workflow Example

```
## Initialize
game = chess.pgn.Game()
board = game.board()
node = game

## Game loop
while not board.is_game_over():

    # Get move somehow (engine/player)
    move = get_move()
    # Make move
    board.push(move)
    node = node.add_variation(move)
    # Check position
    if board.is_checkmate():
        break
Get final PGN
pgn_string = str(game)
```

## Get final PGN

pgn_string = str(game)
