import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import chess
import chess.pgn
from stockfish import Stockfish
import json
from datetime import datetime
import re
from memory_utils import print_memory_stats
import os
from formatting_utils import is_plausible_san, polish_pgn


# Setup device for any accelerator (CUDA, MPS, or CPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

def generate_move(model, tokenizer, pgn, board, max_attempts=5):
    """Generate next move using the model"""
    base_temperature = 0.6
    temperature_increment = 0.2

   

    # Strip PGN headers and just keep moves
    cleaned_pgn = pgn.split("\n\n")[-1] if "\n\n" in pgn else pgn
    # remove * ---> this doesn't help either
    # cleaned_pgn = cleaned_pgn.replace("*", "")

    is_first_move = cleaned_pgn.strip() in ["*", ""]

    # this doesn't actually seem to help
    # # Add partial move numbers for clarity
    # if not is_first_move:
    #     # Count existing moves
    #     num_moves = len(re.findall(r"\d+\.", cleaned_pgn))
    #     print(f"board turn: {board.turn}")
    #     if board.turn == chess.WHITE:
    #         print("its white")
    #         cleaned_pgn += f" {num_moves+1}."
    #     else:
    #         cleaned_pgn += f" {num_moves+1}..."
            
    print(f"Cleaned PGN: {cleaned_pgn}")



    if is_first_move:
        print("First move")

    
    # prompt = f"You are a chess master. Given this game in PGN format, play the next move as best as you can:\n\n{pgn}\n\nNext move:"
    # Different prompt for first move
    # if is_first_move:
    #     prompt = (
    #         "You are a chess master. Starting a new chess game.\n"
    #         "You are White. Make a strong opening move.\n"
    #         "Respond with ONLY the move in standard algebraic notation (SAN).\n"
    #         "Your move:"
    #     )
    # else:
    #     prompt = (
    #         "You are a chess master. Given this chess game pgn:\n\n"
    #         f"Moves so far: {cleaned_pgn}\n\n"
    #         "Respond with a strong valid move in standard algebraic notation (SAN).\n"
    #         "Next move:"
    #     )

    if is_first_move:
        prompt = (
            "You are a chess grandmaster playing as White starting a new chess game. Generate ONLY ONE legal opening move "
            "in Standard Algebraic Notation (SAN). Your response must contain ONLY the move.\n\n"
            "Your opening move:"
        )
    else:
        prompt = (
            "You are a chess grandmaster. Current game (PGN):\n"
            f"{cleaned_pgn}\n\n"
            "Generate ONLY ONE legal next move in SAN. "
            "Your response must contain ONLY the move.\n\n"
            "Next move:"
        )

    print(f"Prompt:\n{prompt}\n")
    
    print(f"Model is a string: {isinstance(model, str)}")
    # Only do string checks if model is actually a string
    if isinstance(model, str):
        print(f" gpt in model: {'gpt' in model.lower()}")
        print(f" claude in model: {'claude' in model.lower()}")
        # Check if we should use API models
        if "gpt" in model.lower() or "claude" in model.lower():
            try:
                from api_models import generate_move_api
                for move in generate_move_api(model, prompt):
                    yield move
                return
            except ImportError as e:
                print(f"API model error: {e}")
                yield "ILLEGAL_MOVE"
                return
    
    all_moves_found = False
    
    for attempt in range(max_attempts):
        temperature = base_temperature + (temperature_increment * attempt)
        print(f"\nAttempt {attempt + 1}, Temperature: {temperature}")
        
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt")
        # Move inputs to the same device as the model
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model.generate(
            **inputs,
            max_new_tokens=7,
            temperature=temperature,
            top_k=200
        )
        response = tokenizer.decode(
            outputs[0], 
            skip_special_tokens=True,
            spaces_between_special_tokens=True  # Add spaces between tokens for readability and easier parsing
            )
        # print(f"Raw response: {response}")
        
        # Extract move from response
        new_text = response[len(prompt):].strip() # remove prompt and remove whitespace
        # print(f"Extracted text: {new_text}")

        new_text_no_numbers = re.sub(r'\d+\.\s*', '', new_text) # remove move numbers
        # print(f"After removing numbers: {new_text_no_numbers}")
        tokens = new_text_no_numbers.split() # split into tokens (split on spaces)
        print(f"Tokens: {tokens}")

        # Try only the FIRST plausible token
        move_count = 0  # Initialize counter
        for token in tokens:
            clean_move = token.strip().rstrip('+#') # remove +# from end of move
            print(f"Trying token: {clean_move}")
            if len(clean_move) < 2: # if move is too short, skip it
                # print(f"Token too short, skipping: {clean_move}")
                continue
            if not is_plausible_san(clean_move):
                print(f"Token not plausible, skipping: {clean_move}")
                continue

            # We found a plausible token
            # print(f"  - Yielding plausible move token: {clean_move}")
            yield clean_move
            move_count += 1

        # If any moves were found during this attempt, we won't keep re-trying in new attempts
        # this is why it's ending early if we didn't find any valid moves then we want to keep trying and go to attempt 2
        # if move_count > 0:
        #     all_moves_found = True
        #     break
  
    # If we never yielded any moves in all attempts, yield an illegal move to signal failure
    if not all_moves_found:
        print("No valid moves found; yielding illegal move to signal failure...")
        yield "ILLEGAL_MOVE"

def evaluate_model(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    num_games=3,
    stockfish_elo=1200,
    output_file="results.json"
):
    # print_memory_stats()
    print(f"Model name: {model_name}")
    # Check if we're using an API-based model (GPT-4 or Claude)
    if "gpt" in model_name.lower() or "claude" in model_name.lower():
        # For API models, we just pass the model name string
        # No need to load weights or tokenizer
        model = model_name
        tokenizer = None
    else:
        # Setup for local models (Qwen, Llama, etc.)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",  # This will handle CUDA automatically
            torch_dtype=torch.float16  # Use half precision to save memory
        )
        # Only move to MPS if we're using it
        if device.type == "mps":
            model = model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    stockfish = Stockfish(parameters={"UCI_Elo": stockfish_elo})
    # let's check elo of stockfish
    # print("stockfish params: ", stockfish.get_parameters())

    # Create results directory at the START of the function
    results_dir = "src/results"  # Path relative to container's /workspace directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Sanitize model name for filename by replacing / with _
    safe_model_name = model_name.replace('/', '_')
    output_path = os.path.join(results_dir, f"results-{safe_model_name}-{stockfish_elo}.json")

    results = []
    
    for game_id in range(num_games):
        print(f"Starting game {game_id + 1} of {num_games}")
        
        # Initialize game and board
        game = chess.pgn.Game() # Create game (root node)
        game.headers["Event"] = f"Test Game {game_id}"
        game.headers["White"] = f"Stockfish-{stockfish_elo}"
        game.headers["Black"] = model_name # model plays better as black cause format is a little clearer
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        game.headers["Round"] = f"{game_id + 1}"
        game.headers["Site"] = "Docker Arena"
        
        board = game.board() # Board tracks actual position
        node = game # Node tracks position in game tree for PGN # reference to same game object

     
    
        # Play game
        while not board.is_game_over(): # this function is indeed real
            # stockfish goes first
            stockfish.set_fen_position(board.fen())
            best_move = stockfish.get_best_move()
            move = chess.Move.from_uci(best_move)
            board.push(move)
            node = node.add_variation(move)
            # Model's turn
            valid_move = None
            for move_text in generate_move(model, tokenizer, str(game), board):
                try:
                    move = board.parse_san(move_text)
                    if move in board.legal_moves:
                        valid_move = move
                        break
                except:
                    continue
            
            if not valid_move: 
                # Model forfeits - couldn't make legal move
                game.headers["Result"] = "0-1"
                game.headers["Termination"] = "Model forfeit"
                break
                
            # Make model's move
            board.push(valid_move)
            node = node.add_variation(valid_move)
            # 1. Adds move to game tree
            # 2. Returns new node at that position
            # 3. Original game object maintains entire tree
     
            if board.is_game_over(): 
                break
                
        
            # # Stockfish's turn (if model was white)
            # stockfish.set_fen_position(board.fen())
            # best_move = stockfish.get_best_move()
            # move = chess.Move.from_uci(best_move)
            # board.push(move)
            # node = node.add_variation(move)
            
            
        # If game ended normally, set the result
        if "Result" not in game.headers:
            if board.is_checkmate():
                # Set explicit result based on who delivered checkmate
                game.headers["Result"] = "0-1" if board.turn == chess.WHITE else "1-0"
                game.headers["Termination"] = "Checkmate"
            else:
                game.headers["Result"] = board.result()


        # Record results
        results.append({
            "game_id": game_id,
            "model": model_name,
            "stockfish_elo": stockfish_elo,
            "result": game.headers["Result"],
            "termination": game.headers.get("Termination", "Normal"),
            "pgn": str(game),
            "num_moves": board.fullmove_number
        })


        print(f"Results: {results}")

        # Save after each game (using the path we set up at the start)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)



def test_move_generation():
    """Test move generation with various board positions"""
    print("\n=== Starting Move Generation Tests ===")
    # print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f}MB")
    else:
        print("Running on CPU only")

    # Print memory stats before loading model
    # print_memory_stats()
    # Setup
    
    # Initialize model
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Test positions
    test_positions = [
        {
            "name": "Opening position (no moves)",
            "moves": []
        },
        {
            "name": "After 1.e4",
            "moves": ["e4"]
        },
        {
            "name": "After 1.e4 e5 2.Nf3",
            "moves": ["e4", "e5", "Nf3"]
        }
    ]
    
    for test_case in test_positions:
        print(f"\n\n--- Testing: {test_case['name']} ---")
        
        # Create a PGN game with the specified moves
        game = chess.pgn.Game()
        board = game.board()
        node = game
        
        # Apply moves
        for move_san in test_case["moves"]:
            move = board.parse_san(move_san)
            board.push(move)
            node = node.add_variation(move)
        
        # Show current board and PGN
        print(f"Current board:\n{board}")
        print("PGN so far:")
        print(str(game))
        
        # Collect valid moves
        valid_moves_found = 0
        
        # Generate from model
        for move_text in generate_move(model, tokenizer, str(game)):
            print(f"Trying to parse generated move: '{move_text}'")
            try:
                move_obj = board.parse_san(move_text)
                if move_obj in board.legal_moves:
                    print(f"  ✓ Valid + legal move: {move_text}")
                    valid_moves_found += 1
                else:
                    print(f"  ✗ Move parsed but not legal: {move_text}")
            except Exception as e:
                print(f"  ✗ Invalid SAN format: {move_text} ({str(e)})")
        
        print(f"\nResults for '{test_case['name']}': {valid_moves_found} valid moves found.")


if __name__ == "__main__":
    # evaluate_model(model_name="Qwen/Qwen2.5-3B-Instruct")
    
    # evaluate_model(model_name="meta-llama/Llama-3.2-3B-Instruct")
    # evaluate_model(model_name="mistralai/Mistral-7B-Instruct-v0.3") # the worst


    # evaluate_model(model_name="Qwen/Qwen2.5-7B-Instruct")
    # evaluate_model(model_name="gpt-4o")
    # evaluate_model(model_name="claude-3-5-sonnet-20241022")
    evaluate_model(model_name="meta-llama/Llama-3.1-8B-Instruct")

    # Qwen/Qwen2.5-14B-Instruct

    # test_move_generation()





# def test_setup():
#     """Verify everything loads correctly"""
#     try:
#         # Test tiny model first
#         model_name = "Qwen/Qwen1.5-0.5B-Chat"  # Small for testing
#         model = AutoModelForCausalLM.from_pretrained(model_name)
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#         print("✓ Model loaded")
        
#         # Test chess setup
#         board = chess.Board()
#         stockfish = Stockfish()
#         print("✓ Chess environment ready")
        
#         return True
#     except Exception as e:
#         print(f"Setup failed: {e}")
#         return False

# if __name__ == "__main__":
#     test_setup()