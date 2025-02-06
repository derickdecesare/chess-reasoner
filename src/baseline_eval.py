import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import chess
import chess.pgn
from stockfish import Stockfish
import json
from datetime import datetime
import re
from utils.memory_utils import print_memory_stats
import os
from utils.formatting_utils import is_plausible_san, polish_pgn
from utils.chessUtils import generate_move


# Setup device for any accelerator (CUDA, MPS, or CPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# def generate_move(model, tokenizer, pgn, board, max_attempts=5):
#     """Generate next move using the model"""
#     base_temperature = 0.6
#     temperature_increment = 0.2

#     # extract fen from board
#     fen = board.fen()
   

#     # Strip PGN headers and just keep moves
#     cleaned_pgn = pgn.split("\n\n")[-1] if "\n\n" in pgn else pgn
#     # remove * ---> this doesn't help either
#     # cleaned_pgn = cleaned_pgn.replace("*", "")

#     is_first_move = cleaned_pgn.strip() in ["*", ""]

#     # this doesn't actually seem to help
#     # # Add partial move numbers for clarity
#     # if not is_first_move:
#     #     # Count existing moves
#     #     num_moves = len(re.findall(r"\d+\.", cleaned_pgn))
#     #     print(f"board turn: {board.turn}")
#     #     if board.turn == chess.WHITE:
#     #         print("its white")
#     #         cleaned_pgn += f" {num_moves+1}."
#     #     else:
#     #         cleaned_pgn += f" {num_moves+1}..."
            
#     print(f"Cleaned PGN: {cleaned_pgn}")
#     # print(f"FEN: {fen}")



#     if is_first_move:
#         print("First move")

    
#     # prompt = f"You are a chess master. Given this game in PGN format, play the next move as best as you can:\n\n{pgn}\n\nNext move:"
#     # Different prompt for first move
#     # if is_first_move:
#     #     prompt = (
#     #         "You are a chess master. Starting a new chess game.\n"
#     #         "You are White. Make a strong opening move.\n"
#     #         "Respond with ONLY the move in standard algebraic notation (SAN).\n"
#     #         "Your move:"
#     #     )
#     # else:
#     #     prompt = (
#     #         "You are a chess master. Given this chess game pgn:\n\n"
#     #         f"Moves so far: {cleaned_pgn}\n\n"
#     #         "Respond with a strong valid move in standard algebraic notation (SAN).\n"
#     #         "Next move:"
#     #     )



#     # Prompt for instruct models
#     # if is_first_move:
#     #     prompt = (
#     #         "You are a chess grandmaster playing as White starting a new chess game. Generate ONLY ONE legal opening move "
#     #         "in Standard Algebraic Notation (SAN). Your response must contain ONLY the move.\n\n"
#     #         "Your opening move:"
#     #     )
#     # else:
#     #     prompt = (
#     #         "You are a chess grandmaster. Current game (PGN):\n"
#     #         f"{cleaned_pgn}\n\n"
#     #         # f"Current FEN: {fen}\n\n"
#     #         "Generate ONLY ONE legal next move in SAN. "
#     #         "Your response must contain ONLY the move.\n\n"
#     #         "Next move:"
#     #     )


#     # Prompt for base models (raw pgn prediction)
#     # remove the *
#     cleaned_pgn_for_base_models = cleaned_pgn.replace("*", "")
#     prompt = cleaned_pgn_for_base_models

#     print(f"Prompt:\n{prompt}\n")
    
#     # print(f"Model is a string: {isinstance(model, str)}")
#     # Only do string checks if model is actually a string
#     if isinstance(model, str):
#         print(f" gpt in model: {'gpt' in model.lower()}")
#         print(f" claude in model: {'claude' in model.lower()}")
#         print(f" o3 in model: {'o3' in model.lower()}")
#         print(f" o1 in model: {'o1' in model.lower()}")
#         # Check if we should use API models
#         if "gpt" in model.lower() or "claude" in model.lower() or "o3" in model.lower() or "o1" in model.lower():
#             try:
#                 from api_models import generate_move_api
#                 for move in generate_move_api(model, prompt):
#                     yield move
#                 return
#             except ImportError as e:
#                 print(f"API model error: {e}")
#                 yield "ILLEGAL_MOVE"
#                 return
    
#     all_moves_found = False
    
#     for attempt in range(max_attempts):
#         temperature = base_temperature + (temperature_increment * attempt)
#         print(f"\nAttempt {attempt + 1}, Temperature: {temperature}")
        
#         # Generate response
#         inputs = tokenizer(prompt, return_tensors="pt")
#         # Move inputs to the same device as the model
#         inputs = {k: v.to(device) for k, v in inputs.items()}
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=7,
#             temperature=temperature,
#             top_k=200
#         )
#         response = tokenizer.decode(
#             outputs[0], 
#             skip_special_tokens=True,
#             spaces_between_special_tokens=True  # Add spaces between tokens for readability and easier parsing
#             )
#         # print(f"Raw response: {response}")
        
#         # Extract move from response
#         new_text = response[len(prompt):].strip() # remove prompt and remove whitespace
#         # print(f"Extracted text: {new_text}")

#         new_text_no_numbers = re.sub(r'\d+\.\s*', '', new_text) # remove move numbers
#         # print(f"After removing numbers: {new_text_no_numbers}")
#         tokens = new_text_no_numbers.split() # split into tokens (split on spaces)
#         print(f"Tokens: {tokens}")

#         # Try only the FIRST plausible token
#         move_count = 0  # Initialize counter
#         for token in tokens:
#             clean_move = token.strip().rstrip('+#') # remove +# from end of move
#             print(f"Trying token: {clean_move}")
#             if len(clean_move) < 2: # if move is too short, skip it
#                 # print(f"Token too short, skipping: {clean_move}")
#                 continue
#             if not is_plausible_san(clean_move):
#                 print(f"Token not plausible, skipping: {clean_move}")
#                 continue

#             # We found a plausible token
#             # print(f"  - Yielding plausible move token: {clean_move}")
#             yield clean_move
#             move_count += 1

#         # If any moves were found during this attempt, we won't keep re-trying in new attempts
#         # this is why it's ending early if we didn't find any valid moves then we want to keep trying and go to attempt 2
#         # if move_count > 0:
#         #     all_moves_found = True
#         #     break
  
#     # If we never yielded any moves in all attempts, yield an illegal move to signal failure
#     if not all_moves_found:
#         print("No valid moves found; yielding illegal move to signal failure...")
#         yield "ILLEGAL_MOVE"

def evaluate_model(
     model_name="Qwen/Qwen2.5-3B-Instruct",
    local_model_path=None,
    results_dir="src/results",
    prompt_type="instruct",
    include_fen=False,
    num_games=10,
    stockfish_elo=1200
):
    # print_memory_stats()
    print(f"Model name: {model_name}")


    # Load model and tokenizer from a local path if provided; otherwise use model_name.
    if local_model_path is not None:
        print(f"Loading fine-tuned model from local path: {local_model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            device_map="auto",
            torch_dtype=torch.float16
        )
        if device.type == "mps":
            model = model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
     
    else:
        # For API models, we just pass the model name string.
        if any(x in model_name.lower() for x in ["gpt", "claude", "o3", "o1"]):
            model = model_name
            tokenizer = None
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",  # This will handle device allocation automatically.
                torch_dtype=torch.float16  # Use half precision to save memory.
            )
            # More device logic: if you're using MPS, move the model accordingly.
            if device.type == "mps":
                model = model.to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_name)


    # # Check if we're using an API-based model (GPT-4 or Claude)
    # if "gpt" in model_name.lower() or "claude" in model_name.lower() or "o3" in model_name.lower() or "o1" in model_name.lower():
    #     # For API models, we just pass the model name string
    #     # No need to load weights or tokenizer
    #     model = model_name
    #     tokenizer = None
    # else:
    #     # Setup for local models (Qwen, Llama, etc.)
    #     model = AutoModelForCausalLM.from_pretrained(
    #         model_name,
    #         device_map="auto",  # This will handle CUDA automatically
    #         torch_dtype=torch.float16  # Use half precision to save memory
    #     )
    #     # Only move to MPS if we're using it
    #     if device.type == "mps":
    #         model = model.to(device)
    #     tokenizer = AutoTokenizer.from_pretrained(model_name)


    stockfish = Stockfish(parameters={"UCI_Elo": stockfish_elo})
    # let's check elo of stockfish
    # print("stockfish params: ", stockfish.get_parameters())

    # Create results directory at the START of the function
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
            # for move_text in generate_move(model, tokenizer, str(game), board):
            for move_text in generate_move(model, tokenizer, str(game), board, max_attempts=5, prompt_type=prompt_type, include_fen=include_fen):
                try:
                    move = board.parse_san(move_text)
                    if move in board.legal_moves:
                        print(f"Valid move ✅: {move_text}")
                        valid_move = move
                        break
                except:
                    print(f"Invalid move ❌: {move_text}")
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

        # if it's the last game then let's take an average of the num_moves
        if game_id == num_games - 1:
            total_moves = sum(result["num_moves"] for result in results)
            average_num_moves = total_moves / num_games
            results[-1]["average_num_moves"] = average_num_moves


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


def simple_test_generation(rel_path = "models/llama-3.2-1b-instruct-finetune_png_10k"):
    """
    Loads a locally fine-tuned model and runs basic generation on a few PGN examples.
    It prints the input PGN and the generated continuation directly to the terminal.
    """
    # Compute the absolute path to the local model directory.
    # __file__ is the path to this current script (src/baseline_eval.py).
    
    local_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", rel_path))
    print(f"Loading model from local path: {local_model_path}")
    
    # Verify that the local model directory exists.
    if not os.path.exists(local_model_path):
        raise FileNotFoundError(f"Local model directory '{local_model_path}' not found.")
    
    # Load the model and tokenizer, forcing local file-only mode.
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        local_files_only=True  # Ensure we are loading from local files only.
    )
    tokenizer = AutoTokenizer.from_pretrained(
        local_model_path,
        local_files_only=True
    )
    
    # Determine the best available device: CUDA > MPS > CPU.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    
    # Define some sample PGN inputs.
    pgn_examples = [
        # "1. e4 e5 2. Nf3 Nc6",
        # "1. d4 Nf6 2. c4 e6 3. Nc3 Bb4",
        # "1. c4", 
        """You are a chess grandmaster. Please analyze this chess position and provide your reasoning and next move.

    Current game (PGN):
    1. d4 Nf6 2. c4 e6 3. Nc3 Bb4

    Provide your analysis and move in the following format:

    <think>
    Your detailed reasoning, outlining key threats, piece positions, and any plans.
    </think>
    <answer>
    Your chosen move in standard algebraic notation (SAN)
    </answer>
    """
    ]


    
    print("\n--- PGN Generation Test ---")
    for idx, pgn in enumerate(pgn_examples, start=1):
        print(f"\n--- PGN Example {idx} ---")
        print("Input PGN:", pgn)
        
        # Tokenize the PGN and ensure tensors are on the model's device.
        inputs = tokenizer(pgn, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate a continuation with sampling.
        outputs = model.generate(
            **inputs,
            max_new_tokens=2000,   # Adjust token count as needed.
            do_sample=True,
            temperature=0.7
        )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Generated continuation:")
        print(generated_text)


def debug_local_model_directory(rel_path = "models/llama-3.2-1b-instruct-finetune_png_10k"):
    """
    Computes the absolute path to the local model directory,
    checks if it exists, and prints its contents.
    """
    # Compute the absolute path relative to this script.
    
    local_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", rel_path))
    print(f"Computed local model path: {local_model_path}")
    
    if not os.path.exists(local_model_path):
        print(f"ERROR: Local model directory does not exist: {local_model_path}")
    else:
        print("Local model directory exists. Listing contents:")
        for item in os.listdir(local_model_path):
            print(" -", item)
    return local_model_path

def test_debug_model_load():
    """
    Attempts to load both the model and the tokenizer from the local directory.
    Intermediate steps and errors are printed so you can diagnose the issue.
       IMPORTANT:
    For proper loading, the local model directory should contain all necessary files.
      Typically, you must have:
      - config.json             (Model configuration file.)
      - model.safetensors       (Model weights in safetensors format; alternatively, model.bin)
      - special_tokens_map.json (Mapping for the tokenizer’s special tokens)
      - tokenizer_config.json   (Tokenizer configuration file)
      - tokenizer.json          (Tokenizer vocab/rules)
      - generation_config.json  (Generation parameters for some models)
      - training_args.bin       (Optional: Training arguments from fine-tuning)
    """
    local_model_path = debug_local_model_directory()
    
    # First, try loading the model.
    try:
        print("\nAttempting to load the model from local files...")
        model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            local_files_only=True  # Force local loading
        )
        print("SUCCESS: Model loaded successfully.")
    except Exception as e:
        print("\nERROR: Model loading failed with the following exception:")
        print(e)
    
    # Next, try loading the tokenizer.
    try:
        print("\nAttempting to load the tokenizer from local files...")
        tokenizer = AutoTokenizer.from_pretrained(
            local_model_path,
            local_files_only=True  # Force local loading
        )
        print("SUCCESS: Tokenizer loaded successfully.")
    except Exception as e:
        print("\nERROR: Tokenizer loading failed with the following exception:")
        print(e)
        print("\nHINT: The local model directory may be missing necessary tokenizer files (e.g., tokenizer_config.json, tokenizer.json, vocab files, merges.txt, or similar).")
        print("       If you fine-tuned only the model, you might need to download the tokenizer from the original model source and then save it locally using tokenizer.save_pretrained(your_local_path).")

if __name__ == "__main__":
    # Print the device being used.
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    # else:
    #     device = torch.device("cpu")
    # print(f"Using device: {device}")
    
    # test_debug_model_load()

    # simple_test_generation("models/llama-3.2-1b-instruct-finetune_png_10k_cot_1k")

    # fine tuned models --> mine
    evaluate_model(
        model_name="llama-3.2-1b-instruct-finetune_png_10k_cot_1k",
        local_model_path="models/llama-3.2-1b-instruct-finetune_png_10k_cot_1k",
        results_dir="src/resultsCOT1800Stockfish",
        prompt_type="cot",
        stockfish_elo=1800
    )

    # Base models
    # QWEN
    # evaluate_model(model_name="Qwen/Qwen2.5-1.5B")
    # evaluate_model(model_name="Qwen/Qwen2.5-3B")
    # evaluate_model(model_name="Qwen/Qwen2.5-7B")
    # LLAMA
    # evaluate_model(model_name="meta-llama/Llama-3.2-1B")
    # evaluate_model(model_name="meta-llama/Llama-3.2-3B")
    # evaluate_model(model_name="meta-llama/Llama-3.1-8B")
 

    # Instruct models
    # QWEN
    # evaluate_model(model_name="Qwen/Qwen2.5-1.5B-Instruct")
    # evaluate_model(model_name="Qwen/Qwen2.5-3B-Instruct")
    # evaluate_model(model_name="Qwen/Qwen2.5-7B-Instruct")
    # # LLAMA
    
    # evaluate_model(model_name="meta-llama/Llama-3.2-1B-Instruct", results_dir="src/resultsInstructWithCOTPROMPT", prompt_type="cot")
    # evaluate_model(model_name="meta-llama/Llama-3.2-3B-Instruct", results_dir="src/resultsInstructWithCOTPROMPT", prompt_type="cot")
    # evaluate_model(model_name="meta-llama/Llama-3.1-8B-Instruct", results_dir="src/resultsInstructWithCOTPROMPT", prompt_type="cot")

    
    # # MISTRAL
    # evaluate_model(model_name="mistralai/Mistral-7B-Instruct-v0.3") # the worst
    

    # SOTA models
    # evaluate_model(model_name="gpt-4o")
    # evaluate_model(model_name="claude-3-5-sonnet-20241022")
   
    # Reasoning models
    # evaluate_model(model_name="o1-preview")




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