"""
This module contains chess-related utility functions.
In particular, we extract the move-generation code from baseline_eval.py,
and add dynamic options for generating instruct vs. base prompts.
"""

import torch
import re
import chess
from utils.formatting_utils import is_plausible_san

def generate_move(model, tokenizer, pgn, board, max_attempts=5, prompt_type="base", include_fen=False, device=None):
    """Generate next move using the model

    Parameters:
      model: either a loaded language model or a string identifier (for API models).
      tokenizer: the associated tokenizer (if applicable).
      pgn (str): the PGN string for the current game.
      board: a chess.Board() object representing the current position.
      max_attempts (int): number of attempts to generate a valid move.
      prompt_type (str): "base" for a raw PGN prompt; "instruct" for a more instructive prompt or "cot" for a COT prompt that will extract next move from the reasoning response.
      include_fen (bool): If True and prompt_type=="instruct", include the current FEN in the prompt.
      device: torch device to run the generation; if None, it is determined automatically.
      
    This function yields moves extracted from the model’s output.
    """

    # Setup device if not passed in.
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    print(f"Using device: {device}")

    base_temperature = 0.4
    temperature_increment = 0.2

    # extract fen from board
    fen = board.fen()
   
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
    # print(f"FEN: {fen}")

    if is_first_move:
        print("First move")

    # The following commented-out versions of the prompt have been left in for context:
    #
    # prompt = f"You are a chess master. Given this game in PGN format, play the next move as best as you can:\n\n{pgn}\n\nNext move:"
    #
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
    #
    # Prompt for instruct models (one variant)
    # if is_first_move:
    #     prompt = (
    #         "You are a chess grandmaster playing as White starting a new chess game. Generate ONLY ONE legal opening move "
    #         "in Standard Algebraic Notation (SAN). Your response must contain ONLY the move.\n\n"
    #         "Your opening move:"
    #     )
    # else:
    #     prompt = (
    #         "You are a chess grandmaster. Current game (PGN):\n"
    #         f"{cleaned_pgn}\n\n"
    #         # f"Current FEN: {fen}\n\n"
    #         "Generate ONLY ONE legal next move in SAN. "
    #         "Your response must contain ONLY the move.\n\n"
    #         "Next move:"
    #     )

    # Selecting prompt dynamically based on prompt_type and include_fen.
    if prompt_type.lower() == "instruct":
        if is_first_move:
            prompt = (
                "You are a chess grandmaster playing as White starting a new chess game. Generate ONLY ONE legal opening move "
                "in Standard Algebraic Notation (SAN). Your response must contain ONLY the move.\n\n"
                "Your opening move:"
            )
        else:
            if include_fen:
                prompt = (
                    "You are a chess grandmaster. Current game (PGN):\n"
                    f"{cleaned_pgn}\n\n"
                    f"Current FEN: {fen}\n\n"
                    "Generate ONLY ONE legal next move in SAN. "
                    "Your response must contain ONLY the move.\n\n"
                    "Next move:"
                )
            else:
                prompt = (
                    "You are a chess grandmaster. Current game (PGN):\n"
                    f"{cleaned_pgn}\n\n"
                    "Generate ONLY ONE legal next move in SAN. "
                    "Your response must contain ONLY the move.\n\n"
                    "Next move:"
                )
    elif prompt_type.lower() == "cot":
        # Prompt for COT models (extract next move from reasoning response)
        prompt = f"""You are a chess grandmaster. Please analyze this chess position and provide your reasoning and next move.

    Current game (PGN):
    {cleaned_pgn}

    Provide your analysis and move in the following format:

    <think>
    Your detailed reasoning, outlining key threats, piece positions, and any plans.
    </think>
    <answer>
    Your chosen move in standard algebraic notation (SAN)
    </answer>\n
    """
    elif prompt_type.lower() == "base":
        # Prompt for base models (raw pgn prediction)
        # remove the *
        cleaned_pgn_for_base_models = cleaned_pgn.replace("*", "")
        prompt = cleaned_pgn_for_base_models

    print(f"Prompt:\n{prompt}\n")
    
    # print(f"Model is a string: {isinstance(model, str)}")
    # Only do string checks if model is actually a string
    if isinstance(model, str):
        print(f" gpt in model: {'gpt' in model.lower()}")
        print(f" claude in model: {'claude' in model.lower()}")
        print(f" o3 in model: {'o3' in model.lower()}")
        print(f" o1 in model: {'o1' in model.lower()}")
        # Check if we should use API models
        if "gpt" in model.lower() or "claude" in model.lower() or "o3" in model.lower() or "o1" in model.lower():
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
            max_new_tokens=2000 if prompt_type.lower() == "cot" else 7,
            temperature=temperature,
            top_k=200,
        )
        response = tokenizer.decode(
            outputs[0], 
            skip_special_tokens=True,
            spaces_between_special_tokens=True  # Add spaces between tokens for readability and easier parsing
        )
        # print(f"Raw response: {response}")
        
        if prompt_type.lower() == "cot":
            response_text = response[len(prompt):].strip()  # remove prompt and remove whitespace
            print(f"Response from model:\n\n {response_text}")
            move_match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL)
            if move_match:
                move_text = move_match.group(1).strip()
                if len(move_text) > 10:
                    print("bad formatting, skipping")
                    continue
                print(f"Move text extracted from COT response: {move_text}")
                tokens = [move_text]
            else:
                print("No move found in COT response")
                continue

        else :
            # Extract move from response
            new_text = response[len(prompt):].strip()  # remove prompt and remove whitespace
            # print(f"Extracted text: {new_text}")

            new_text_no_numbers = re.sub(r'\d+\.\s*', '', new_text)  # remove move numbers
            # print(f"After removing numbers: {new_text_no_numbers}")
            tokens = new_text_no_numbers.split()  # split into tokens (split on spaces)
            
        # print(f"Tokens: {tokens}")
        # Try only the FIRST plausible token
        move_count = 0  # Initialize counter
        for token in tokens:
            clean_move = token.strip().rstrip('+#')  # remove +# from end of move
            print(f"Trying token: {clean_move}")
            if len(clean_move) < 2:  # if move is too short, skip it
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