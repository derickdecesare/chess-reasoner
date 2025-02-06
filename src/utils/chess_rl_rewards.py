import re
import torch
import chess
import chess.engine
from typing import Optional, List
from src.utils.get_stockfish_path import get_stockfish_path

engine = chess.engine.SimpleEngine.popen_uci(get_stockfish_path())
analysis_limit = chess.engine.Limit(time=0.1)


# REWARD FUNCTIONS --> imported from utils/chess_rl_rewards.py
# legal_move_reward_func:
# Checks if the extracted move is legal (given the FEN).
# Reward: +0.5 if legal; 0 otherwise.

# good_move_reward_func:
# Evaluates the move’s improvement using engine analysis (only for legal moves).
# Reward: +2.0 when the move significantly improves the position outcome.

# checkmate_reward_func:
# Detects if the move results in checkmate or a decisive advantage (based on evaluation difference).
# Reward: +3.0 or +4.0 as a bonus when the move leads to checkmate or a near-decisive advantage.


# strict_format_reward_func:
# Ensures the response exactly follows the strict XML format with <think> and <answer> tags.
# Reward: +0.5 if the strict format is matched; 0 otherwise.


# soft_format_reward_func:
# Checks for the presence of the XML structure in a looser way.
# Reward: +0.5 if the soft pattern is detected; 0 otherwise.


# xmlcount_reward_func:
# Counts the correctly placed XML tags and applies small rewards (and minor penalties for extraneous content).
# Reward: Up to +0.5 based on the quality and accuracy of the XML formatting.
# This set of functions will be computed for each response, and the GRPOTrainer will sum their outputs to produce the total reward signal for training.


# -------------------------------------------------------------------
# Helper functions (for move extraction, legality, and evaluation)
# -------------------------------------------------------------------

def extract_move_from_response(response_text: str) -> Optional[str]:
    """
    Extracts the move from <answer>...</answer> tags in the response.
    Removes extraneous characters (e.g. '+' or '#') often attached to check/mate.
    """
    move_match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL)
    if move_match:
        move_text = move_match.group(1).strip()
        move_text = move_text.replace("+", "").replace("#", "")
        return move_text
    else:
        print("No <answer> tag found; returning None.")
        return None

def is_move_legal(fen: str, move_text: str) -> (bool, chess.Board):
    """
    Checks whether move_text is legal from the board represented by the given FEN.
    Returns a tuple: (is_legal, board_state).
    """
    board = chess.Board(fen)
    try:
        candidate_move = board.parse_san(move_text)
    except Exception as e:
        print(f"Failed to parse move '{move_text}': {e}")
        return (False, board)
    if candidate_move in board.legal_moves:
        print(f"✓ Valid & legal move: {move_text}")
        return (True, board)
    else:
        print(f"✗ Move parsed but not legal: {move_text}")
        return (False, board)

def evaluate_move(board: chess.Board, move_text: str) -> Optional[float]:
    """
    Evaluates the move using the engine. Returns the evaluation difference (new_eval - init_eval).
    If any error happens during evaluation, returns None.
    """
    try:
        init_result = engine.analyse(board, analysis_limit)
        init_eval = init_result["score"].relative.score(mate_score=10000)
    except Exception as e:
        print(f"Error during initial evaluation: {e}")
        return None
    try:
        candidate_move = board.parse_san(move_text)
    except Exception as e:
        print(f"Error parsing candidate move for evaluation: {e}")
        return None

    board.push(candidate_move)
    try:
        new_result = engine.analyse(board, analysis_limit)
        new_eval = new_result["score"].relative.score(mate_score=10000)
    except Exception as e:
        print(f"Error during evaluation after move: {e}")
        board.pop()  # revert move in case of error
        return None
    board.pop()  # revert the move so that board remains unchanged

    eval_diff = new_eval - init_eval
    print(f"Eval before move: {init_eval}, after move: {new_eval}, diff: {eval_diff}")
    return eval_diff

# -------------------------------------------------------------------
# Move Reward Functions
# -------------------------------------------------------------------

def legal_move_reward_func(completions, **kwargs) -> List[float]:
    """
    Checks whether the move (extracted from <answer> tags) is legal for a given FEN.
    Expects 'fen' to be passed in kwargs.
    Reward: +0.5 if legal; 0.0 otherwise.
    """
    fen = kwargs.get("fen", None)
    if fen is None:
        print("No FEN provided for legal_move_reward_func.")
        return [0.0 for _ in completions]
    
    rewards = []
    for comp in completions:
        response_text = comp[0]["content"]
        move = extract_move_from_response(response_text)
        if move is None or len(move) > 14:
            rewards.append(0.0)
        else:
            legal, _ = is_move_legal(fen, move)
            rewards.append(0.5 if legal else 0.0)
    return rewards

def good_move_reward_func(completions, **kwargs) -> List[float]:
    """
    For legal moves (checked via is_move_legal), evaluates the move improvement.
    Expects 'fen', 'engine', and 'analysis_limit' in kwargs.
    Reward: +2.0 if the move's evaluation improvement exceeds a threshold.
    """
    fen = kwargs.get("fen", None)
    if fen is None or engine is None or analysis_limit is None:
        print("Missing requirements for good_move_reward_func (fen, engine, analysis_limit required).")
        return [0.0 for _ in completions]
    
    # Set a threshold where we consider the move substantially improving the position.
    good_threshold = 0  # This threshold is tunable. --> anything above 0 is good.
    rewards = []
    for comp in completions:
        response_text = comp[0]["content"]
        move = extract_move_from_response(response_text)
        if move is None or len(move) > 14: # if it's over 14 we don't waste our time.. it's not a legal move.
            rewards.append(0.0)
        else:
            legal, board = is_move_legal(fen, move)
            if not legal:
                rewards.append(0.0)
            else:
                eval_diff = evaluate_move(board, move, engine, analysis_limit)
                if eval_diff is None:
                    rewards.append(0.0)
                else:
                    rewards.append(2.0 if eval_diff > good_threshold else 0.0)
    return rewards

def checkmate_reward_func(completions, **kwargs) -> List[float]:
    """
    Checks whether the move leads to a checkmate or near-decisive advantage.
    Expects 'fen', 'engine', and 'analysis_limit' in kwargs.
    Reward: +4.0 if the move results in immediate checkmate;
            +2.0 if the eval difference exceeds a higher threshold meaning it was a really good move.
    """
    fen = kwargs.get("fen", None)
    if fen is None or engine is None or analysis_limit is None:
        print("Missing requirements for checkmate_reward_func (fen, engine, analysis_limit required).")
        return [0.0 for _ in completions]
    
    eval_threshold = 300  # Tunable threshold for near-decisive advantage. --> 300 centipawn eval difference is a good move.
    rewards = []
    for comp in completions:
        response_text = comp[0]["content"]
        move = extract_move_from_response(response_text)
        if move is None:
            rewards.append(0.0)
            continue
        legal, board = is_move_legal(fen, move)
        if not legal:
            rewards.append(0.0)
            continue
        try:
            candidate_move = board.parse_san(move)
        except Exception as e:
            rewards.append(0.0)
            continue
        
        # Use a copy of the board for checkmate detection
        temp_board = board.copy()
        temp_board.push(candidate_move)
        if temp_board.is_checkmate():
            rewards.append(4.0)
        else:
            # Evaluate the move using the original board state (which is unmodified)
            eval_diff = evaluate_move(board, move, engine, analysis_limit)
            rewards.append(2.0 if (eval_diff is not None and eval_diff > eval_threshold) else 0.0)
    return rewards

# -------------------------------------------------------------------
# Format Reward Functions
# -------------------------------------------------------------------

def strict_format_reward_func(completions, **kwargs) -> List[float]:
    """
    Checks if the response exactly follows the strict XML structure with <think> and <answer> tags.
    Reward: +0.5 if the strict format is fully matched; 0 otherwise.
    """
    # The strict pattern anchors the entire string.
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n*$"
    rewards = []
    for comp in completions:
        response_text = comp[0]["content"]
        match = re.match(pattern, response_text, re.DOTALL)
        rewards.append(0.5 if match else 0.0)
    return rewards

def soft_format_reward_func(completions, **kwargs) -> List[float]:
    """
    Checks for the presence of the XML structure (looser match) for <think> and <answer> tags.
    Reward: +0.5 if the soft pattern is detected; 0 otherwise.
    """
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    rewards = []
    for comp in completions:
        response_text = comp[0]["content"]
        match = re.search(pattern, response_text, re.DOTALL)
        rewards.append(0.5 if match else 0.0)
    return rewards

def count_xml(text: str) -> float:
    """
    Counts the correctly placed XML tags (<think> and <answer>) in the text.
    Awards 0.125 per correctly formatted tag, with small penalties for extraneous characters after closing tags.
    """
    count = 0.0
    if text.count("<think>\n") == 1:
        count += 0.125
    if text.count("\n</think>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        # Penalize extra characters after </answer>
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> List[float]:
    """
    Applies count_xml to each completion’s text.
    Reward: Up to +0.5 based on how well the XML (<think> and <answer>) is formatted.
    """
    rewards = []
    for comp in completions:
        response_text = comp[0]["content"]
        rewards.append(count_xml(response_text))
    return rewards