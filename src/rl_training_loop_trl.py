"""
Chess RL Training Loop using TRL GRPO

This script implements an RL loop for a chess reasoning model using TRL’s GRPO method.
It samples prompts for chess positions from a dataset, generates responses in a structured format
(with <think> and <answer> tags), and computes rewards based on both correct formatting and
the move’s effect on the position as evaluated by Stockfish.

Requirements:
- A dataset that includes a "prompt" column (asking for cot reasoning about a pgn string) and a "fen" column (to initialize board state for ground truth evaluation) also full_moves if we want to filter by length or do curriculum learning.
- Libraries: torch, transformers, pandas, chess, trl, etc.
- Stockfish path 
"""

"""
The GRPOTrainer is designed to take a preprocessed dataset (with a prompt column) and then internally:
Sample a prompt,
Generate completions (using generation parameters provided via GRPOConfig such as max_prompt_length and max_completion_length),
Compute rewards using your list of reward functions, and
Perform the policy update.
"""

import os
import re
import logging
from typing import Optional, List
from datetime import datetime
import torch
import pandas as pd
from datasets import Dataset
import chess
import chess.engine
import chess.pgn
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, GRPOConfig, GRPOTrainer
from src.utils.get_stockfish_path import get_stockfish_path # may have to modify this in collab.. 


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
                eval_diff = evaluate_move(board, move)
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
            eval_diff = evaluate_move(board, move)
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





# ------------------------------------------
# RL Trainer Class using TRL's GRPO/PPOTrainer
# ------------------------------------------
class ChessRLTrainer:
    def __init__(
        self,
        model_name: str = "llama-3.2-1b-instruct-finetune_png_10k_cot",
        dataset_path: str = "src/data/chess/chess_rl_fen_pgn_prompt_100k.parquet",
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name

        # Setup logging.
        self.logger = logging.getLogger("ChessRL")
        logging.basicConfig(level=logging.INFO)

        # Setup output directory.
        if "llama" in model_name:
            self.output_dir = "models/llama-1B-GRPO"
            self.run_name = f"llama-1B-GRPO-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            self.output_dir = "models/qwen-1.5B-GRPO"
            self.run_name = f"qwen-1.5B-GRPO-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
       
        # Load the chess position dataset.
        # It will have "pgn", "fen", "prompt", "full_moves"
        self.dataset = pd.read_parquet(dataset_path)
        self.logger.info(f"Loaded {len(self.dataset)} positions from dataset.")

        # Convert Pandas DataFrame to HuggingFace Dataset if necessary.
        try:
            if not isinstance(self.dataset, Dataset):
                self.dataset = Dataset.from_pandas(self.dataset)
                self.logger.info("Converted dataset from pandas DataFrame to HuggingFace Dataset.")
        except Exception as e:
            self.logger.error("Failed to convert dataset to HuggingFace Dataset.", exc_info=e)


        # Initialize tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load the model and the reference model (used for KL regularization).
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name, device_map="auto")
        self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name, device_map="auto")

        self.training_args = GRPOConfig(
            output_dir=self.output_dir,
            run_name=self.run_name,
            learning_rate=5e-6,
            adam_beta1 = 0.9,
            adam_beta2 = 0.99,
            weight_decay = 0.1,
            warmup_ratio = 0.1,
            lr_scheduler_type='cosine',
            logging_steps=1,
            bf16=True,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_generations=16,
            max_prompt_length=800, # this is max length based on max length of png
            max_completion_length=2000, # we need this to be longer for chess cot reasoning
            num_train_epochs=1,
            save_steps=100,
            max_grad_norm=0.1,
            # report_to="wandb",
            log_on_each_node=False,

        )

        self.trainer = GRPOTrainer(
                model=self.model,
                processing_class=self.tokenizer,
                reward_funcs=[
                        legal_move_reward_func,
                        good_move_reward_func,
                        checkmate_reward_func,
                        strict_format_reward_func,
                        soft_format_reward_func,
                        xmlcount_reward_func],
                args=self.training_args,
                train_dataset=self.dataset,
            )




    def train(self):
        """
        Run the full RL training loop.
        The GRPOTrainer internally handles the rollout, response generation, reward evaluation,
        and policy update.
        """
        try:
            self.logger.info("Starting training...")
            self.trainer.train()  # This will run the entire training loop.
            self.logger.info("Training complete.")
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by KeyboardInterrupt. Shutting down engine gracefully.")
            raise  # Optionally re-raise exception if you want to exit with error.
        finally:
            engine.quit()  # Ensure that engine is properly shut down.
            


# ------------------------------------------
# Main Execution
# ------------------------------------------
def main():
    trainer = ChessRLTrainer(
        model_name="derickio/llama-3.2-1b-instruct-finetune_png_10k_cot_1k",
        dataset_path="src/data/chess/chess_rl_fen_pgn_prompt_100k.parquet",
    )
    trainer.train()


if __name__ == "__main__":
    main()