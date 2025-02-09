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
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer, TrainerCallback
from utils.get_stockfish_path import get_stockfish_path # may have to modify this in collab.. 


engine = chess.engine.SimpleEngine.popen_uci(get_stockfish_path())
analysis_limit = chess.engine.Limit(time=0.1)


# -------------------------------------------------------------------
# Helper functions (for move extraction, legality, evaluation, and xml counting)
# -------------------------------------------------------------------


############### TESTING DATASTRUCUTRE ########################
def experiment_print_completions(completions, **kwargs) -> list[float]:
    print("=== Experiment: Structure of completions ===")
    for idx, comp in enumerate(completions):
        print(f"Completion index {idx}: type: {type(comp)}")
        # If comp is a list, print the type and value of its first element.
        if isinstance(comp, list) and len(comp) > 0:
            print(f"   First element type: {type(comp[0])} - value: {repr(comp[0])}")
        # If comp is a dict, print its keys and value.
        elif isinstance(comp, dict):
            print(f"   Keys: {list(comp.keys())} - value: {comp}")
        else:
            print(f"   Value: {repr(comp)}")
    
    print("\n=== Experiment: Structure of keyword arguments (extra columns) ===")
    for key, value in kwargs.items():
        print(f"Key: '{key}' - type: {type(value)}")
        if isinstance(value, list):
            print(f"   List length: {len(value)}")
            if len(value) > 0:
                # Print a sample of the first few elements.
                sample = value[:3]
                for sample_idx, item in enumerate(sample):
                    print(f"   Sample index {sample_idx}: type: {type(item)} - value: {repr(item)}")
        else:
            print(f"   Value: {repr(value)}")
    
    # Return a dummy reward list for testing (same length as completions)
    return [0.0] * len(completions)


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

def legal_move_reward_func(completions: List[str], fen: List[str], **kwargs) -> List[float]:
    """
    Award +0.5 if the move extracted from the <answer> tag is legal for the corresponding FEN.
    
    Parameters:
      completions: List of responses as strings (with XML-like tags).
      fen: List of FEN strings, one per completion.
      kwargs: Additional dataset columns.
      
    Returns:
      List of rewards (0.5 for legal moves; else 0.0).
    """
    rewards = []
    for idx, response_text in enumerate(completions):
        # Extract the move from the <answer> tag
        move = extract_move_from_response(response_text)
        current_fen = fen[idx]
        
        if move is None or len(move) > 14:
            rewards.append(0.0)
        else:
            legal, _ = is_move_legal(current_fen, move)
            rewards.append(0.5 if legal else 0.0)
            
    return rewards


def good_move_reward_func(completions: List[str], fen: List[str], **kwargs) -> List[float]:
    """
    Evaluate move quality for legal moves.
    
    Award +2.0 if the move's evaluation improvement exceeds a threshold.
    
    Parameters:
      completions: List of responses as strings (with XML-like tags).
      fen: List of FEN strings, one per response.
      engine: Engine instance for move evaluation.
      analysis_limit: Analysis depth/time limit.
      kwargs: Other dataset columns.
      
    Returns:
      List of rewards (2.0 for moves with evaluation improvement above threshold; else 0.0).
    """
    if not fen or engine is None or analysis_limit is None:
        print("Missing requirements for good_move_reward_func (fen, engine, analysis_limit required). --> engine and analysis_limit are GLOBAL VARIABLES")
        return [0.0 for _ in completions]
    
    good_threshold = 0  # Anything above 0 is considered an improvement.
    rewards = []
    for idx, response_text in enumerate(completions):
        move = extract_move_from_response(response_text)
        if move is None or len(move) > 14:
            rewards.append(0.0)
        else:
            current_fen = fen[idx]
            legal, board = is_move_legal(current_fen, move)
            if not legal:
                rewards.append(0.0)
            else:
                eval_diff = evaluate_move(board, move)
                rewards.append(2.0 if (eval_diff is not None and eval_diff > good_threshold) else 0.0)
    return rewards

def checkmate_reward_func(completions: List[str], fen: List[str], **kwargs) -> List[float]:
    """
    Reward moves that lead to checkmate or produce a substantial evaluation improvement.
    
    Award +4.0 for immediate checkmate;
    Award +2.0 if the evaluation difference exceeds a threshold.
    
    Parameters:
      completions: List of responses as strings (with XML-like tags).
      fen: List of FEN strings, one per response.
      engine: Engine instance for move evaluation.
      analysis_limit: Analysis depth/time limit.
      kwargs: Additional dataset columns.
      
    Returns:
      List of rewards (4.0 for checkmate; 2.0 for substantial eval improvement; else 0.0).
    """
    if not fen or engine is None or analysis_limit is None:
        print("Missing requirements for checkmate_reward_func (fen, engine, analysis_limit required). --> engine and analysis_limit are GLOBAL VARIABLES.")
        return [0.0 for _ in completions]
    
    eval_threshold = 300  # Tunable threshold (in centipawn difference).
    rewards = []
    for idx, response_text in enumerate(completions):
        move = extract_move_from_response(response_text)
        if move is None:
            rewards.append(0.0)
            continue
        
        current_fen = fen[idx]
        legal, board = is_move_legal(current_fen, move)
        if not legal:
            rewards.append(0.0)
            continue
        
        try:
            candidate_move = board.parse_san(move)
        except Exception:
            rewards.append(0.0)
            continue
        
        temp_board = board.copy()
        temp_board.push(candidate_move)
        if temp_board.is_checkmate():
            rewards.append(4.0)
        else:
            eval_diff = evaluate_move(board, move)
            rewards.append(2.0 if (eval_diff is not None and eval_diff > eval_threshold) else 0.0)
    return rewards

def strict_format_reward_func(completions: List[str], **kwargs) -> List[float]:
    """
    Award +0.5 if the response strictly follows the XML structure (<think> and <answer>).
    
    Parameters:
      completions: List of responses as strings.
      kwargs: Additional dataset columns.
      
    Returns:
      List of rewards: +0.5 for strict XML format; else 0.0.
    """
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n*$"
    rewards = []
    for response_text in completions:
        match = re.match(pattern, response_text, re.DOTALL)
        rewards.append(0.5 if match else 0.0)
    return rewards

def soft_format_reward_func(completions: List[str], **kwargs) -> List[float]:
    """
    Award +0.5 if the response contains the XML structure (<think> and <answer>) with a looser match.
    
    Parameters:
      completions: List of responses as strings.
      kwargs: Additional dataset columns.
      
    Returns:
      List of rewards: +0.5 if the soft XML pattern is detected; else 0.0.
    """
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    rewards = []
    for response_text in completions:
        match = re.search(pattern, response_text, re.DOTALL)
        rewards.append(0.5 if match else 0.0)
    return rewards

def xmlcount_reward_func(completions: List[str], **kwargs) -> List[float]:
    """
    Count the correct placement of XML tags and award up to +0.5.
    
    Parameters:
      completions: List of responses as strings.
      kwargs: Additional dataset columns.
      
    Returns:
      List of rewards based on the counted XML tags.
    """
    rewards = []
    for response_text in completions:
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
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
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
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # loading model example from will
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.device != "mps" else "auto",
            attn_implementation="flash_attention_2" if self.device != "mps" else None,
            device_map=None,
            trust_remote_code=True
        ).to(self.device)


        # Seemed to reduce memory usage
        self.model.gradient_checkpointing_enable()


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
            bf16=True if self.device != "mps" else False,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_generations=10, # reduced this for memory
            max_prompt_length=702, # this is max length based on dataset analysis
            max_completion_length=1200, # we need this to be longer for chess cot reasoning --> max in cot dataset was 1706 but we can reduce for testing
            num_train_epochs=1,
            save_steps=100,
            max_grad_norm=0.1,
            # report_to="wandb",
            log_on_each_node=False,
            use_mps_device=True if self.device == "mps" else False,
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