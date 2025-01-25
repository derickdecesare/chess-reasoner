import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import chess
import chess.engine
import chess.pgn  # Added for PGN handling
from dataclasses import dataclass
from typing import List, Optional
import re
from utils import get_stockfish_path
import io

@dataclass
class ChessExample:
    # Changed to store PGN instead of FEN since we're working with game history
    pgn: str  
    response: str
    reward: float
    advantage: Optional[float] = None

class ChessRLTrainer:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
    ):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # For local training, float16 is generally fine even on CPU for inference
        # The performance benefit outweighs potential minor precision loss
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32 if device == "cpu" else torch.float16,
            device_map="auto"
        )
        
        # Initialize Stockfish with a moderate ELO for training
        self.engine = chess.engine.SimpleEngine.popen_uci(get_stockfish_path())
        # Set time limit for analysis to balance speed vs accuracy
        self.analysis_limit = chess.engine.Limit(time=0.1)
        
    def generate_response(self, pgn: str, temperature: float = 0.4) -> str:
        """Generate a response for a given game position in PGN format"""
        prompt = self._create_prompt(pgn)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Using similar generation parameters to baseline_eval for consistency
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=200,  # Increased to allow for thinking process
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        return self.tokenizer.decode(outputs[0])

    def compute_reward(self, pgn: str, response: str) -> float:
        """
        Compute reward for a response given the game state
        Returns a float between -1 and 1
        """
        # Parse the PGN to get current position
        game = chess.pgn.read_game(io.StringIO(pgn))
        if not game:
            return 0.0
        
        # Get the current board position
        board = game.end().board()
            
        # Extract move from response using XML tags
        move_match = re.search(r'<answer>(.*?)</answer>', response)
        if not move_match:
            return 0.0  # No move found in proper format
            
        move_text = move_match.group(1).strip()
            
        # Format reward - check if response follows requested structure
        format_reward = 1.0 if (
            '<think>' in response and 
            '</think>' in response and 
            '<answer>' in response and 
            '</answer>' in response
        ) else 0.0
            
        try:
            # Validate move
            chess_move = board.parse_san(move_text)
            if chess_move not in board.legal_moves:
                return 0.0
                
            # Make the move and get evaluation
            board.push(chess_move)
            result = self.engine.analyse(board, self.analysis_limit)
            eval_score = result["score"].relative.score()
            
            # Convert centipawns to [-1, 1] range using sigmoid
            # 100 centipawns = 1 pawn advantage
            move_reward = 2 / (1 + torch.exp(-torch.tensor(eval_score / 100))) - 1
            
            # Combine format and move quality rewards
            # Format is 20% of score, actual move quality is 80%
            # we probably want to adjust this more heavily weighted towards format in the beginning
            return float(format_reward * 0.2 + move_reward * 0.8)
            
        except Exception as e:
            print(f"Error computing reward: {e}")
            return 0.0

    def _create_prompt(self, pgn: str) -> str:
        """
        Create a prompt for the model using PGN format
        Similar to baseline_eval's approach but with added structure
        """
        # Clean PGN - remove headers if present
        cleaned_pgn = pgn.split("\n\n")[-1] if "\n\n" in pgn else pgn
        
        return f"""Analyze this chess position and suggest the best move.
Current game (PGN):
{cleaned_pgn}

Think step by step and provide your reasoning in <think></think> tags.
Then give your chosen move in <answer></answer> tags using standard algebraic notation (SAN).
"""
