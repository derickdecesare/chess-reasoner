"""
generate_dataset.py

This script generates a structured CoT (chain-of-thought) chess reasoning dataset.
It will:
  1. Randomly select a chess game represented by a PNG file (with a corresponding PGN file)
     from a games directory – only pick games that are shorter than 24 moves.
  2. Build a prompt instructing a chess grandmaster (via an API call) to analyze the position
     and provide a structured answer with <think> and <answer> tags.
  3. Send the prompt to the OpenAI, deepseek or anthropic API and wait for a response.
  4. Check the API response for correct formatting and extract the candidate move.
  5. Rebuild the game state from the PGN or FEN and check if the move is legal.
  6. Evaluate the move using Stockfish and only if the evaluation is maintained or improved,
     include it as a training example.
  7. Save all valid training examples to a Parquet file for standard access.
  
Prerequisites:
  - .env file with OPENAI_API_KEY, ANTHROPIC_API_KEY, and STOCKFISH_PATH.
  - A games directory (defaulted to "./src/data/chess/") containing parquet files with the following columns:
    - truncated_pgn: the truncated pgn text (less than 24 moves)
    - fen: the fen string of the position (for easy board reconstruction and subsequent move legality check/ evaluation)

  - python-chess, pandas, python-dotenv, openai and pyarrow libraries installed.
"""

import os
import random
import re
import chess
import chess.pgn
import chess.engine
import pandas as pd
from dotenv import load_dotenv
from src.utils.get_stockfish_path import get_stockfish_path
from src.api_models import openai_client, anthropic_client

# Load environment variables from .env
load_dotenv()
STOCKFISH_PATH = get_stockfish_path()  # Path to your Stockfish binary


# Configuration constants
INPUT_PATH = "./src/data/chess/chess_truncated_pgns_with_fen_100k.parquet"  # file containing PNG (and fen) strings
OUTPUT_PARQUET = "./src/data/chess/cot_training_examples_1k.parquet"
ANALYSIS_LIMIT = chess.engine.Limit(time=0.1)  # Analysis time limit for stockfish (adjust as needed)
DESIRED_MOVES_LIMIT = 24  # Only process games with fewer than 24 moves --> our dateset already has this limit

class DatasetGenerator:
    def __init__(self, input_path=INPUT_PATH, num_games=10, model_name="gpt-4o"):
        self.input_path = input_path
        self.num_games = num_games
        self.model_name = model_name
        self.training_examples = []
        
        # Load the input dataframe
        try:
            self.games_df = pd.read_parquet(input_path)
            print(f"Loaded {len(self.games_df)} games from parquet file")
            # Reset index to ensure clean iteration
            self.games_df = self.games_df.reset_index(drop=True)
        except Exception as e:
            raise RuntimeError(f"Error loading parquet file from {input_path}: {e}")
        
        # Initialize the chess engine (Stockfish)
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        except Exception as e:
            raise RuntimeError(f"Error starting Stockfish at {STOCKFISH_PATH}: {e}")

    def get_next_game(self, index):
        """
        Gets the game at the specified index from the dataframe.
        
        Args:
            index (int): The index of the game to retrieve
            
        Returns:
            A dictionary containing the pgn text and fen string, or None if index is out of bounds
        """
        if index >= len(self.games_df):
            print("No more games available in the dataframe.")
            return None
        
        game = self.games_df.iloc[index]
        return {
            "pgn": game.truncated_pgn,
            "fen": game.fen
        }

    def build_prompt(self, game_info):
        """
        Constructs the prompt to send to the API.
        
        Args:
            game_info (dict): A dictionary containing at least the PGN text and moves.
            
        Returns:
            A string with the prompt instructions.
        """

        print("Pgn:", game_info['pgn'])

        if 'o1' in self.model_name.lower() or 'o3' in self.model_name.lower(): # need a different prompt for reasoning models
            print("Using reasoning model prompt")
            prompt = f"""Analyze this position from the lense of a chess grandmaster who is explaining this to a 1800 rated player ... understanding the pieces positions and thinking about different threats and plans.. after a quick analysis provide your recommended move.. Keep your response concise and useful for 1800 rated player. Ensure to output your final move in standard algebraic notation (SAN).

Chess Position (PGN):
{game_info['pgn']}
"""
            return prompt
        else: # regular gpt-4o models
            print("Using regular model prompt")
            # For this example we will include the PGN.
            # Optionally you might include the FEN too.
            prompt = f"""You are a chess grandmaster. Please analyze this chess position and provide your reasoning and next move.

    Current game (PGN):
    {game_info['pgn']}

    Provide your analysis and move in the following format:

    <think>
    Your detailed reasoning, outlining key threats, piece positions, and any plans.
    </think>
    <answer>
    Your chosen move in standard algebraic notation (SAN)
    </answer>

    Make sure to only output the tags and your answer without extra commentary. And in the answer tag only output the SAN <answer>e4</answer>
    """
        return prompt

    def call_api(self, prompt):
        """
        Sends the prompt to the specified API model and returns the response text.
        
        Args:
            prompt (str): The prompt string to send.
        
        Returns:
            str: The raw text response from the API.
        """
        try:
            if "gpt-4" in self.model_name.lower() or "o3" in self.model_name.lower() or "o1" in self.model_name.lower():
                if "o3" in self.model_name.lower() or "o1" in self.model_name.lower():

                    # Reasoning models don't use temperature or system prompt or max tokens
                    response = openai_client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            # {"role": "developer", "content": "You are an experienced chess grandmaster."},
                            {"role": "user", "content": prompt}
                        ],
                    )
                else:
                    # Regular GPT-4o models use temperature
                    response = openai_client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": "You are a chess grandmaster."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.4,
                        max_tokens=1000
                    )
                api_reply = response.choices[0].message.content
            elif "claude" in self.model_name.lower():
                response = anthropic_client.messages.create(
                    model=self.model_name,
                    max_tokens=1000,
                    temperature=0.4,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                api_reply = response.content[0].text if isinstance(response.content, list) else str(response.content)
            else:
                raise ValueError(f"Unsupported API model: {self.model_name}")
                
            return api_reply.strip()
        except Exception as e:
            print(f"Error calling API: {e}")
            return ""

    def calculate_format_reward(self, response_text):
        """
        Checks if the response text has correct format using <think> and <answer> tags.
        
        Args:
            response_text (str): The API's raw text response.
        
        Returns:
            float: The format reward (0.5 per valid pair of tags found).
        """
        format_reward = 0.0
        if '<think>' in response_text and '</think>' in response_text:
            format_reward += 0.5
            print("Found think tags: +0.5")
        if '<answer>' in response_text and '</answer>' in response_text:
            format_reward += 0.5
            print("Found answer tags: +0.5")
        return format_reward

    def extract_move_from_response(self, response_text):
        """
        Extracts the move string from <answer>...</answer> tags in the API response.
        
        Args:
            response_text (str): The API's response text.
            
        Returns:
            str or None: Returns the move text if found, else None.
        """
        move_match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL)
        if move_match:
            move_text = move_match.group(1).strip()
            # Remove any extraneous characters (e.g., '+' or '#' often used for check/mate)
            move_text = move_text.replace("+", "").replace("#", "")
            return move_text
        else:
            print("No <answer> tag found; returning None.")
            return None

    def is_move_legal(self, fen, move_text):
        """
        Checks if move_text is legal from the given FEN position.
        
        Args:
            fen (str): The FEN string representing the current position
            move_text (str): The candidate move in SAN.
            
        Returns:
            tuple: (bool, board) where bool indicates legality and board is the board state before making the move.
        """
        board = chess.Board(fen)
        try:
            # Parse the candidate SAN move in the current board's context
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
    
    def evaluate_move(self, board, move_text):
        """
        Uses Stockfish analysis to evaluate the current board state, then the state after the move.
        Returns the difference in evaluation if the move is made.
        
        Args:
            board (chess.Board): The current board state.
            move_text (str): The candidate move in SAN.
        
        Returns:
            float: The evaluation difference (new_eval - old_eval).
        """
        # Get evaluation before move
        try:
            init_result = self.engine.analyse(board, ANALYSIS_LIMIT)
            # Use relative evaluation (score relative to the side to move)
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
            new_result = self.engine.analyse(board, ANALYSIS_LIMIT)
            new_eval = new_result["score"].relative.score(mate_score=10000)
        except Exception as e:
            print(f"Error during evaluation after move: {e}")
            board.pop()  # revert move
            return None
        # Revert the move to keep board unchanged
        board.pop()
        eval_diff = new_eval - init_eval
        print(f"Eval before move: {init_eval}, after move: {new_eval}, diff: {eval_diff}")
        return eval_diff

    def process_game(self, game_info):
        """
        Processes a single game.
        
        Args:
            game_info (dict): Dictionary containing the game's pgn and fen
        """
        prompt = self.build_prompt(game_info)
        # print("Prompt:", prompt)
        print("Sending prompt to API...")
        api_response = self.call_api(prompt)
        print("API Response:", api_response)
        
        format_reward = self.calculate_format_reward(api_response)
        if format_reward < 1.0:
            print("Response did not include the required tags. Disregarding this example.")
            return None

        candidate_move = self.extract_move_from_response(api_response)
        if candidate_move is None:
            return None

        # Check move legality using the FEN string
        legal, board = self.is_move_legal(game_info["fen"], candidate_move)
        if not legal:
            return None

        # Evaluate the move
        eval_diff = self.evaluate_move(board, candidate_move)
        if eval_diff is None:
            return None
        if eval_diff < 0:
            print(f"Move {candidate_move} is not an improvement (eval diff {eval_diff}). Discarding this example.")
            return None

        # Construct a training example entry
        training_example = {
            "pgn": game_info["pgn"],
            "fen": game_info["fen"],
            "prompt": prompt,
            "response": api_response,
            "candidate_move": candidate_move,
            "format_reward": format_reward,
            "eval_diff": eval_diff
        }
        return training_example

    def save_examples_to_parquet(self):
        """
        Saves the collected training examples to the specified Parquet file.
        """
        if not self.training_examples:
            print("No training examples to save.")
            return
        df = pd.DataFrame(self.training_examples)
        try:
            df.to_parquet(OUTPUT_PARQUET, index=False)
            print(f"Training examples saved to {OUTPUT_PARQUET}")
        except Exception as e:
            print(f"Error saving to parquet: {e}")

    def run(self):
        """
        Runs the dataset generation process, processing the specified number of games sequentially.
        """
        collected = 0
        index = 0
        attempts = 0
        
        while collected < self.num_games and index < len(self.games_df):
            attempts += 1
            print(f"\nProcessing game attempt {attempts}/{self.num_games}...")
            
            game_info = self.get_next_game(index)
            if game_info:
                example = self.process_game(game_info)
                if example:
                    self.training_examples.append(example)
                    collected += 1
                    print(f"Collected {collected}/{self.num_games} examples.")
                else:
                    print("Example was invalid; proceeding to next.")
            index += 1

        print(f"\nFinished processing. Collected {collected} valid examples from {attempts} attempts.")

        print("\nResults:")
        i =0
        for example in self.training_examples:
            print("\n---Example---")
            print(example)
            i += 1
            if i > 2:
                break

        self.engine.quit()

        # return # early return for testing..
        
        self.save_examples_to_parquet()


def main():
    # Add command line argument parsing if desired
    num_games = 1000  
    model_name = "gpt-4o"  # Can be changed to other models like "claude-3-opus-20240229"
    generator = DatasetGenerator(num_games=num_games, model_name=model_name)
    generator.run()

if __name__ == "__main__":
    main()