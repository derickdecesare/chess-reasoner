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
import pandas as pd
import copy
import wandb
import logging
from datetime import datetime
import os
from tqdm import tqdm
import numpy as np


@dataclass
class ChessRLExperience:
    # Changed to store PGN instead of FEN since we're working with game history
    prompt: str
    pgn: str  
    response: str
    reward: float
    advantage: Optional[float] = None




class ChessRLTrainer:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        samples_per_position: int = 5,
        num_positions: int = 8,
        kl_coef: float = 0.001,
        use_wandb: bool = False
        ):
        # Setup logging
        self.setup_logging()
        self.logger.info(f"Initializing ChessRLTrainer with model: {model_name}")
        self.logger.info(f"Using device: {device}")

        # Initialize wandb if enabled
        self.use_wandb = use_wandb
        if self.use_wandb:
            wandb.init(
                project="chess-rl",
                config={
                    "model_name": model_name,
                    "samples_per_position": samples_per_position,
                    "num_positions": num_positions,
                    "kl_coef": kl_coef,
                }
            )


        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=(
                torch.float32 
                if self.device in ["cpu", "mps"]  # Use float32 for CPU/MPS
                else torch.float16  # Use float16 for CUDA
            ),
            device_map="auto"
        )
        self.optimizer = torch.optim.AdamW(
        self.model.parameters(),
        lr=1e-5  # Learning rate 
        )
        # Load the positions dataset at initialization
        try:
            self.positions = pd.read_parquet('src/data/chess/chess_positions_lichess.parquet')
            print(f"Loaded {len(self.positions)} positions from dataset")
        except Exception as e:
            raise RuntimeError(f"Failed to load positions dataset: {e}")
    
        
        # Initialize Stockfish with a moderate ELO for training
        self.engine = chess.engine.SimpleEngine.popen_uci(get_stockfish_path())
        # Set time limit for analysis to balance speed vs accuracy
        self.analysis_limit = chess.engine.Limit(time=0.1)
        self.samples_per_position = samples_per_position
        self.num_positions = num_positions
        # self.positions = pd.read_parquet(positions_dataset_path) # --> still need to build this dataset
        self.position_index = 0
        self.kl_coef = kl_coef 
        self.scaler = torch.cuda.amp.GradScaler(enabled=('cuda' in self.device))
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.model.config.use_cache = False  # Important for gradient checkpointing
        torch.set_default_dtype(torch.float32)


    def setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger('ChessRL')
        self.logger.setLevel(logging.INFO)

        # Create logs directory relative to current working directory
        os.makedirs('logs', exist_ok=True)
        
        # Create handlers
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fh = logging.FileHandler(f'logs/training_{timestamp}.log')
        ch = logging.StreamHandler()
        
        # Create formatters
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
    def generate_response(self, pgn: str, prompt: str, temperature: float = 0.4) -> str:
        """Generate a response for a given game position in PGN format or a prompt"""
        if pgn and not prompt:
            prompt = self._create_prompt(pgn)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Using similar generation parameters to baseline_eval for consistency
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=300,  # Increased to allow for thinking process
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            # Add repetition penalty to prevent loops
            repetition_penalty=1.2,
            # Add number of beams for more focused generation
            # num_beams=1,
            # Early stopping helps prevent rambling
            # early_stopping=True

        )
        # response = self.tokenizer.decode(outputs[0])
        # Only decode the new tokens (exclude the prompt)
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:])
        return response

    def compute_reward(self, pgn: str, eval: float, response: str) -> float:
        """
        Compute reward for a response given the game state
        Returns a float between -1 and 1
        """
        # print(f"Computing reward for response:\n{response}")


        # Format reward - check if response follows requested structure
        format_reward = 0.0
        if '<think>' in response and '</think>' in response:
            format_reward += 0.5
            print("Found think tags: +0.5")
        if '<answer>' in response and '</answer>' in response:
            format_reward += 0.5
            print("Found answer tags: +0.5")
    
        print(f"Total format reward: {format_reward}")

        format_reward_norm = 2.0 * format_reward - 1.0


        # Parse the PGN to get current position
        game = chess.pgn.read_game(io.StringIO(pgn))
        if not game:
            self.logger.warning("Could not parse PGN.")
            return format_reward_norm
        
        # Get the current board position
        board = game.end().board()
            
        # Extract move from response using XML tags
        move_match = re.search(r'<answer>(.*?)</answer>', response)
        if not move_match:
            # No move found, return ONLY the normalized format reward
            print("No <answer> tag found; returning format reward only.")
            return format_reward_norm
            
        move_text = move_match.group(1).strip()
        self.logger.debug(f"Extracted move: {move_text}")

        if not move_text:
            print("Empty move text; returning format reward only.")
            return format_reward_norm
            
        # Validate move and compute reward
        try:
            chess_move = board.parse_san(move_text)
            if chess_move not in board.legal_moves:
                self.logger.debug(f"Illegal move '{move_text}'; returning format reward only.")
                return format_reward_norm
                
            # Make move and get new eval score
            board.push(chess_move)
            result = self.engine.analyse(board, self.analysis_limit)
            new_eval = result["score"].relative.score()

            # calculate eval difference
            eval_diff = new_eval - eval
            
            # Normalize eval difference to [-1, 1] range using tanh
            # Scale factor of 100 means a 1 pawn (100 centipawns) improvement gives ~0.76 reward
            move_reward = torch.tanh(torch.tensor(eval_diff / 100.0))
            self.logger.debug(f"move_reward: {move_reward:.4f} (for eval_diff={eval_diff:.1f})")

            final_reward = 0.2 * format_reward_norm + 0.8 * move_reward
            final_reward_clamped = float(max(min(final_reward, 1.0), -1.0))

            self.logger.debug(f"Combined final reward: {final_reward_clamped:.4f}")
            
          
            return final_reward_clamped

            
        except Exception as e:
            self.logger.warning(f"Error computing move reward: {e}")
            # If error occurs while parsing/applying the move, return format reward only
            return format_reward_norm

    def _create_prompt(self, pgn: str) -> str:
        """
        Create a prompt for the model using PGN format
        Similar to baseline_eval's approach but with added structure
        """
        # Clean PGN - remove headers if present
        cleaned_pgn = pgn.split("\n\n")[-1] if "\n\n" in pgn else pgn
#         return f"""Analyze this chess position and suggest the best move.
# Current game (PGN):
# {cleaned_pgn}

# Provide your analysis in <think></think> tags.
# Then give ONLY your chosen move in <answer></answer> tags using standard algebraic notation (SAN).
# For example:
# <think>
# The position is open, with control of the center being crucial...
# </think>
# <answer>e4</answer>
# """
      
        
        return f"""Analyze this chess position and suggest the best move.
Current game (PGN):
{cleaned_pgn}

Think step by step and provide your reasoning in <think></think> tags.
Then give your chosen move in <answer></answer> tags using standard algebraic notation (SAN).
"""
    


    def save_checkpoint(self, checkpoint_name: str):
        """Save a checkpoint of the model and training state"""
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.pt")
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_name': self.model.name_or_path,
            # Add any other state you want to save
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")

    def sample_position(self):
        """Sample a random position from our Lichess dataset"""
        # Sample random position
        position = self.positions.sample(n=1).iloc[0]
        # replace=False means we don't sample the same position twice
        pgn = position['pgn']
        eval_score = position['position_eval']

        # 1. Truncate if PGN is too long
        max_pgn_length = 1024  # Adjust based on your model’s max token limit
        if len(pgn) > max_pgn_length:
            pgn = pgn[:max_pgn_length]

        # 2. Return both the eval and the truncated PGN
        return eval_score, pgn

    
    def collect_experience(self, pgn: str, eval: float) -> List[ChessRLExperience]:
        """Collect multiple responses/rewards for a single position"""
        experiences = []

        self.logger.debug(f"Collecting experiences for position with eval: {eval}")
        
        # Generate multiple responses for same position
        for i in range(self.samples_per_position):
            prompt = self._create_prompt(pgn) # since they will all be the same position and same prompt
            response = self.generate_response(pgn, prompt)
            reward = self.compute_reward(pgn, eval, response) # need to pass the eval score as well here
            
            experiences.append(ChessRLExperience(
                pgn=pgn,
                response=response,
                reward=reward, 
                prompt=prompt
            ))
            # self.logger.debug(f"Sample {i+1}: Reward = {reward}")
        
        average_reward = sum(exp.reward for exp in experiences) / len(experiences)
        self.logger.info(f"Average reward for position: {average_reward}")

        if self.use_wandb:
            wandb.log({
                "avg_reward_per_position": average_reward,
                "min_reward": min(exp.reward for exp in experiences),
                "max_reward": max(exp.reward for exp in experiences)
            })
        return experiences
        
    def compute_advantages(self, experiences: List[ChessRLExperience]) -> torch.Tensor:
        """Compute advantages for a single group of experiences"""
        rewards = torch.tensor([exp.reward for exp in experiences])
        
        # Normalize within this group
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        return advantages # also a tensor bc rewards is a tensor
    
    def policy_update(self, experiences: List[ChessRLExperience], advantages: torch.Tensor, old_policy):
        """Update model using GRPO
        Args:
            experiences: List of ChessRLExperience objects containing pgn/response pairs
            advantages: Tensor of advantages corresponding to each experience
        Returns:
            loss: Total loss value for this update
        """

        # Improvements --->  TO BE MADE
        # Accumulate loss over multiple batches (8-32 experiences or more) --> check experiences into batches compute the total (average) loss over that batch and then do one backward pass and stop

        # Set model to training mode - enables gradient computation and dropout
        self.model.train()
        total_loss = 0 # Initialize accumulator for batch loss
        policy_losses = []
        kl_losses = []

        accumulation_steps = 4  # Accumulate 4 experiences before updating
        self.optimizer.zero_grad()

        self.logger.info(f"Starting policy update with {len(experiences)} experiences")

        # Add advantage normalization --> another attempt to fix the invalid log_probs issue --> maybe we would also do this by calculating all of the advantages based on the batch of experiences rather than just one group of local experiences on one position?
        # this normalized advantages across the entire batch of experiences
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.clamp(-5, 5)  # Prevent extreme advantage values

        if 'cuda' in self.device:
            device_type = 'cuda'
        elif 'mps' in self.device:
            device_type = 'mps'
        else:
            device_type = 'cpu'

        self.logger.info(f"RL device: {self.device} -> device_type={device_type}")

        # If you want to log actual autocast dtype on CPU/CUDA only:
        if device_type in ('cuda', 'cpu'):
            autocast_dtype = torch.get_autocast_dtype(device_type)
            self.logger.info(f"Autocast config: device={device_type}, dtype={autocast_dtype}")
        else:
            self.logger.info("Autocast config: device=mps => no autocast float32")

        # # Determine device type once
        # device_type = 'cuda' if 'cuda' in self.device else 'mps' if 'mps' in self.device else 'cpu'
        # if device_type in ('cuda', 'cpu'):
        #     autocast_dtype = torch.get_autocast_dtype(device_type)
        #     self.logger.info(f"Autocast config: device={device_type}, dtype={autocast_dtype}")
        # else:
        #     self.logger.info("Autocast config: device=mps, float32 only")

        
        # Iterate through experiences and their corresponding advantages together
        for i, (exp, advantage) in enumerate(zip(experiences, advantages)):
            # ------------------------------------------------------------------------------------
            # OLD APPROACH:
            #
            # # The problem: passing prompt as 'inputs' and
            # # response as 'labels' separately can lead to token misalignment.
            # # The model tries to predict a response with only the prompt tokens
            # # in the input, which might cause shape issues or partial training signals.
            #
            # inputs = self.tokenizer(exp.prompt, return_tensors="pt").to(self.device)
            # targets = self.tokenizer(exp.response, return_tensors="pt").to(self.device)
            #
            # outputs = self.model(**inputs, labels=targets.input_ids)
            # log_probs = -outputs.loss
            # policy_loss = -log_probs * advantage
            # ------------------------------------------------------------------------------------


            # NEW APPROACH:
            # We concatenate the prompt and the response into one sequence,
            # use the same token IDs as both input and labels, and set the prompt portion
            # of 'labels' to -100 (ignored in cross-entropy). This way, the model learns
            # to produce 'response' tokens from the 'prompt' tokens, without penalizing it
            # for re-predicting the prompt itself.
            prompt_tokens = self.tokenizer(
                exp.prompt, 
                return_tensors="pt", 
                add_special_tokens=False
            ).to(self.model.device)
            prompt_len = prompt_tokens.input_ids.shape[1]

            combined_text = exp.prompt + exp.response
            encoded = self.tokenizer(combined_text, return_tensors="pt", add_special_tokens=False).to(self.model.device)

            input_ids = encoded.input_ids
            labels = input_ids.clone()

            # Mask out the prompt portion so loss is only computed on the response.
            labels[0, :prompt_len] = -100  # -100 is ignored by the model's loss function


            
            # Forward pass to get response probabilities
            # Enable gradient tracking for training
            with torch.set_grad_enabled(True):
                # Forward pass through model
                # New Approach:


                # this the proposed solution to our invalid log_probs issue
                # Add logit clamping and numerical stability
                # it will never actualy use bfloat16 if it's mps cause we disable autocast if not cuda
                with torch.autocast(
                        device_type=device_type,
                        dtype=(torch.float16 if device_type == 'cuda' else torch.bfloat16),
                        enabled=(device_type == 'cuda')
                    ):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=encoded.attention_mask,
                        labels=labels
                    )


                
                # Model outputs negative log likelihood loss
                # We negate it to get log probabilities
                # outputs.loss is the standard cross-entropy over "response" tokens only
                log_probs = -outputs.loss.clamp(min=-1e6, max=1e6)  # Prevent explosion --> another attempt to fix the invalid log_probs issue
                # log_probs = -outputs.loss 
                policy_loss = -log_probs * advantage

               
        


                 # Add safety checks
                if torch.isnan(log_probs) or torch.isinf(log_probs):
                    self.logger.warning(f"Invalid log_probs detected: {log_probs}")
                    continue
                


                # attempt to improve KL divergence calculation with temperature
                with torch.no_grad():
                    old_logits = old_policy(input_ids=input_ids, attention_mask=encoded.attention_mask).logits
                    old_logits = old_logits.to(outputs.logits.dtype)  # Match precision
                    
                # Add temperature and clamp
                temperature = 0.7


                log_probs = torch.nn.functional.log_softmax(outputs.logits / temperature, dim=-1)
                old_log_probs = torch.nn.functional.log_softmax(old_logits / temperature, dim=-1)

                # Use PyTorch's built-in KL with stable reduction
                kl_div = torch.nn.functional.kl_div(
                    log_probs,
                    torch.nn.functional.softmax(old_logits / temperature, dim=-1),
                    reduction='batchmean',
                    log_target=False
                ) * (temperature ** 2)  # Temperature scaling compensation
                # probs = torch.nn.functional.softmax(outputs.logits / temperature, dim=-1)
                # old_probs = torch.nn.functional.softmax(old_logits / temperature, dim=-1).clamp(1e-8, 1.0)

                # kl_div = torch.sum(
                #     probs * (torch.log(probs) - torch.log(old_probs)),
                #     dim=-1
                # ).mean()


                # # KL with old policy
                # with torch.no_grad():
                #     old_outputs = old_policy(input_ids=input_ids, attention_mask=encoded.attention_mask)

                # kl_div = torch.nn.functional.kl_div(
                #     torch.nn.functional.log_softmax(outputs.logits, dim=-1),
                #     torch.nn.functional.softmax(old_outputs.logits, dim=-1),
                #     reduction='batchmean'
                # )

                # safety check
                if torch.isnan(kl_div) or torch.isinf(kl_div):
                    self.logger.error("NaN/Inf detected, aborting batch")
                    self.optimizer.zero_grad()
                    raise RuntimeError("Numerical instability detected")

            
                kl_loss = self.kl_coef * kl_div

                policy_losses.append(policy_loss.item()) # for logging
                kl_losses.append(kl_loss.item()) # for logging
                # Total loss
                loss = policy_loss + kl_loss

                # safety check
                if torch.isnan(loss) or torch.isinf(loss):
                    self.logger.warning(f"Invalid loss detected: {loss}")
                    continue


                # Backward pass and optimize
                # loss.backward()
                self.scaler.scale(loss).backward()  # Scaled backward pass


                # Handle both complete batches and the last incomplete batch
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(experiences):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)


                    # Replace self.optimizer.step() with:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                    # self.optimizer.step()
                    # self.optimizer.zero_grad()

                    # Log gradients norm periodically
                    if self.use_wandb:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float('inf'))
                        wandb.log({
                            "gradient_norm": grad_norm,
                            "policy_loss": np.mean(policy_losses[-accumulation_steps:]),
                            "kl_loss": np.mean(kl_losses[-accumulation_steps:]),
                            "loss_scale": self.scaler.get_scale(),
                        })

                # Add this experience's loss to total (detached from graph)
                total_loss += loss.item()
        
        # Compute average loss across all experiences
        avg_loss = total_loss / len(experiences)
        self.logger.info(f"Policy update complete. Average loss: {avg_loss:.4f}")
        self.logger.info(f"Average policy loss: {np.mean(policy_losses):.4f}")
        self.logger.info(f"Average KL loss: {np.mean(kl_losses):.4f}")
        
        # Set model back to evaluation mode
        self.model.eval() 
        return avg_loss


        
    def train_step(self):
        """Run one training iteration"""
        # Potential improvements ---> may want to compute advantages across the entire batch of experiences to lead to more stable updates


        self.logger.info("Starting new training step")

        # freeze the current model as 'old_policy' before gathering experiences
        old_policy = copy.deepcopy(self.model)
        old_policy.eval() # just to be safe, set to eval mode


        # correspond to eachother by index
        all_experiences = [] # array of ChessRLExperience objects
        all_advantages = [] # will be array of tensors
        position_rewards = []
        
        # Collect experiences for multiple positions
        for pos_idx in tqdm(range(self.num_positions), desc="Collecting experiences"):
            eval, pgn = self.sample_position() # make sure we return eval too 
            # Get experiences for this position
            position_experiences = self.collect_experience(pgn, eval)
            position_advantages = self.compute_advantages(position_experiences) # still a tensor

            all_experiences.extend(position_experiences)
            all_advantages.extend(position_advantages) # this converts tensor back to a list... 

            # Track average reward per position
            position_rewards.append(np.mean([exp.reward for exp in position_experiences]))
        
        # Convert back to tensors for policy update
        advantages = torch.tensor(all_advantages)

        # Log statistics before policy update
        self.logger.info(f"Collected {len(all_experiences)} total experiences")
        self.logger.info(f"Average position reward: {np.mean(position_rewards):.4f}")

        if self.use_wandb:
            wandb.log({
                "avg_position_reward": np.mean(position_rewards),
                "min_position_reward": min(position_rewards),
                "max_position_reward": max(position_rewards),
                "advantage_mean": advantages.mean().item(),
                "advantage_std": advantages.std().item()
            })
        
        # Update policy using all collected data
        loss = self.policy_update(all_experiences, advantages, old_policy)


        if self.use_wandb:
            wandb.log({"training_loss": loss})

        
        return loss



    def train(self, num_steps: int):
            """Train for a specified number of steps"""
            self.logger.info(f"Starting training for {num_steps} steps")
            
            losses = []
            try:
                for step in tqdm(range(num_steps), desc="Training"):
                    loss = self.train_step()
                    losses.append(loss)
                    
                    self.logger.info(f"Step {step + 1}/{num_steps} - Loss: {loss:.4f}")
                    
                    # Save checkpoint periodically
                    if (step + 1) % 10 == 0:  # Every 10 steps
                        self.save_checkpoint(f"checkpoint_step_{step + 1}")
            
            except Exception as e:
                self.logger.error(f"Training interrupted due to error: {str(e)}")
                raise
            
            finally:
                if self.use_wandb:
                    wandb.finish()
            
            return losses
