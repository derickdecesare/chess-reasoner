"""
Chess RL Training Loop using Unsloth + TRL GRPO + vLLM
=======================================================

This is the FAST version of rl_training_loop_trl.py.
The only difference is HOW the model is loaded and HOW generation happens:
  - Model loading:  Unsloth's FastLanguageModel  (instead of AutoModelForCausalLM + BitsAndBytesConfig)
  - Generation:     vLLM backend via fast_inference=True (instead of HuggingFace generate())
  - LoRA setup:     Unsloth's get_peft_model  (instead of passing peft_config to GRPOTrainer)

Everything else — reward functions, dataset loading, GRPOConfig, GRPOTrainer — is identical.
The reward functions are imported from the original file so there's exactly ONE source of truth.

Why this is faster:
  - vLLM uses PagedAttention and continuous batching for generation
  - Unsloth shares GPU memory between vLLM (inference) and training via "sleep mode":
    vLLM releases VRAM during backward pass, reclaims it during forward/generation
  - This can reduce per-step time from ~70s to ~15-20s (3-5x speedup)

Requirements (in addition to the base requirements):
  pip install unsloth vllm
  
Usage:
  HF_HOME=/workspace/hf_cache python3 src/rl_training_loop_unsloth.py
"""

import os
import logging
from typing import List
from datetime import datetime
import pandas as pd
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

# ---------------------------------------------------------------
# Unsloth replaces: AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LoraConfig
# All of those are handled internally by FastLanguageModel.
# ---------------------------------------------------------------
from unsloth import FastLanguageModel

# ---------------------------------------------------------------
# Import ALL reward functions from the original training script.
# This ensures there is exactly ONE definition of each reward function
# shared between the TRL-only and Unsloth versions.
# ---------------------------------------------------------------
from rl_training_loop_trl import (
    move_analysis_log_func,
    legal_move_reward_func,
    good_move_reward_func,
    checkmate_reward_func,
    strict_format_reward_func,
    soft_format_reward_func,
)

os.environ["HF_HOME"] = "/workspace/hf_cache"
os.environ["WANDB_PROJECT"] = "chess-reasoner"


class ChessRLTrainerUnsloth:
    """
    Same training loop as ChessRLTrainer but using Unsloth for model loading
    and vLLM for fast generation. All reward functions are identical.
    
    Key differences from ChessRLTrainer (marked with ★ in comments):
      1. Model loaded via FastLanguageModel.from_pretrained() — handles quantization internally
      2. LoRA applied via FastLanguageModel.get_peft_model() — not passed to GRPOTrainer
      3. fast_inference=True enables vLLM backend on the same GPU
      4. gpu_memory_utilization controls the vLLM ↔ training memory split
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-14B-Instruct",
        dataset_path: str = "src/data/chess/chess_rl_fen_pgn_prompt_10k.parquet",
        max_rows: int = None,
    ):
        self.model_name = model_name
        self.logger = logging.getLogger("ChessRL-Unsloth")
        logging.basicConfig(level=logging.INFO)

        self.output_dir = "/workspace/models/qwen-14B-GRPO-unsloth"
        self.run_name = f"qwen-14B-GRPO-unsloth-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # -----------------------------------------------------------
        # Dataset loading — identical to the TRL-only version
        # -----------------------------------------------------------
        self.dataset = pd.read_parquet(dataset_path)
        if max_rows is not None and len(self.dataset) > max_rows:
            self.dataset = self.dataset.sample(n=max_rows, random_state=42).reset_index(drop=True)
            self.logger.info(f"Sampled {max_rows} positions from dataset.")
        self.logger.info(f"Loaded {len(self.dataset)} positions from dataset.")

        if not isinstance(self.dataset, Dataset):
            self.dataset = Dataset.from_pandas(self.dataset)
            self.logger.info("Converted dataset from pandas DataFrame to HuggingFace Dataset.")

        def wrap_prompt_as_chat(example):
            example["prompt"] = [{"role": "user", "content": example["prompt"]}]
            return example
        self.dataset = self.dataset.map(wrap_prompt_as_chat)
        self.logger.info("Wrapped prompts as chat messages for chat template formatting.")

        # -----------------------------------------------------------
        # ★ MODEL LOADING — this is where Unsloth differs from TRL-only
        # -----------------------------------------------------------
        # In the TRL-only version, we did 3 separate steps:
        #   1. BitsAndBytesConfig (4-bit quantization settings)
        #   2. AutoModelForCausalLM.from_pretrained(quantization_config=bnb_config)
        #   3. LoraConfig passed to GRPOTrainer which calls get_peft_model internally
        #
        # Unsloth collapses all of that into TWO calls:
        #   1. FastLanguageModel.from_pretrained() — loads model + applies quantization
        #   2. FastLanguageModel.get_peft_model() — applies LoRA adapters
        #
        # The key new parameters:
        #   - fast_inference=True: enables vLLM as the generation backend
        #     Instead of HuggingFace's generate() (which is slow, sequential, no KV paging),
        #     generation now goes through vLLM (PagedAttention, continuous batching, optimized CUDA kernels).
        #     Unsloth runs vLLM on the SAME GPU in "colocate mode" — they share VRAM.
        #
        #   - gpu_memory_utilization=0.6: tells vLLM to use 60% of GPU memory for its KV cache.
        #     The remaining 40% is for the training forward/backward pass.
        #     If you get OOM during training: lower this (e.g. 0.5).
        #     If you get OOM during generation: raise this (e.g. 0.7).
        #
        #   - max_lora_rank=64: pre-allocates LoRA adapter space in vLLM's memory pool.
        #     Must match the r= value used in get_peft_model below.
        # -----------------------------------------------------------
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=1280,          # prompt (~230 tokens) + completion (1024 tokens)
            load_in_4bit=True,            # ★ replaces BitsAndBytesConfig — same NF4 quantization under the hood
            fast_inference=True,          # ★ NEW: enable vLLM generation backend on the same GPU
            max_lora_rank=64,             # ★ NEW: pre-allocate LoRA rank space in vLLM (must match r= below)
            gpu_memory_utilization=0.6,   # ★ NEW: 60% of VRAM for vLLM KV cache, 40% for training
        )

        # ★ Apply LoRA adapters via Unsloth (instead of passing peft_config to GRPOTrainer)
        # Unsloth's get_peft_model handles the PEFT wrapping internally with its own optimizations
        # (gradient checkpointing is handled more efficiently by Unsloth than the standard PEFT implementation)
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=64,                                # same rank as TRL-only version
            lora_alpha=16,                       # same alpha — effective LR scales as alpha/r
            target_modules=[                     # same target modules
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0.05,
            use_gradient_checkpointing="unsloth",  # ★ Unsloth's smarter gradient checkpointing (offloads to system RAM)
        )

        # -----------------------------------------------------------
        # GRPOConfig — mostly identical to TRL-only version
        # -----------------------------------------------------------
        self.training_args = GRPOConfig(
            output_dir=self.output_dir,
            run_name=self.run_name,
            learning_rate=5e-6,
            adam_beta1=0.9,
            adam_beta2=0.99,
            weight_decay=0.1,
            warmup_steps=100,
            lr_scheduler_type='cosine',
            logging_steps=1,
            bf16=True,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_generations=4,
            max_completion_length=1024,
            num_train_epochs=1,
            save_steps=50,
            max_grad_norm=0.1,
            report_to="wandb",
            log_on_each_node=False,
            # ★ No gradient_checkpointing=True here — Unsloth handles it via
            # use_gradient_checkpointing="unsloth" in get_peft_model above.
            # Setting it in both places can cause conflicts.
        )

        # -----------------------------------------------------------
        # ★ GRPOTrainer — note: NO peft_config argument
        # -----------------------------------------------------------
        # In the TRL-only version, we passed peft_config=self.peft_config to GRPOTrainer
        # and it called get_peft_model internally.
        # With Unsloth, LoRA is already applied above, so we pass the wrapped model directly.
        # Passing peft_config here would cause it to try to wrap LoRA a second time → error.
        self.trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=[
                move_analysis_log_func,
                legal_move_reward_func,
                good_move_reward_func,
                checkmate_reward_func,
                strict_format_reward_func,
                soft_format_reward_func,
            ],
            args=self.training_args,
            train_dataset=self.dataset,
            # ★ NO peft_config — LoRA already applied by FastLanguageModel.get_peft_model()
        )


    def train(self):
        """
        Run the GRPO training loop. Identical to ChessRLTrainer.train()
        except for the output directory paths.
        """
        try:
            self.logger.info("Starting training with Unsloth + vLLM...")
            self.trainer.train(resume_from_checkpoint=True)
            self.logger.info("Training complete. Saving final model...")

            final_path = f"{self.output_dir}/final"
            self.trainer.save_model(final_path)
            self.tokenizer.save_pretrained(final_path)
            self.logger.info(f"Final model saved to {final_path}")

        except KeyboardInterrupt:
            self.logger.info("Training interrupted. Saving checkpoint...")
            interrupt_path = f"{self.output_dir}/interrupted"
            self.trainer.save_model(interrupt_path)
            self.tokenizer.save_pretrained(interrupt_path)
            self.logger.info(f"Interrupted checkpoint saved to {interrupt_path}")
            raise


def main():
    trainer = ChessRLTrainerUnsloth(
        model_name="Qwen/Qwen2.5-14B-Instruct",
        dataset_path="src/data/chess/chess_rl_fen_pgn_prompt_10k.parquet",
        max_rows=700,  # same as TRL-only for comparison; increase for longer runs
    )
    trainer.train()


if __name__ == "__main__":
    main()
