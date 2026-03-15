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

RunPod setup (fresh pod):
  Recommended pod config: A100-80GB, 50GB+ container disk, 100GB+ volume disk.
  If container disk is only 20GB, pip installs will be very tight (~12GB of packages)
  and you'll need to clear pip cache constantly. 50GB avoids this entirely.

  # 1. Install vllm (pulls torch 2.10, correct CUDA libs, transformers, etc.)
  pip install vllm

  # 2. Install unsloth + its deps (must install AFTER vllm so versions align)
  pip install --no-deps unsloth
  pip install unsloth_zoo hf_transfer tyro xformers diffusers

  # 3. Install flash-attn prebuilt wheel (building from source takes forever / fails)
  #    Go to https://github.com/mjun0812/flash-attention-prebuild-wheels/releases
  #    and pick the wheel matching your torch + CUDA + python version.
  #    For torch 2.10, CUDA 12.8, Python 3.11:
  pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu128torch2.10-cp311-cp311-linux_x86_64.whl

  # 4. Clear pip cache (reclaim space on root partition)
  pip cache purge

  # 5. Sanity check — should print "ALL CHECKS PASSED" with no errors
  HF_HOME=/workspace/hf_cache python -c "
  import os; os.environ['HF_HOME']='/workspace/hf_cache'
  from unsloth import FastLanguageModel, PatchFastRL
  PatchFastRL('GRPO', FastLanguageModel)
  import vllm, torch
  from trl import GRPOConfig, GRPOTrainer
  from datasets import Dataset
  print(f'torch={torch.__version__}, vllm={vllm.__version__}, cuda={torch.version.cuda}')
  print(f'GPU: {torch.cuda.get_device_name(0)}')
  print('ALL CHECKS PASSED')
  "

  # 6. Make sure /workspace isn't over quota (delete old checkpoints if needed)
  touch /workspace/.quota_test && rm /workspace/.quota_test && echo "OK" || echo "QUOTA EXCEEDED"
  du -sh /workspace/* | sort -rh

Usage:
  tmux new -s training
  cd /root/chess-reasoner && HF_HOME=/workspace/hf_cache python3 src/rl_training_loop_unsloth.py
  # Detach: Ctrl+B, D
  # Reattach: tmux attach -t training
  # Kill: tmux kill-session -t training
  # Check: tmux ls

Workspace & checkpoint management:
  # Check /workspace disk usage (RunPod volume — usually has a quota)
  du -sh /workspace/* | sort -rh
  df -h /workspace

  # List checkpoints with sizes
  du -sh /workspace/models/qwen-14B-GRPO-unsloth/checkpoint-* | sort -t'-' -k2 -n

  # List just checkpoint names (sorted)
  ls -d /workspace/models/qwen-14B-GRPO-unsloth/checkpoint-* | sort -t'-' -k2 -n

  # Delete ALL checkpoints (keep final/ and interrupted/ if they exist)
  rm -rf /workspace/models/qwen-14B-GRPO-unsloth/checkpoint-*

  # Delete all checkpoints EXCEPT the latest one
  ls -d /workspace/models/qwen-14B-GRPO-unsloth/checkpoint-* | sort -t'-' -k2 -n | head -n -1 | xargs rm -rf

  # Delete a specific checkpoint
  rm -rf /workspace/models/qwen-14B-GRPO-unsloth/checkpoint-300

  # Check HF model cache size
  du -sh /workspace/hf_cache/hub/models--* | sort -rh

  # Test if /workspace is writable (quota not exceeded)
  touch /workspace/.quota_test && rm /workspace/.quota_test && echo "OK" || echo "QUOTA EXCEEDED"

  # Clear pip cache (lives on root partition, fills up fast on 20GB RunPod root)
  pip cache purge
"""

import os
import logging
from typing import List
from datetime import datetime

# ---------------------------------------------------------------
# Unsloth MUST be imported before trl, transformers, peft so it can
# monkey-patch them for optimized kernels and memory management.
# PatchFastRL patches TRL's vLLM weight sync path to handle Unsloth's
# quantized LoRA tensors correctly — without it, merge_adapter() fails
# with a tensor size mismatch during the first generation step.
# ---------------------------------------------------------------
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

import pandas as pd
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

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
    xmlcount_reward_func,
    thinking_length_reward_func,
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
            max_seq_length=2560,          # prompt (~230 tokens) + completion (2048 tokens)
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
            lora_dropout=0.0,
            use_gradient_checkpointing="unsloth",  # ★ Unsloth's smarter gradient checkpointing (offloads to system RAM)
        )

        # -----------------------------------------------------------
        # GRPOConfig — mostly identical to TRL-only version
        # -----------------------------------------------------------
        self.training_args = GRPOConfig(
            output_dir=self.output_dir,
            run_name=self.run_name,
            learning_rate=1e-5,
            adam_beta1=0.9,
            adam_beta2=0.99,
            weight_decay=0.1,
            warmup_steps=100,
            lr_scheduler_type='cosine',
            logging_steps=1,
            bf16=True,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_generations=12,
            max_completion_length=2048,
            num_train_epochs=1,
            save_steps=50,
            max_grad_norm=1.0,
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
                xmlcount_reward_func,
                thinking_length_reward_func,
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
            self.trainer.train()
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
        max_rows=None,  # full 10k — ~2500 steps
    )
    trainer.train()


if __name__ == "__main__":
    main()
