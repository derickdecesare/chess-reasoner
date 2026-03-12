1. Setup Phase:

- Install system dependencies: `apt-get install -y tmux`
- Install Python dependencies (transformers, trl, peft, accelerate, bitsandbytes, datasets, wandb, python-chess, unsloth, vllm)
- Verify GPU/CUDA availability
- Verify model loading and tokenizer
- Verify precomputed eval parquet loads correctly
- Test reward function imports

2. Sanity Checks:

- Run a small import test of the training script
- Confirm dataset loads (10k prompts + 315k precomputed evals)
- Confirm Unsloth + vLLM imports work

3. Training:

- Run rl_training_loop_unsloth.py (Unsloth + vLLM fast version)
- Monitor via wandb
- Checkpoints saved to /workspace/models/
