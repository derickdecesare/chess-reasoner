## Project Overview

This project aims to build a **chess reasoner** using a purely **rule-based Reinforcement Learning (RL)** approach—drawing inspiration from methods like DeepSeek-R1 in which a model is incentivized to produce detailed “chain-of-thought” reasoning. We begin with an existing 7B-parameter base language model and integrate a reward mechanism that relies solely on programmatic checks:

1. **Chess Engine Evaluation** – We use Stockfish (or another engine) to assess the quality of moves.
2. **Chain-of-Thought Formatting** – We reward the model for placing its reasoning in `<think></think>` tags and providing a final move in `<answer></answer>`.
3. **Group-based PPO/GRPO** – We generate multiple responses per board state, compute a reward for each, and then update the policy by comparing how each response fares relative to the group average.
4. **Potentially other rewards to encourage cot length** -- Need more research on this to ensure it will not be exploited.

Through iterative training, the model learns to:

- Produce valid, reasonably strong chess moves,
- Provide readable, step-by-step reasoning (chain of thought) for its move choices,
- Improve its behavior without relying on any supervised fine-tuning data.

The end goal is to replicate the _RL-only_, “self-evolving” approach—where correctness and formatting are rewarded—thus encouraging the model to refine its chess knowledge and reasoning skills purely from the reward signals, much like **DeepSeek-R1-Zero** did for math/coding tasks. This README contains the essential setup steps, from choosing libraries and hardware to defining prompts and configuring the training loop.

# Initial Setup for a Chess RL Project with a 7B Model

Below is a concise checklist to get started on a GRPO-like RL pipeline for chess, using a 7B LLM and Stockfish.

---

## 1. Environment & Libraries

- **Python 3.8+**
- **Hugging Face Transformers**
  - For loading a 7B pre-trained model and handling text generation.
- **Hugging Face TRL** (or a fork/alternative that supports group-based PPO)
  - For the RL loop (sampling, advantage calculation, policy updates).
- **PyTorch (GPU-enabled)**
  - Core deep-learning framework.
- **Stockfish (or Another Chess Engine)**
  - For evaluating the model’s proposed moves and providing numeric rewards.
  - Install via a system package or a Python wrapper (e.g., `pip install python-chess`) which can interface with Stockfish.

---

## 2. Hardware / Compute

- **GPUs**
  - Cloud or on-premise. At least one high-memory GPU if possible (e.g., 24GB+), since a 7B model can be large in VRAM.
- **Multi-GPU or Distributed Setup (Optional)**

  - If you want faster RL training, consider multi-GPU (e.g., using DeepSpeed or another parallel framework).

  ### Cloud Options:

  ✅ RunPod Pros:

- Simple setup
- Pay-per-use
- Good GPU options (A100s, H100s)
- Docker-based (portable)
- SSH access

❌ GCP/Azure Cons:

- Complex setup
- Ongoing costs
- More overhead

## Were going to use RUNPOD

---

## 3. Data / Prompts

- **Chess Positions**

  - You’ll feed the model positions in PGN notation (or a simple textual description).

- **Prompt Format**

  - "Board PGN: <pgn_here>\nThink step by step (in <think></think>) about the next move.\nProvide final move in <answer></answer>."

- **Multiple Samples per Position**
  - For each position, you’ll sample _G_ responses from the old policy for advantage calculation.

---

## 4. Reward Function

- **Chess Engine Evaluation**

  - Parse the move from `<answer>` and feed it into Stockfish.
  - Stockfish returns an evaluation (e.g., +1.5 means a better position for White).
  - Convert that numeric eval into a reward, e.g., `reward = engine_eval / scaling_factor`.

- **Validity & Formatting Checks**

  - If the move is invalid or the model omits `<answer></answer>` tags, `reward = 0`.

- **Optional**
  - Small bonus for a chain-of-thought that meets length or formatting criteria.

---

## 5. Training Loop (GRPO / PPO)

1. **Sampling**

   - Use `model.generate()` to produce _G_ responses per position (prompts from your dataset).

2. **Compute Rewards**

   - Evaluate each response with Stockfish + format checks.

3. **Calculate Advantages**

   - \(A_i = \frac{r_i - \mu(r)}{\sigma(r)}\) (group-based)

4. **Policy Update**

   - Pass `(prompts, responses, advantages)` to the PPO/GRPO step.
   - Apply a KL penalty to avoid policy collapse.

5. **Repeat**
   - Iterate for many RL steps/epochs, sampling new responses from the updated policy each time.

---

## 6. Hyperparameters & Logging

- **Key Hyperparams**

  - Batch size
  - Number of responses per position (G)
  - KL coefficient (`β`)
  - Learning rate

- **Logging**
  - Track average reward, advantage, engine evaluations, etc.
  - Evaluate the model periodically (e.g., sample test positions).

---

## 7. Practical Tips

- **Start Small**

  - Test on fewer board states or a shallow engine depth first.

- **Checkpointing**

  - Save model states frequently.

- **Validate**

  - Check that the final moves are indeed improving over time and that the chain-of-thought is not nonsense.

- **Costs**
  - Watch out for GPU usage and time spent calling Stockfish.

### baseline measurements

we should have the instruct version of qwen play some chess games against different levels of stockfish to get a baseline on how good they are at playing chess already...

we could even compare with small versions of llama as well to see which one has better abilities for chess already..

we could consider doing a basic fine tune on the base models if we want to ensure reliable output format, but I think the instruct fine tunes should probably be sufficient -- since there is evidence that basic instruction fine tunes doesn't cause catestropic chess forgetting like rlhf does.. unless the basic finetune with the instruct models does contain rlhf as well.. so we would need to test and verify..

Would be even good to try to get some baselines established for the deepseek r1 models as well.. to see if their reasoning rl feedback improved their chess playing abilities..

basically need to set up a script that will deal with all of the nightmare formatting to get these things to play against different level of stockfish..

we will need to do these before we start training so that we can understand if the darn thing is actually improving at chess

## pod plan

A100 80GB: ~$1.99/hour
Start with single A100 80GB for:
3B model experiments (plenty of headroom)
7B model training (comfortable fit)
14B model with reduced batch size or gradient accumulation
When ready for full 14B training:
Upgrade to 2x A100 80GB setup
This gives you the headroom needed for full batch training

### A100 GPU Options Trade-off

**A100 PCIe ($1.64/hr)**

- 80GB VRAM, 8 vCPUs, 117GB RAM
- High availability
- Good for sequential processing

**A100 SXM ($1.89/hr)**

- 80GB VRAM, 16 vCPUs, 125GB RAM
- Low availability
- Better for parallel processing

The extra $0.25/hr (~$6/day) for SXM gets you 2x CPU cores, which could be valuable if running many parallel Stockfish evaluations or processing large batch sizes. For more sequential workloads, the PCIe version should suffice.

## minimal plan for rl loop locally

# Example minimal test setup

def test_rl_components(): # Test with tiny model (or dummy model)
tiny_model = "Qwen/Qwen2.5-0.5B-Instruct" or Qwen/Qwen2.5-0.5B

    # Test reward calculation
    def test_reward_function():
        stockfish = Stockfish()
        # Test various positions and moves

    # Test training loop without actual training
    def test_training_loop():
        # Use small batch size
        # Verify data flow
        # Check logging

    # Test format validation
    def test_move_formatting():
        # Verify think/answer tags
        # Check move parsing

## Development & Testing Pipeline

### 1. Local Development (Mac/MPS)

- Initial development with MPS acceleration
- Test core logic and training loop
- Debug reward functions
- Verify memory management

### 2. Local Docker Testing

- Validate container builds
- Test dependencies and Stockfish
- Verify data mounting
- Ensure environment consistency

### 3. RunPod Deployment

- Deploy only after local testing passes
- Start with short training runs
- Monitor GPU utilization
- Track costs
