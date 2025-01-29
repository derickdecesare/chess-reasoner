# RL Training Approach for Chess Reasoning

## Overview

We plan to start with a base model (e.g., Qwen/Qwen2.5-0.5B-Base) and train it using only RL signals (no instruct fine tuning). If the model fails to learn the required XML-like format under pure RL, we will minimally fine-tune the base model specifically to improve format compliance. We prefer not to use any instruct fine-tuned model because prior attempts have shown catastrophic forgetting of chess knowledge.

## Base Model Selection

- Using Qwen/Qwen2.5-0.5B-Base for local testing worst case using the instruct model or doing on own custom fine tuning on the base model perferablly
- Using 3b, 7b, 14b for runpod A100 GPU training
- Much smaller than DeepSeek's model but good for rapid experimentation

1. **Initial Pure RL Phase**
   - Provide rewards for valid chess moves and correct format usage.
   - Low complexity chess positions at first.
   - Minimal (or no) reliance on any instruct-tuned checkpoint.

### 2. Training Format

<think>
Step 1: Analyze the current board position...
Step 2: Consider potential moves...
Step 3: Evaluate the consequences...
</think>
<answer>e4</answer>

### 3. Reward Structure

a) **Primary Rewards**:

- Stockfish evaluation of proposed move (+1 to -1 scaled)
- Move validity check (0 if invalid)

b) **Format Rewards**:

- Small reward for using correct XML tags
- Length-based rewards for sufficient reasoning steps
- Penalty for missing tags

### 4. Potential Challenges & Solutions

1. **Initial Format Compliance**

   - Risk: Base model may never output in correct format
   - Solutions:
     - Start with very small format rewards
     - Gradually increase format requirements
     - Last resort: Minimal SFT with ~100 examples just for format or start with instruct model (extremley undesirable)

2. **Reward Sparsity**

   - Risk: Model gets stuck with all zero rewards
   - Solutions:
     - Implement curriculum learning
     - Start with simple positions
     - Provide partial rewards for "almost correct" formats

3. **Training Stability**
   - Risk: Training collapse or divergence
   - Solutions:
     - Conservative KL penalty
     - Group-based advantages (GRPO)
     - Regular evaluation checkpoints

### 5. Evaluation Metrics

1. **Format Metrics**:

   - % of responses with correct tags
   - Average reasoning steps per response
   - Format consistency across responses

2. **Chess Metrics**:

   - Stockfish evaluation delta
   - % of legal moves
   - Win rate against different Stockfish levels

3. **Reasoning Metrics**:
   - Reasoning step coherence
   - Pattern recognition accuracy
   - Strategic concept demonstration

## Success Criteria

- **Format**: Model consistently produces valid XML-like structures.
- **Chess Reasoning**: Moves are significantly better than random, validated by a chess engine or a heuristic score.
- **Scaling**: Approach remains stable on single-GPU training for model sizes up to ~7B or 14B parameters.

## NOtes to verify..

in gpro we sample the same position multiple times and then compare the rewards for each response??

# other ideas

provide partial reward if it is in san format even if it isn't a legal move
start with basic positions.. like by sorting our dataset by length or something like that
maybe provide the fen as well to give the model more context
the instruct model may have catestropic forgeting of it's chess knowledge and we may have to work with the base model and do some fine tuning to be workable with us and then progress into the rl the optimized rewards

O1 thoughts
– Instead of partial reward for correct syntax alone, require both correct syntax AND a plausible SAN. For instance:
"If move is purely random or fails a legality check, reward = 0."
"If it’s valid SAN but still fails the engine’s actual legality check, partial reward < legal move reward."

a) Boost the Weight of Legal-Move Rewards
– If a correct format yields +0.1 reward but a legal move yields +1.0 reward, the model has more motivation to find a valid move.

c) Use Curriculum (Simplify the Chess Task)
– Start with extremely simple positions (like a forced mate in 1) so the model quickly sees big positive rewards for the correct move. That helps bootstrap it out of random nonsense.
d) Possibly Fine-Tune the Model to Output Legal SAN Moves
– If the model doesn’t know how to produce legal moves at all, you might do a brief supervised fine-tuning pass on valid SAN sequences from a known game dataset. Then add RL to refine.

“Parallelization” notes
“Parallelization” in RL typically means generating experiences simultaneously from multiple environment instances or “actors,” so the agent sees more data per unit time.
– For example, 8 parallel chess boards, each generating moves, so in a single minute you get 8× the experiences.
• “Batch Size” is how many examples (or experiences) you pass to your network in one forward/backward pass.
• For a large language model + chess, you mainly want to avoid slow single-threaded data collection. So some form of concurrency or parallel environment rollout is beneficial if the environment is your bottleneck.

• Libraries like Ray RLlib, Stable Baselines3, PettingZoo (for multi-agent), or Hugging Face’s RL libraries often provide built-in tools to:
– Launch many environment instances in parallel.
– Synchronize experiences (rollouts) across processes.
– Perform policy updates on the aggregated experience buffer.
• In your custom RL approach, you can replicate that by using multiprocessing or distributed data collection, then periodically combining experiences and doing a policy update.
• Necessity: If your environment (in this case, chess simulations + model inference) is slow to generate data, parallelization speeds up experience gathering significantly. For large models (1.5B+), you also might distribute inference steps across multiple GPUs if needed.
