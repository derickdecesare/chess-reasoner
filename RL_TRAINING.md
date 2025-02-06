# RL Training Approach for Chess Reasoning

## Goal

Use rl training through GRPO to improve a opensource LLM's ability to play chess by encouraging longer thought processes like was done in other domains like math and coding in deepseeks r-1 paper.

## Overview

We plan to start with a base or instruct model (e.g., Qwen/Qwen2.5-1.5B-Instruct) and train it using only RL signals (no instruct fine tuning). If the model fails to learn the required XML-like format under pure RL, we will minimally fine-tune the base model specifically to improve format compliance. We prefer not to use any instruct fine-tuned model because prior attempts have shown catastrophic forgetting of chess knowledge.

--Update
After analyzing base models vs instruct models there doesn't seem to be noticeable catastrophic forgetting of chess data. However when trying the RL loop with the instruct model it learned to produce the format but never legal moves... so fine tuning on raw PNGs and example chess COT was required.

## Base Model Selection

- Using Qwen/Qwen2.5-1.5B-Base for local testing worst case using the instruct model or doing on own custom fine tuning on the base model perferablly
- Using 3b, 7b, 14b for runpod A100 GPU training (80gb vram)
- Much smaller than DeepSeek's model but good for rapid experimentation
- Use google collab (pro version with a100 and 40gb vram) for rapid iteration before deploying runpod docker.

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

(1) Ray RLlib (https://docs.ray.io/en/latest/rllib.html)
• Highly flexible. Works well if you have a custom environment (i.e., your chess logic + LLM policy).
• Handles distributed rollouts across many worker processes, with a centralized trainer.
• You can implement your “ChessEnv” in Gym style, then plug in your LLM-based policy.
• Good for scaling to large clusters if needed

(4) Custom Multi-Process or Multi-Threaded Implementation
• You can spawn multiple chess environments yourself, gather experiences in parallel, and unify them for policy updates.
• Requires the most boilerplate, but no new dependencies beyond Python’s multiprocessing + PyTorch.

## Thoughts on improving base/instruct models starting chess performance

There is no reason by a 1.5 billion param model shouldn't be able to at least to basic moves.. my theory is there has been some catastrophic forgetting.. I think that we should one.. stick with llama since it performed a little better on our basic tests.. Then we should fine tune that model (full param update) so that it can predict chess pgns better.. and we should also fine tune it with some basic instruct so that it follows instructions? or maybe it would be better to directly fine tune the already instruct fine tuned model... not sure..

if we wanted to be extra safe we would start with the base model and then fine tune it to do some chain of thought thinking maybe intially and then maybe pass the png and then fen as well?

I'm not sure.. since we have proven that pngs are sufficient for the neural net to learn to play chess.. but for chain of though reasoning it might help it to have an explicit understanding of the position with the fen string as well.. --> we could actually test this pretty easily by adding this information to our evaluation logic that we built... and seeing if our models perform any better predicting the next token.. when we provide them with the fen string as well
We should also bookmark the performance against like O1 preview and deepseek r1 to see how much better reasoning models already perform at playing chess in this way... this is definitely solvable

## Using Hugging Face’s TRL + GRPO (instead of custom)

Why:
Simplified PPO/GRPO logic: TRL handles reference models, KL penalty, advantage normalization, etc.
Easy multi-GPU/multi-node training: Built on Hugging Face Accelerate for distributed training.
Logging + checkpointing included: Integrated with Weights & Biases or equivalent.
PEFT/LoRA integration: Finetune large models efficiently by training only low-rank adapters.
Speeding Up Generation with vLLM
Generation is often the biggest bottleneck (especially if you generate many completions).
TRL’s GRPOTrainer can use vLLM (set use_vllm=True in GRPOConfig) to significantly accelerate token sampling and reduce latency during rollout collection.
This is particularly valuable if your environment spends a lot of time waiting on model completions or if you do lots of sampling per step.
Parallelizing the Chess Environment (Short Version)
Use a parallel framework like Ray to create multiple “chess workers.”
Each worker runs Stockfish locally, takes PGN positions, queries the model for completions, computes environment rewards, and returns (prompt, completion, reward) tuples.
A central driver (running TRL’s GRPOTrainer) gathers these experiences and runs the policy update.
After updating, you can broadcast new policy weights back to the workers so the next rollout uses the updated model.
By combining TRL’s built-in GRPO (with vLLM for fast generation) and Ray (or another parallel system) for the environment logic, you can scale both the LLM side (multiple GPUs) and the environment side (multiple Stockfish workers) to achieve efficient training.

# Note on LoRA vs. Full Fine-Tuning During RL

It may be worth to do an empirical experiment to see how far using peft/lora would take it.. even though it's a complex reasoning domain like chess that probably will require significant reshaping of the laten space that I suspect that lora is not capable of delivering..

# Problems/thoughts after intial RL Loop testing

The model we tested qwen2.5 1.5BInstruct was able to output it's response using the correct format.. with the thinking and answer tags.. however it never produced an legal move.

## ideas for improvment

- We need to do some more testing to ensure that our move validation is working correctly.
- If our move validation is working correctly then we need to take measures so that the model will at least start making legal moves with a decent frequency so our rewards are not too sparse.
- Possible ideas to address this... start with shorter games.. so filter our dataset so that we are only showing pgn's with less than 5 moves in the beginning so that it is easier for the model to produce a legal move.
  -Include the fen in the prompt (in addition to the pgn) which may make it easier for the model to think about the position of the chess pieces to produce a legal move.
  -Also based on our initial baseline results llama3.2 1.5BInstruct showed better results when it came to producing legal moves.. so we may want to switch over to that.
- Another idea would be to start with the base model and then do fine tuning on examples of predicting next moves in a chess pgn. Or we could also do this for the instruct model as well.
- We could also start with the base model and then just apply basic formating fine tuning.. as it may be possible that the bad performance from the instruct models could be do to catostropic forgetting as a result of policy drift/ latent space rearrangement during fine tuning
- So maybe it would be good to fine tune a base model to follow format... of thinking and answer and then actually have the right answer (legal and good) chess move in there ... this would take some dataset preparation but might just be what would give us initaly the improvement in performance that could kick off a successful rl training loop.

Validate the Move Validation Logic:
Double-check that board.parse_san(move_text) and subsequent legality checking (with if chess_move not in board.legal_moves) are working as intended.
It might help to log (or even visualize in a test run) the board state and the parsed move to confirm that the execution pipeline for move validation is correct.

Dataset Generation for SFT:
The goal is to create a dataset where each example shows a proper chain-of-thought (i.e. <think>...</think>) followed by a legal chess move in the <answer>...</answer> block.
This dataset doesn’t necessarily have to be generated manually from scratch. You have a couple of options:
Manual Curation:
Engage data labelers or experts to produce a set of examples. While effective, it might be cost and time intensive. (Appen, Scale AI, Lionbridge, Figure Eight)
(An initial dataset of around 100 to 200 examples could be sufficient.)
(Might scale to 500 to 1000 examples as well)
High-Quality Model Filtering:
Use a strong model like o1 or o3 (if available) to generate candidate examples. Then filter out the generations that do not have a legal/good move. Even if the chain-of-thought isn’t expert-level, the goal here is to “seed” your training so that the base model learns the desired format and legal move production early on.

Overall Strategy:
Initial SFT Phase: Fine-tune the base model on a curated or filtered set where each example adheres to the desired format and demonstrates a correct legal move.
RL Phase: Once the model reliably produces the correct output format and legal moves, move into reinforcement learning. At this stage, you can start using the RL signal, potentially with additional rewards for reasoning quality.

## Example prompt for O1 O3 to see if they can generate synthetic data for us (first attempt with o3 seemed promising) -- we could improve it by having it think even simpler about the position and just like confirming where everything is and such

You are a chess grandmaster. Please analyze this chess position and provide your reasoning and next move.

Current game (PGN):

1. c4 e5 2. a3 d6 3. e3 Nf6 4. Nc3 \*

Current position (FEN):
Current FEN: rnbqkb1r/ppp2ppp/3p1n2/4p3/2P5/P1N1P3/1P1P1PPP/R1BQKBNR b KQkq - 2 4

Provide your analysis and move in the following format:

1. First, explain your thinking process inside <think> tags
2. Then provide your chosen move in standard algebraic notation (SAN) inside <answer> tags

For example, your response should look like:
<think>
I see that the center is contested. White has control of e4 with the knight, while Black's bishop on c5 targets f2. The position calls for careful development while maintaining central control...
</think>
<answer>
Nf3
</answer>

---

## experiements to run

- About to run a test on our baseline eval to see if including fen string will help the performance of our models
- Run a test to see the raw next move prediction of raw pgn of our base models (no prompt) just pgn
- See if SOTA reasoning models like o1,o3 and deepseek can generate examples in our desired dataset

Baseline with FEN:
GPT-4o: 25.4 moves (with 5 checkmates!)
Claude-3.5-Sonnet: 22.8 moves
Llama-3.2-3B: 6.5 moves
Llama-3.1-8B: 5.4 moves
Qwen-2.5-1.5B: 4.6 moves --> best bang for buck
Qwen-2.5-3B: 4.1 moves
Qwen-2.5-7B: 3.8 moves
Llama-3.2-1B: 3.0 moves
Mistral-7B: 2.2 moves

Without FEN:
Qwen-2.5-7B: 7.0 moves (↑ from 3.8)
Llama-3.1-8B: 6.5 moves (↑ from 5.4)
Llama-3.2-3B: 5.9 moves (↓ from 6.5)
Qwen-2.5-3B: 5.4 moves (↑ from 4.1)
Qwen-2.5-1.5B: 4.4 moves (≈ same as 4.6)
Llama-3.2-1B: 4.3 moves (↑ from 3.0)
Mistral-7B: 1.5 moves (↓ from 2.2)

This suggests that for most models, the FEN string might actually be adding confusion rather than helping them understand the game state better. This is quite counterintuitive since FEN provides the exact board position!

So the base models aren't getting off to that good of a start .. so maybe it makes sense to add a healthy dose of just raw pgns into the fine tuning as well? or continue pretraining it with a bunch more pngs.. not sure if that process is different than regular fine tuning or not.. but not sure if we need that... cause then we could run into the lack of latent space connection? and maybe it would be better if it developed it's understanding of how to play good chess moves in conjunction with reasoning about the position rather than just throwing out then next move without thinking... but having that latent space representation somewhere in the model could help with it's intuitions? interesting.. and I don't think it would hurt to have it predict next token in a bunch of pgns that are filtered for high elo players

Ok actually so the base models definitely performed worse or qwen... much worse.. but it wasn't so bad for llama

Qwen-2.5-1.5B: 1 moves (↓ from 4.6 and 4.4)
Qwen-2.5-3B: 1.7 moves (↓ from 5.4 and 4.1)
Qwen-2.5-7B: 2.8 moves (↓ from 7 and 3.8)
Llama-3.2-1B: 5.4 moves (↑ from 4.3 and 3.0)
Llama-3.2-3B: 3.7 moves (↓ from 6.5 and 5.4)
Llama-3.1-8B: 2.5 moves (↓ from 5.4 and 6.5)

Ok i've decided to start with further fine-tuning with PGN next token prediction

## Summary: Further Fine-Tuning with PGN Next-Token Prediction

Further Fine-Tuning:
We’re moving forward with further fine-tuning using PGN next-token prediction. The idea is that a model (even with 1B parameters) can reliably predict legal moves deep into a game if it’s been trained on chess PGNs.
Empirical Observations:
In our experiments, instruct models outperformed the base models on PGN next-move prediction.
These observations suggest that catastrophic forgetting might not be an issue with the instruction fine-tuned models. This is encouraging because it means that by further fine-tuning on PGN data (continued pre-training), we can potentially improve chess knowledge while still retaining strong instruction-following capabilities.

## Differences Between Normal Fine-Tuning and Continued Pre-Training:

Data Format:
Normal Fine-Tuning (Instruction Tuning):
Uses a structured prompt with instruction, input, and response formats.
Continued Pre-Training (Domain-Adaptive Pre-Training):
Uses raw, linear PGNs for next-token prediction without extra structure.
Learning Rate & Optimization:
Normal Fine-Tuning:
May use a moderately higher learning rate to quickly adapt to following instructions.
Continued Pre-Training:
Generally uses a smaller learning rate to gently adjust the model’s weights toward better chess representations. This helps avoid overwriting existing general language knowledge.
Scope of Weight Updates:
Normal Fine-Tuning (with PEFT/LoRA options):
Often uses parameter-efficient fine-tuning methods (like LoRA) that update only a subset of model parameters.
Continued Pre-Training:
Requires model-wide weight updates because improving the latent space for chess (the overall chess knowledge) likely requires more holistic changes. So, using methods like PEFT/LoRA is not preferred in this case.

## Plan Moving Forward:

Data Preparation:
Retain a simplified next-token prediction setup—similar to what we have in fine_tuning.py—but adjust the training hyperparameters (especially the learning rate) to be more conservative.
Model Selection:
Based on the observation that instruct models perform better for PGN prediction, consider further fine-tuning the instruct models. This allows you to retain the ability to follow instructions while internally enhancing chess knowledge.
Overall Goal:
By solidifying the model’s baseline chess knowledge through domain-adaptive pre-training on PGNs, we create a foundation for later experiments—such as chain-of-thought reasoning and self-play—that rely on the model’s latent representation of chess.

## Results from 10k fine tune on llama 1B model

Pre finetune:
Instruct Model
Instruct prompt with fen: 3.0 moves
Instruct prompt without fen: 4.3 moves
Base Model
Raw pgn: 2.5 moves

Post finetune 10k pngs:
Instruct Model FineTuned on Raw PGNS
Raw PNG: 8.5
Instruct prompt without fen: 15.4

Post finetune 100k pgns:
Raw PNG: 14
Instruct prompt without fen: 17.1

So kind of marginal performance for how much more compute and data we threw at it with the 100k examples

## moving forward

let's try to train with 100k examples too see if we can match gpt-4o average number of moves
then we should apply this to the 3b param model so that we don't waste time doing rl on a model with insuffient parameters for integrating the rl signal (which the .5b param model has shown does not integrate reward model) --> this will require doing our png finetuing/continued pretraining using runpod since google collab doesn't have enough vram..

### it lost ability to think step by step before giving a chess move:

You: Tell me the 3 most populated cities in the world
Setting `pad_token_id` to `eos_token_id`:128009 for open-end generation.
Model: According to the United Nations, the 3 most populated cities in the world are:

1. Tokyo, Japan
2. Delhi, India
3. Shanghai, China

These cities are part of a larger metropolitan area known as a megacity. The population of these cities is often cited as a key indicator of a city's economic and social influence.

In terms of population, Tokyo has a population of approximately 38,140,000, Delhi has a population of approximately 29,280,000, and Shanghai has a population of approximately 24,160,000. These numbers are estimates and may vary depending on the source.
You: given the following chess png please think about what the best move is and then tell me your recomendation in SAN 1. e4 e5 2.1. e4 b6 2. f4 Bb7 3. Nc3 e6 4. Nf3 Ne7 5. Bb5
Setting `pad_token_id` to `eos_token_id`:128009 for open-end generation.
Model: Bc5 6. O-O O-O 7. d3 a6 8. Bxc6 Nxc6 9. d4 Bb7 10. a3 Ng6 11. f5 Nxf5 12. Kh1 Nc5 13. d5 Ne5 14. Bg5 h6 15. Bxe7 Qxe7 16. Nd4 Bxe4 17. Re1 Qf6 18. Nxe6 dxe6 19. Rxe4 Nd3 20. Qxd3 exd5 21. Qxd5 Qxd4 22. Rad1 Qc5+ 23. Qd4 c5
You: Here is the current png: 1. e4 b6 2. f4 Bb7 3. Nc3 e6 4. Nf3 Ne7 5. Bb5 please think step by step being explicit with your reasoning about why a move might be good and then after telling me your reasoning provide the best move in SAN format.
Setting `pad_token_id` to `eos_token_id`:128009 for open-end generation.
Model: ## Step 1: d3 Bb4+ 6. c3 Bxc3+ 7. bxc3 d5 8. e5 Nbc6 9. O-O Qd7 10. Bd2 Nf5 11. Qe1 O-O-O 12. a4 h6 13. Be3 g5 14. fxg5 hxg5 15. a5 f6 16. axb6 axb6 17. Ba6+ Kb8 18. Rxa8+ Rxa8 19. Qb1 fxe5 20. d4 exf4 21. Bxf4 Nxd4 22. cxd
You: Please solve for x: 3x = 16x + 85 please think step by step and then give the answer
Setting `pad_token_id` to `eos_token_id`:128009 for open-end generation.
Model: Here is the problem:

3x = 16x + 85

Step 1: Subtract 16x from both sides
3x - 16x = 16x - 3x
Step 2: Simplify
x = 85

The answer is 85. Please make sure to check the work.

Here is the work:

3x - 16x = 16x - 3x
3x - 16x = 16x - 3x
3x - 3x = 16x - 16x
3x - 3x = 16x - 16x
3x - 3x = 16x - 16x
You: Here is a chess png: 1. e4 b6 2. f4 Bb7 3. Nc3 e6 4. Nf3 Ne7 5. Bb5 your job is to think about in words the next best move.. after you have though using words about why a move is the best please ouput your move in brackets {} using SAN format
Setting `pad_token_id` to `eos_token_id`:128009 for open-end generation.
Model: . a4 a5 6. Ba4 Nbc6 7. Bb3 Nf5 8. d3 h6 9. Bxf5 Bxf5 10. O-O d5 11. e5 Nd7 12. d4 c5 13. dxc5 bxc5 14. Nd4 Nc6 15. Nb3 Qc7 16. Nc1 O-O 17. Qe2 g6 18. Nd3 Bg7 19. Qe3 Ne7 20. Ne2 Nd5 21. c3 Nxc3 22. bxc3 Bxe5 23. Qxe5 Qxe5
You:

## 100k model --> exhibiting catostrohpic forgetting

Loading model from local path: /Users/derickio/Desktop/chess-reasoner/models/llama-3.2-1b-instruct-finetune_png_100k
Using device: mps

Interactive chat mode. Type 'quit' or 'exit' to stop.

You: Hi please tell me the three cities with the highest population in the world.
Setting `pad_token_id` to `eos_token_id`:128009 for open-end generation.
Model: e4 c5 Nf3 Nc6 d3 d5 Nxe1 dxe4 Qd5 Nf3 Qxe4 Ng5 Qf5 Nf3 Bd7 Bg4 Qe5 Be7 O-O-O Bxf3 Bxf3 Bf6 Qb5 Qc4 Qxc4 Bxc4 Nd4 Bxd3 Nxc2 Bxb1 Nxa1 Nxb1 Rxd1+ Kxb1 Rd8 Nd2 Rxd2 Bb2 Nc2 Bxc1 Kc2 Bxa3 b

## to mitigate catostrophic forgetting we need to not train it for as long on pure pngs and we need to also include the COT examples as well.. or potentially do some regularization strategies

## three options

1. structured cot reasoning fine tune first, then add more raw pgn prediction tasks
2. pgn raw prediction first, then structured cot reasoning
3. a fine tuning round where we have a mix of raw png prediction and cot reasoning tasks

It's an emperical question as to which is best.. but either way we need to build our structured cot reasoning dataset.. so let's work on that

## Structured COT chess reasoning dataset

It would be easiest if we could use a model like deepseek or o3 mini to produce high quality examples for us.. then we could filter them to ensure all of their formatting is correct and the moves are legal and good (using python chess and stockfish to determine legality and quality of the moves) and then we would only include the high quality examples with legal/good moves into the dataset... starting off with around 1k examples could be sufficient and I don't think that would be crazy expensive other then ~ $200 that we would need to pay to get to tier 3 of openai api.. and then we would cross our fingers that they don't flag us... o3mini is prefered bc it showed the ability to generate legal moves and produce good cot.. where as deepseek would often hallucinate moves so it would require more api calls and more filtering..

Before paying $200 to upgrade to o3 let's try it with o1 mini in the api and see if they flag us

Other option would be to pay human labelers which would frankly be annoying, time consuming and expensive too...

Other random thoughts here.. if we wanted to build a really good cot chess reasoning dataset we could provide all of the legal moves to gpt-4o and also we could provide moves that stockfish picked and then have gpt-4o justify these? Maybe it would perform better and explain moves better if it knew what the legal moves were in advance...

Also maybe the reason the api is rejecting our requests for structured response from the reasoning models is because we are telling it to think step by step... maybe if we prompted it in a different way then it wouldn't reject our request.. but it is still an emperacle question as to whether the reasoning model like o3 could provide better chess reasoning than gpt-4o -- especially if we gave it some of the top moves to choose between.

However honestly this dataset doesn't need to be perfect and the point mainly is to get our model already using the right format and playing legal moves so that it can then learn to improve through it's own self play rl. So I think the best bank for your buck would be to not overcomplicate it and just use gpt-4o since it's not rejecting our requests and then we just filter for the ones where it actually plays a good move... and then maybe mess around with a different prompt so we arn't rejected by reasoning models..

## Proceeding with just filtering positive eval moves calls to gpt-4o

Ok so we made 2833 api calls to gpt-4o and we gathered 1000 examples where it generated a cot and then output a legal move that improved the eval of the position. So a 35% chance of generating a reasonable move at any point.

I think 1k examples should be sufficient to train with --> our model that we exposed to 10k examples bc that ones didn't seem to completely break it's other verbal functions.

Here is what a training example looks like:
training_example = {
"pgn": game_info["pgn"],
"fen": game_info["fen"],
"prompt": prompt, # might edit the prompt not sure?
"response": api_response, ## definitely want to use this...
"candidate_move": candidate_move,
"format_reward": format_reward,
"eval_diff": eval_diff
}
example prompt:
You are a chess grandmaster. Please analyze this chess position and provide your reasoning and next move.\n\n Current game (PGN):\n 1. e4 d5 2. Nc3 d4 3. Nce2 e5 4. d3 f6 5. f4 c5 6. fxe5 fxe5 7. Nf3 Nc6 8. Ng3\nNf6 9. Be2 Bg4 10. Bg5 Bxf3 11. Bxf3 Be7 12. O-O \*\n\n Provide your analysis and move in the following format:\n\n <think>\n Your detailed reasoning, outlining key threats, piece positions, and any plans.\n </think>\n <answer>\n Your chosen move in standard algebraic notation (SAN)\n </answer>\n\n Make sure to only output the tags and your answer without extra commentary. And in the answer tag only output the SAN <answer>e4</answer>

example response:
<think>\nIn this position, White has a few strategic considerations. The pawn structure is symmetrical, and both sides have developed their pieces reasonably well. White has a slight lead in development due to having castled already. The bishop on b6 is well-placed, targeting the f2 pawn, but it is also somewhat exposed. White's immediate goals should be to complete development and possibly open the center to take advantage of the lead in development.\n\nOne key idea for White is to play dxe5, which would open the d-file and potentially allow the rook on d1 to become active after the queen moves. However, Black can recapture with the d6 pawn, maintaining a solid pawn structure. Another idea is to develop the c1 bishop, but moves like Bg5 or Be3 don't seem to create immediate threats or improve White's position significantly.\n\nInstead, a move like Na3 could be considered, aiming to reroute the knight to a more active square on c4, where it can pressure the b6 bishop and the d6 pawn. However, this plan is slow and doesn't capitalize on White's lead in development.\n\nA more straightforward approach is to play f4, which supports the e5 pawn and prepares to open the f-file. This move also restricts Black's central pawn structure and prepares for potential kingside attacks.\n\nOverall, the move f4 seems to be the most dynamic and aggressive option, aligning with White's developmental advantage.\n</think>\n<answer>f4</answer>

Some stats on the dataset --> src/data/chess/cot_training_examples_1k.parquet
Prompt length statistics:
count 1000.000000
mean 675.219000
std 80.016259
min 546.000000
25% 607.000000
50% 675.000000
75% 742.000000
max 829.000000
Name: prompt_length, dtype: float64

Response length statistics:
count 1000.00000
mean 1026.46100
std 231.51571
min 584.00000
25% 825.50000
50% 1043.00000
75% 1196.25000
max 1706.00000
Name: response_length, dtype: float64

Evaluation difference statistics:
count 1000.000000
mean 432.846000
std 1003.265017
min 0.000000
25% 64.000000
50% 187.000000
75% 542.000000
max 10956.000000
Name: eval_diff, dtype: float64

max prompt 829.000000 max response 1706.00000
total length is 2176

# Moving forward --> fine tune our llama 1b 10k raw pgn fine tune model with this cot dataset

10k raw pgn fine tuned 1b llama3.2 model showed improvement in generating legal moves.. 15.4 average legal moves generated...
However whenever there was a png in the prompt it would just start predicting the rest of the png... so hopefully the cot dataset will allow it to follow our format and then start to connect it's latent space representation of chess it learned from the pgn fine tuning and connect it with verbal reasoning of the position in the cot dataset... enough so that once we finish with this round of post-training we can kick off the rl loop where it will explore and hopefully start to self learn...

Best case is that this works with the 1b model but if this doesn't work, then I think we should move to either qwen 1.5b or the llama 3b --> because it has been show that 1.5b param model might be the minimum (might bc we know .5b is too small to capture the rl signal).

## Results from finetuning the pngfintuned llama 1b model with our strucutred COT dataset

did a full param fine tune.. on 1k examples..
then did a test in baseline against stockfish .. on average it played "average_num_moves": 7.7
llama-3.2-1b-instruct-png-cot --> 7.7

This is with extracting the move from the <answer></answer> tags... and then having 5 retries with temperature increase like we have done with all of the other tests..

If we provide the same prompt to the 1b,3b,and 8b llama instruct models they literally never produce a legal move in the right format... and this was the reason that our first rl loop failed... they would eventually produce the format but they would never create legal moves..

So this means that we have already improved quite a bit and I think it is worth starting the RL-loop with this model in google collab (a100 with 40gb vram --> collab pro) to rapidly iterate on all of the bugs/optimizations that we might do... want to focus on improving efficiency of our first loop that we set up and utilizing TRL + GRPO --> which will allow us to expand to multiple gpus with accelerate and increase generation speed with vllm which are all integrated in these tools

Then once we do that we should look at the results and then start repeating the process with the larger models.. 3b, 8b, and potentially 14b if they show progress commensurate with their param size...

---

# Journey So Far (SUMMARY):

Initial RL Attempt:
Started by attempting to replicate DeepSeek's approach of pure RL training without fine-tuning
Used base instruct models (Llama, Qwen) directly in RL loop
Models learned to output correct format (<think> and <answer> tags)
However, models never produced legal moves, making reward signals too sparse
This failure led to exploring pre-training/fine-tuning approaches
Initial Testing:
Tested various models (Qwen, Llama) with different sizes
Found models struggled with producing legal moves
Discovered including FEN strings didn't consistently help performance 3. Model Performance Findings:
Base models performed worse than instruct models
Llama showed better baseline chess capabilities than Qwen
GPT-4 and Claude showed strong performance (25+ moves)
Smaller models averaged 3-7 moves
First Improvement Attempt:
Fine-tuned Llama 1B on 10k raw PGN examples
Saw improvement from ~3-4 moves to 15.4 moves average
Further fine-tuning with 100k examples showed marginal gains (17.1 moves)
However, model lost ability to follow produce natural language --> catostropic forgetting of all general knowledge....
Structured COT Dataset Creation:
Created 1000 high-quality examples using GPT-4o
Examples include chain-of-thought reasoning and legal moves
Filtered for moves that improved position evaluation
~35% success rate in generating reasonable moves
Latest Results:
Fine-tuned the PGN-trained Llama 1B model on the 1000 COT examples
Model now averages 7.7 moves while maintaining structured format
Significant improvement over baseline models that couldn't produce legal moves in the correct format
While move count decreased from 15.4 to 7.7, the model now successfully combines chess knowledge with structured reasoning
Current Position:
Have a Llama 1B model that can both play legal moves and follow the structured thinking format
Model shows promise for starting RL training loop
Ready to begin rapid iteration of RL training in Google Colab
Plan to scale successful approaches to larger models (3B, 8B, potentially 14B) if results are promising
Next Steps:
Begin RL training loop with 1B model to identify and resolve implementation challenges
Based on results, consider scaling to larger models
Monitor whether increased parameter count leads to proportional improvements in performance
