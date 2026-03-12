# Chess Dataset Guide

This document describes every parquet file in this folder — what it contains, how it was generated, and how it's used in the training pipeline.

---

## Pipeline Overview

```
chess_pgn_dataset_10k.parquet          (raw games)
         │
         ▼
chess_truncated_pgns_with_fen_10k.parquet   (games cut mid-game + FEN computed)
         │
         ▼
chess_rl_fen_pgn_prompt_10k.parquet    (prompts added → fed to RL training)
         │
         ├──► precomputed_move_evals_10k.parquet   (Stockfish evals for all legal moves)
         │
         └──► cot_training_examples_1k.parquet     (supervised fine-tune examples with full <think> responses)
```

---

## Files

---

### `chess_pgn_dataset_10k.parquet`

**What it is:** Raw chess games scraped from Lichess. Each row is one complete game.

**Shape:** 10,000 rows × 2 columns

| Column | Type   | Description |
|--------|--------|-------------|
| `pgn`  | string | Full game in PGN notation (move list only, no headers) |
| `elo`  | int    | Average Elo of the two players |

**Example row:**
```
pgn: "1. e4 b6 2. f4 Bb7 3. Nc3 e6 4. Nf3 Ne7 5. Bb5 Ng6 6. d3 Bb4 7. O-O O-O ..."
elo: 1812
```

**Used for:** Source material. Fed into `create_10k_prompt_dataset.py` to build the truncated + FEN datasets.

**100k version:** `chess_pgn_dataset_100k.parquet` — same structure, 100,000 rows. Not used in current training (we use the 10k subset because our precomputed evals only cover those 10k games).

---

### `chess_truncated_pgns_with_fen_10k.parquet`

**What it is:** Each game from the 10k dataset is cut at a random mid-game point. The FEN (board state) at that cut point is computed with python-chess. This is the intermediate step before prompts are added.

**Shape:** 10,000 rows × 7 columns

| Column | Type   | Description |
|--------|--------|-------------|
| `truncated_pgn`   | string | PGN up to the splice point (ends with `*`) |
| `fen`             | string | FEN string representing the board state at the splice point |
| `original_pgn`    | string | The full game before truncation |
| `number_of_moves` | int    | Total moves in the original game |
| `splice_point`    | int    | Token index (in the raw PGN string) where the cut was made |
| `last_number`     | int    | The move number at which the game was cut |
| `full_moves`      | int    | Number of complete move pairs played before the cut |

**Example row:**
```
truncated_pgn: "1. e4 b6 2. f4 Bb7 ... 18. Bb3 Nd7 *"
fen:           "r4rk1/1b1n1n1p/p1p2pp1/1p2qN2/1P2P1Q1/1BNPB3/1PP3PP/R4RK1 w - - 0 21"
original_pgn:  "1. e4 b6 2. f4 Bb7 ... (full game)"
number_of_moves: 57
splice_point:    40
last_number:     29
full_moves:      20
```

**FEN format explained:**
```
r4rk1/1b1n1n1p/p1p2pp1/1p2qN2/1P2P1Q1/1BNPB3/1PP3PP/R4RK1 w - - 0 21
│                                                             │ │ │ │ │
│                                                             │ │ │ │ └─ Full move number (21)
│                                                             │ │ │ └─── Halfmove clock (0 = no captures since last pawn move)
│                                                             │ │ └───── En passant target square (- = none)
│                                                             │ └─────── Castling rights (- = neither side can castle)
│                                                             └───────── Side to move (w = White)
└─────────────────────────────────────────────────────────────────────── Board ranks 8→1 (/ separates ranks, numbers = empty squares)
```

**Used for:** Intermediate step. `chess_rl_fen_pgn_prompt_10k.parquet` is built from this by adding the prompt text.

---

### `chess_rl_fen_pgn_prompt_10k.parquet`

**What it is:** The main RL training dataset. Same as the truncated dataset above but with a formatted prompt column added. This is what gets fed directly to `GRPOTrainer`.

**Shape:** 10,000 rows × 7 columns

| Column | Type   | Description |
|--------|--------|-------------|
| `pgn`            | string | Truncated PGN up to the splice point |
| `fen`            | string | FEN at the splice point (used by reward functions to validate moves) |
| `number_of_moves`| int    | Total moves in original game |
| `splice_point`   | int    | Token index of the cut |
| `last_number`    | int    | Move number at cut |
| `full_moves`     | int    | Full move pairs before cut |
| `prompt`         | string | The full text prompt sent to the model |

**Full prompt example:**
```
You are a chess grandmaster. Please analyze this chess position and provide your reasoning and next move.

    Current game (PGN):
    1. e4 b6 2. f4 Bb7 3. Nc3 e6 4. Nf3 Ne7 5. Bb5 Ng6 6. d3 Bb4 7. O-O O-O 8. a3
f6 9. axb4 a6 10. Ba4 b5 11. Bb3 c6 12. f5 Ne5 13. fxe6 dxe6 14. Bxe6+ Nf7 15.
Nh4 Kh8 16. Qh5 Kg8 17. Be3 Qe7 18. Bb3 Nd7 19. Nf5 Qe5 20. Qg4 g6 *

    Current position (FEN):
    r4rk1/1b1n1n1p/p1p2pp1/1p2qN2/1P2P1Q1/1BNPB3/1PP3PP/R4RK1 w - - 0 21

    Provide your analysis and move in the following format:

    <think>
    Your detailed reasoning, outlining key threats, piece positions, and any plans.
    </think>
    <answer>
    Your chosen move in standard algebraic notation (SAN)
    </answer>
```

**Used for:** Direct input to `rl_training_loop_trl.py`. The `fen` column is passed as a kwarg to every reward function so it can validate the model's move against the actual board state.

**100k version:** `chess_rl_fen_pgn_prompt_100k.parquet` — same structure, 100,000 rows. Not used because we only have precomputed Stockfish evals for the 10k games.

---

### `precomputed_move_evals_10k.parquet`

**What it is:** Stockfish evaluations for every legal move at every position in the 10k training dataset. This lets the reward function assign quality scores without running Stockfish live during training (which would be extremely slow).

**Shape:** 315,552 rows × 5 columns (~34 legal moves per position on average)

| Column      | Type   | Description |
|-------------|--------|-------------|
| `fen`       | string | The board position in FEN notation |
| `move`      | string | A legal move in SAN (e.g. `Nf3`, `exd5`, `O-O`) |
| `eval_diff` | float  | Centipawn change after playing this move (`new_eval - init_eval`). Positive = better for the side to move. |
| `init_eval` | float  | Stockfish evaluation (cp) before the move |
| `new_eval`  | float  | Stockfish evaluation (cp) after the move |

**Example rows:**
```
fen                                               move   eval_diff  init_eval  new_eval
1N3rk1/p4ppp/1p1N1qb1/2p4n/P7/7P/1PPQ1PBP/4RRK1  Nf4    -138.0    -386.0    -524.0
1N3rk1/p4ppp/1p1N1qb1/2p4n/P7/7P/1PPQ1PBP/4RRK1  Rxb8   -201.0    -386.0    -587.0
1N3rk1/p4ppp/1p1N1qb1/2p4n/P7/7P/1PPQ1PBP/4RRK1  Rd8    -258.0    -386.0    -644.0
```

**Understanding `eval_diff`:**
- `eval_diff` is always from the perspective of the **side to move**
- Positive = the move improved your position
- Negative = the move worsened your position (blunder)
- Range in this dataset: -19,997 cp to +10,021 cp (extremes = near-checkmate positions)

**How reward tiers map to `eval_diff`:**
```
eval_diff >= +200 cp  →  EXCELLENT  → reward +3.0
eval_diff  +50–199 cp →  GOOD       → reward +2.0
eval_diff    0–49 cp  →  DECENT     → reward +1.0
eval_diff  -50–-1 cp  →  INACCURACY → reward +0.5
eval_diff  < -50 cp   →  BLUNDER    → reward  0.0
```

**Used for:** `lookup_eval_diff()` in `rl_training_loop_trl.py`. Keyed as `(fen, move_san)` dict at startup for O(1) lookup.

---

### `chess_positions.parquet`

**What it is:** Chess positions sampled from Stockfish self-play games at various Elo levels (1320, 1500, 1800, 2000, etc.), with position evaluations and game phase labels.

**Shape:** 4,500 rows × 6 columns

| Column           | Type   | Description |
|------------------|--------|-------------|
| `pgn`            | string | Full PGN with headers (includes Event, Site, Date, Result metadata) |
| `elo`            | int    | Stockfish Elo level used for the game |
| `move_number`    | int    | Move number at the sampled position |
| `position_eval`  | float  | Stockfish centipawn evaluation at this position |
| `phase`          | string | Game phase: `"opening"`, `"middlegame"`, or `"endgame"` |
| `game_result`    | object | Result of the game (`None` if ongoing/unknown) |

**Used for:** Earlier experiments / baseline analysis. Not directly used in the current RL training loop.

**Lichess version:** `chess_positions_lichess.parquet` — same structure, 25,431 rows from real human Lichess games instead of Stockfish self-play.

---

### `cot_training_examples_1k.parquet`

**What it is:** 1,000 supervised fine-tuning examples where each row has a complete model response in `<think>...</think><answer>...</answer>` format. These are hand-crafted or model-generated examples of good reasoning chains. Used for CoT (chain-of-thought) supervised training, not RL.

**Shape:** 1,000 rows × 7 columns

| Column            | Type   | Description |
|-------------------|--------|-------------|
| `pgn`             | string | Truncated PGN (context given to the model) |
| `fen`             | string | FEN at the position |
| `prompt`          | string | Full text prompt (same format as RL dataset) |
| `response`        | string | The expected model output with `<think>` and `<answer>` tags |
| `candidate_move`  | string | The move recommended in the `<answer>` tag |
| `format_reward`   | float  | 1.0 if the response has correct XML format, else 0.0 |
| `eval_diff`       | int    | Centipawn change for the candidate move (from Stockfish) |

**Full response example:**
```
<think>
In this position, White has developed their pieces actively and has castled, while Black
has also castled but has a slightly passive setup due to the pawn structure and piece
placement. The move 8. a3 by White is attacking the bishop on b4, forcing it to make
a decision. Black needs to decide whether to retreat the bishop, capture on c3, or move
it to another square...
</think>
<answer>
Bxc3
</answer>
```

**10-row version:** `cot_training_examples.parquet` — same structure, 10 rows (minimal test set).

**Used for:** Supervised fine-tuning (SFT) warm-start before RL, or for evaluating response format quality. Not used in the current RL run.

---

## Key Relationships

```
chess_rl_fen_pgn_prompt_10k  ←──── the model sees this (prompt + fen)
                │
                │  fen is used as key
                ▼
precomputed_move_evals_10k  ←──── reward function looks up (fen, model_move) here
                                   to get eval_diff → reward tier
```

The `fen` column is the bridge between the training dataset and the eval lookup table. Every position in `chess_rl_fen_pgn_prompt_10k` has corresponding entries in `precomputed_move_evals_10k` (one row per legal move at that position). If the model generates a move for a FEN that isn't in the eval table, the `good_move_reward_func` gives a "lookup miss" and skips the quality reward — but the legality reward still applies.
