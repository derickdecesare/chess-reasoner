import pandas as pd
import matplotlib.pyplot as plt

# Read the parquet file
# Parquet is a columnar storage format that's more efficient than CSV
# It preserves data types and compresses well
df = pd.read_parquet('src/data/chess/chess_positions.parquet')

# Print basic information about the dataset
print("\nDataset Overview:")
print(f"Total positions: {len(df)}")
print("\nColumns in dataset:", df.columns.tolist())

# Show distribution of positions across ELO levels
print("\nPositions per ELO level:")
print(df['elo'].value_counts().sort_index())

# Show distribution across game phases
print("\nPositions per game phase:")
print(df['phase'].value_counts())

# Show distribution of final move numbers to check for 50-move rule clustering
print("\nFinal move numbers in games:")
# Group by game (using PGN) and get the max move number for each game
final_moves = df.groupby(df['pgn'].str.split('\n\n').str[0])['move_number'].max()
print(final_moves.value_counts().sort_index())

# Look at first few positions from a single game to see progression
print("\nMove progression in first game:")
first_game_pgn = df['pgn'].iloc[0].split('\n\n')[0]  # Get PGN header of first game
game_positions = df[df['pgn'].str.startswith(first_game_pgn)].sort_values('move_number')
# Drop duplicates to ensure we only see each move once
game_positions = game_positions.drop_duplicates(subset=['move_number'])

for _, pos in game_positions.head(15).iterrows():
    print(f"\nMove {pos['move_number']}:")
    print(f"Phase: {pos['phase']}")
    print(f"Evaluation: {pos['position_eval']:.2f}")
    # Print just the moves, not the headers
    moves = pos['pgn'].split('\n\n')[1] if '\n\n' in pos['pgn'] else ''
    print(moves)

# Create histogram of final move numbers
plt.figure(figsize=(10, 6))
plt.hist(final_moves, bins=30)
plt.title('Distribution of Game Lengths')
plt.xlabel('Final Move Number')
plt.ylabel('Number of Games')
plt.savefig('src/data/chess/game_lengths.png')
plt.close()

# Look at some example positions
print("\nExample position:")
# Display first position's PGN and evaluation
example = df.iloc[0]
print(f"ELO: {example['elo']}")
print(f"Phase: {example['phase']}")
print(f"Move number: {example['move_number']}")
print(f"Evaluation: {example['position_eval']}")
print("\nPGN:")
print(example['pgn'])

# Optional: Create a simple plot of position evaluations
plt.figure(figsize=(10, 6))
plt.hist(df['position_eval'], bins=50)
plt.title('Distribution of Position Evaluations')
plt.xlabel('Evaluation (pawns)')
plt.ylabel('Count')
plt.savefig('src/data/chess/evaluation_distribution.png')
plt.close()

