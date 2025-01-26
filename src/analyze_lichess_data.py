import pandas as pd
import matplotlib.pyplot as plt
from typing import List
import chess.pgn
import io

def plot_distributions(df: pd.DataFrame):
    """Create multiple plots to visualize the dataset"""
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. ELO Distribution
    plt.subplot(2, 2, 1)
    plt.hist(df['elo'], bins=30)
    plt.title('Distribution of ELO Ratings')
    plt.xlabel('ELO')
    plt.ylabel('Count')
    
    # 2. Game Phase Distribution
    plt.subplot(2, 2, 2)
    df['phase'].value_counts().plot(kind='bar')
    plt.title('Distribution of Game Phases')
    plt.xlabel('Phase')
    plt.ylabel('Count')
    
    # 3. Move Numbers Distribution
    plt.subplot(2, 2, 3)
    plt.hist(df['move_number'], bins=30)
    plt.title('Distribution of Move Numbers')
    plt.xlabel('Move Number')
    plt.ylabel('Count')
    
    # 4. Position Evaluations Distribution
    plt.subplot(2, 2, 4)
    # Clip extreme values for better visualization
    evaluations = df['position_eval'].clip(-5, 5)
    plt.hist(evaluations, bins=30)
    plt.title('Distribution of Position Evaluations')
    plt.xlabel('Evaluation (pawns)')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('src/data/chess/lichess_analysis.png')
    plt.close()

def print_sample_positions(df: pd.DataFrame, n: int = 5):
    """Print details for a sample of positions"""
    print("\nSample Positions:")
    for i, row in df.sample(n).iterrows():
        print(f"\nPosition {i+1}:")
        print(f"ELO: {row['elo']}")
        print(f"Phase: {row['phase']}")
        print(f"Move Number: {row['move_number']}")
        print(f"Evaluation: {row['position_eval']:.2f}")
        print("PGN:")
        # Print just the moves, not the headers
        moves = row['pgn'].split('\n\n')[1] if '\n\n' in row['pgn'] else row['pgn']
        print(moves)

def print_statistics(df: pd.DataFrame):
    """Print various statistics about the dataset"""
    print("\nDataset Statistics:")
    print(f"Total positions: {len(df)}")
    
    print("\nELO Statistics:")
    print(df['elo'].describe())
    
    print("\nPositions per phase:")
    print(df['phase'].value_counts())
    
    print("\nMove number statistics:")
    print(df['move_number'].describe())
    
    print("\nEvaluation statistics:")
    print(df['position_eval'].describe())
    
    # Count unique games
    unique_games = df['pgn'].str.split('\n\n').str[0].nunique()
    print(f"\nUnique games in dataset: {unique_games}")

def main():
    # Read the dataset
    df = pd.read_parquet('src/data/chess/chess_positions_lichess.parquet')
    
    # Generate visualizations
    plot_distributions(df)
    
    # Print statistics
    print_statistics(df)
    
    # Show sample positions
    print_sample_positions(df)

if __name__ == "__main__":
    main()