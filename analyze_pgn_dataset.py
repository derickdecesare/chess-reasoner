import os
import pandas as pd

def main():
    # Compute the absolute path to the dataset (relative to src/)
    dataset_rel_path = os.path.join("src","data", "chess", "chess_pgn_dataset_100k.parquet")
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), dataset_rel_path))
    print(f"Loading dataset from: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print("ERROR: Dataset file not found!")
        return

    # Load the dataset as a DataFrame
    try:
        df = pd.read_parquet(dataset_path)
        print("Dataset loaded successfully!")
    except Exception as e:
        print("ERROR: Failed to load the dataset:")
        print(e)
        return

    # Print basic dataset information
    print(f"Dataset shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    print("\nDataset info:")
    print(df.info())
    
    # Show the first 5 rows of the dataset
    print("\nFirst 5 rows:")
    print(df.head())

    # If the dataset contains a 'pgn' column, analyze it further
    if 'pgn' in df.columns:
        print("\nPreview of 'pgn' column entries:")
        for i, row in df.head(3).iterrows():
            # Print a preview entire PGN string
            print(f"Row {i}: {row['pgn']}")
        
        # Calculate and print statistics about the lengths of the PGN strings
        pgn_lengths = df['pgn'].apply(lambda x: len(x.strip()))
        print("\nStatistics for 'pgn' string lengths:")
        print(pgn_lengths.describe())
    else:
        print("\nWARNING: No 'pgn' column found in the dataset.")
        
    # Process and analyze the 'elo' column if it exists
    if 'elo' in df.columns:
        try:
            # Convert the elo column to numeric in case it isn't already.
            elo_series = pd.to_numeric(df['elo'], errors='coerce')
            print("\nStatistics for 'elo' column:")
            print(f"Average Elo: {elo_series.mean():.2f}")
            print(f"Standard Deviation: {elo_series.std():.2f}")
            print(f"Minimum Elo: {elo_series.min()}")
            print(f"Maximum Elo: {elo_series.max()}")
            print("\nFull Elo description:")
            print(elo_series.describe())
        except Exception as e:
            print("ERROR: Failed to process 'elo' column:")
            print(e)
    else:
        print("\nWARNING: No 'elo' column found in the dataset.")

if __name__ == "__main__":
    main() 