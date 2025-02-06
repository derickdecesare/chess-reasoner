import pandas as pd

def main():
    dataset_path = "src/data/chess/cot_training_examples_1k.parquet"
    
    # Read the dataset saved in parquet format
    try:
        df = pd.read_parquet(dataset_path)
    except Exception as e:
        print(f"Error reading dataset from {dataset_path}: {e}")
        return

    # Display the first 5 records
    print("First 5 records:")
    print(df.head(), "\n")
    
    # Print the total number of training examples
    total_examples = len(df)
    print(f"Total number of training examples: {total_examples}\n")
    
    # Calculate and display statistics for PGN length
    df['pgn_length'] = df['pgn'].apply(len)
    print("PGN length statistics:")
    print(df['pgn_length'].describe(), "\n")
    
    # Calculate and display statistics for prompt length
    df['prompt_length'] = df['prompt'].apply(len)
    print("Prompt length statistics:")
    print(df['prompt_length'].describe(), "\n")
    
    # Calculate and display statistics for response length
    df['response_length'] = df['response'].apply(len)
    print("Response length statistics:")
    print(df['response_length'].describe(), "\n")
    
    # If 'format_reward' is numeric, display its draft statistics
    if pd.api.types.is_numeric_dtype(df['format_reward']):
        print("Format reward statistics:")
        print(df['format_reward'].describe(), "\n")
    
    # If 'eval_diff' is numeric, display its draft statistics
    if pd.api.types.is_numeric_dtype(df['eval_diff']):
        print("Evaluation difference statistics:")
        print(df['eval_diff'].describe(), "\n")
    
    # Display distribution of candidate moves
    print("Candidate move distribution:")
    print(df['candidate_move'].value_counts(), "\n")
    
    # (Optional) Overall descriptive statistics for all columns
    print("Overall DataFrame statistics:")
    print(df.describe(include='all'))

if __name__ == "__main__":
    main()