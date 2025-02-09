from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    TrainerCallback
)
import torch
from datetime import datetime
import os
import argparse

class CustomLoggingCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        last_log = state.log_history[-1] if state.log_history else {}
        print(f"\n>> Epoch {state.epoch:.2f} completed. Logged Metrics: {last_log}")
        return control

def prepare_pgn_dataset(tokenizer):
    """
    Loads a chess PGN dataset and tokenizes it.

    Expected:
      - The dataset contains examples with a key "pgn" for raw PGN moves.

    Returns the tokenized dataset.
    """
    print("Loading chess PGN dataset...")
    dataset = load_dataset("parquet", data_files="/data/chess_pgn_dataset_10k.parquet")  

    def format_pgn(example):
        # Remap the PGN string to the "text" field for tokenization.
        return {"text": example["pgn"]}

    # Format the dataset with our PGN string.
    dataset = dataset.map(format_pgn)

    # To be able to quickly edit size of dataset
    if "train" in dataset:
        print("train in dataset", len(dataset["train"]))
        small_dataset = dataset["train"].select(range(10000))
    else:
        print("no train in dataset", len(dataset))
        small_dataset = dataset.select(range(10000))

    # Print the first 5 examples
    for i in range(5):
        print(small_dataset[i])

    # Set pad token to eos_token
    tokenizer.pad_token = tokenizer.eos_token


    print("Tokenizing the PGN dataset...")
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,  # We keep the start of the PGN; anything beyond 512 tokens is truncated.
            padding="max_length"
        )

    tokenized_dataset = small_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=small_dataset.column_names
    )

    print(f"Prepared tokenized dataset with {len(tokenized_dataset)} examples.")
    return tokenized_dataset



def prepare_pgn_finetuning():
    """
    Sets up and runs PGN-based domain-adaptive fine-tuning.

    Key Points:
      - Uses raw PGN strings for next-token prediction (i.e. domain adaptation).
      - Uses a lower learning rate (1e-5) for gentle, full-model updates.
      - Designed to update the model's entire latent space (hence no PEFT/LoRA).

    The final model and checkpoints will be saved to your Google Drive.
    """
    base_dir = "/workspace/models/pgn_fine_tune/llama_3b_instruct_pgn"
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"run_{timestamp}")
    print(f"Output directory: {output_dir}")


    print("Loading instruct model and tokenizer for chess PGN fine-tuning...")
    # Use the instruct model that has shown better PGN prediction performance.
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="auto",   # Automatically allocate layers to available GPU(s) or CPU.
        # quantization_config=quant_config, # this messes everything up so avoid at all costs
        # torch_dtype=torch.float16  # Mixed-precision training. # autocast auto handles these conversions
    )

    # Prepare the tokenized PGN dataset.
    tokenized_dataset = prepare_pgn_dataset(tokenizer)

    # Split the tokenized dataset into train (90%) and eval (10%) subsets
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    print("Configuring training arguments...")
    # Steps --> 9,000 train dataset / effective batch size 4 * 4 = 16 = 562 steps per epoch --> 3 epochs = 1686 steps
    training_args = TrainingArguments(
        output_dir=output_dir,            # Directory to store model checkpoints.
        num_train_epochs=3,               # Number of training epochs.
        per_device_train_batch_size=4,    # Batch size per device. --> may need to reduce this for memory constraints depending...
        gradient_accumulation_steps=4,    # To simulate a larger batch size.
        save_steps=600,                   # Save checkpoints every x steps.
        save_total_limit=2,               # Only save the last 2 checkpoints.
        logging_steps=10,                 # Log training metrics every 10 steps.
        learning_rate=1e-5,               # Lower learning rate for gentle fine-tuning.
        fp16=True,                        # Use 16-bit precision.
        warmup_steps=50,                  # Warmup steps.
        report_to="wandb",                 # Disable external logging.
        evaluation_strategy="steps",
        eval_steps=100,                   # Evaluate every 100 steps.
        weight_decay=0.01,                # L2 regularization to prevent overfit
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False 
        ),
        callbacks=[CustomLoggingCallback()]
    )

    print("Starting PGN fine-tuning...")
    trainer.train()

    final_output_dir = f"{output_dir}_final"
    print(f"Saving final model to {final_output_dir}...")
    trainer.save_model(final_output_dir)
    print("PGN fine-tuning completed!")



def main():
    # Todo: for when we want to run this script from the command line
    # parser = argparse.ArgumentParser(description='Fine-tune LLM on chess PGN data')
    # parser.add_argument('--data_path', type=str, required=True,
    #                   help='Path to the parquet dataset')
    # parser.add_argument('--output_dir', type=str, required=True,
    #                   help='Base directory for output files')
    # parser.add_argument('--num_epochs', type=int, default=5,
    #                   help='Number of training epochs')
    # parser.add_argument('--batch_size', type=int, default=4,
    #                   help='Training batch size per device')
    # parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
    #                   help='Number of gradient accumulation steps')
    # parser.add_argument('--learning_rate', type=float, default=1e-5,
    #                   help='Learning rate')

    # args = parser.parse_args()
    prepare_pgn_finetuning()

if __name__ == "__main__":
    main()