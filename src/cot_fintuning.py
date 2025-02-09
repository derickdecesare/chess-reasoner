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

def prepare_cot_dataset(tokenizer):
    """
    Edit the prompt --> standardize our request and then add only pgn
    Concatenate the Text: Merge the edited prompt and the response.
    Record Prompt Length: Tokenize the edited prompt separately to know how many tokens belong to it.
    Prepare Labels with Masking: Create a labels tensor that exactly matches your input IDs but set the loss for all prompt tokens to -100 (so that during training only the response tokens contribute to the loss).
    """
    print("Preparing cot dataset...")
    dataset = load_dataset("parquet", data_files="/data/cot_training_examples_1k.parquet") # set up for local path already inside src folder


    # If dataset is a DatasetDict (with a default "train" split), select that split first:
    if isinstance(dataset, dict):
        print("dataset is a DatasetDict")
        dataset = dataset[list(dataset.keys())[0]]

    # Print first 5 examples using slicing:
    print(dataset[:5])




    def format_cot_example(example):
        response = example["response"]
        pgn = example["pgn"]

        # Edit prompt
        prompt = f""" You are a chess grandmaster. Please analyze this chess position and provide your reasoning and next move.

    Current game (PGN):
    {pgn}

    Provide your analysis and move in the following format:

    <think>
    Your detailed reasoning, outlining key threats, piece positions, and any plans.
    </think>
    <answer>
    Your chosen move in standard algebraic notation (SAN)
    </answer>"""

        full_text = prompt + "\n" + response
        example["text"] = full_text
        # Tokenize the prompt only, to compute its length
        full_prompt = prompt + "\n"  # include the newline token explicitly
        prompt_tokens = tokenizer(full_prompt, add_special_tokens=False)["input_ids"]
        example["prompt_length"] = len(prompt_tokens)
        return example

    # Format the dataset with our cot example.
    dataset = dataset.map(format_cot_example)


    # Tokenize the dataset (we want to mask the prompt)
    print("Tokenizing the COT dataset...")
    def tokenize_and_mask(example):
        # Tokenize the full text prompt+\n+response
        tokenized = tokenizer(
            example["text"],
            truncation=True,
            max_length=2176,  # Max prompt was 829, max response was 1706
            padding="max_length"
        )
        input_ids = tokenized["input_ids"]
        labels = input_ids.copy()  # initialize labels to be the same as input

        # Mask out the prompt tokens so they do not contribute to the loss
        prompt_length = example.get("prompt_length", 0)
        for i in range(min(prompt_length, len(labels))):
            labels[i] = -100  # -100 is the default ignore index in PyTorch's CrossEntropyLoss
        tokenized["labels"] = labels
        return tokenized

    # Remove any extra columns from the dataset and tokenize
    tokenized_dataset = dataset.map(
        tokenize_and_mask,
        batched=False,
        remove_columns=dataset.column_names
    )

    print(f"Prepared tokenized dataset with {len(tokenized_dataset)} examples.")
    return tokenized_dataset





def prepare_cot_finetuning():
    """
    Sets up and runs fine-tuning on structured cot prompt response pairs.

    Key Points:
      - We provide -100 lables to the prompt so it is only calculating loss on the response

    The final model and checkpoints will be saved to your Google Drive.
    """

    base_dir = "/workspace/models/cot_fine_tune/llama_3b_instruct_pgn_cot"
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"run_{timestamp}")
    print(f"Output directory: {output_dir}")


    print("Loading instruct model and tokenizer for chess cot fine-tuning...")
    # Custom uploaded previous png fine tuned version of llaman

    # for using model from hub
    # model_id = "derickio/llama-3.2-1b-instruct-finetune_png_10k"

    # Use the locally fine-tuned model from our previous pgn training
    model_id = "/workspace/models/pgn_fine_tune/llama_3b_instruct_pgn/run_TIMESTAMP_final" # need to edit that once it exists
    
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)
    # ensure we add end token as padding
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=True,
        device_map="auto",   # Automatically allocate layers to available GPU(s) or CPU.
        # torch_dtype=torch.float16  # Mixed-precision training. # autocast auto handles these conversions
    )

    # Enable gradient checkpointing to reduce memory usage
    model.gradient_checkpointing_enable()
    # Disable the use of cached activations (not needed during training)
    model.config.use_cache = False

    # Prepare the tokenized cot dataset.
    tokenized_dataset = prepare_cot_dataset(tokenizer)

    # Split the tokenized dataset into train (90%) and eval (10%) subsets
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    print("Configuring training arguments...")
    # we might want to change some of the args since these were build for the pgn fine tune...
    # Target effective batch size = 16
    # 1k dataset * .9 == 900 dataset / 16 == 168

    training_args = TrainingArguments(
        output_dir=output_dir,            # Directory to store model checkpoints.
        num_train_epochs=3,               # Number of training epochs.
        per_device_train_batch_size=1,    # Batch size per device.
         per_device_eval_batch_size=1,    # so we don't blow up memory during eval
        gradient_accumulation_steps=16,    # To simulate a larger batch size.
        save_steps=50,                   # 168 total steps with 900 train dataset
        save_total_limit=2,               # Only save the last 2 checkpoints.
        logging_steps=10,                 # Log training metrics every 10 steps.
        learning_rate=1e-5,               # Lower learning rate for gentle fine-tuning.
        fp16=True,                        # Use 16-bit precision.
        warmup_steps=10,                  # Warmup steps. we are only doing 168 total steps
        report_to="none",                 # Disable external logging.
        evaluation_strategy="steps",
        eval_steps=10,                   # Evaluate every 56 steps.
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
            mlm=False  # Causal language modeling (next-token prediction).
        ),
        callbacks=[CustomLoggingCallback()]
    )

    print("Starting COT fine-tuning...")
    trainer.train()

    final_output_dir = f"{output_dir}_final"
    print(f"Saving final model to {final_output_dir}...")
    trainer.save_model(final_output_dir)
    print("COT fine-tuning completed!")




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
    prepare_cot_finetuning()

if __name__ == "__main__":
    main()