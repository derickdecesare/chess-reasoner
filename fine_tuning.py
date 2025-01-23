from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import torch
from datetime import datetime

def prepare_dataset(tokenizer):
    """
    Loads a dataset and formats each example into an instruction-based string, then tokenizes it.
    Returns the tokenized dataset for training.
    """
    
    print("Loading and preparing dataset...")
    dataset = load_dataset("yahma/alpaca-cleaned")  # Load a specific dataset from Hugging Face
    
    def format_instruction(example):
        """
        Format the instruction in a structure the model can learn from, including "Instruction", 
        "Input" (if present), and "Response".
        """
        if example["input"]:
            # If there's an input in addition to the instruction, we include it as well
            instruction = (
                f"### Instruction: {example['instruction']}\n"
                f"### Input: {example['input']}\n"
                f"### Response: {example['output']}"
            )
        else:
            # If there's no separate input, only include the instruction and the response
            instruction = (
                f"### Instruction: {example['instruction']}\n"
                f"### Response: {example['output']}"
            )
        return {"text": instruction}

    print("Formatting dataset...")
    # The dataset.map(...) applies the format_instruction function to every example
    formatted_dataset = dataset.map(format_instruction)
    
    # For quick experimentation, we take only the first 1000 examples from the 'train' split
    small_dataset = formatted_dataset['train'].select(range(1000))
    
    print("Tokenizing dataset...")
    def tokenize_function(examples):
        """
        Tokenizes the text in the examples. 
        We use truncation to ensure sequences fit into a max length, 
        and padding to make them uniform in size.
        """
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,  # Adjust as needed for longer sequences
            padding="max_length"
        )

    # Apply the tokenizer to every example and remove the original columns (like 'instruction', etc.)
    tokenized_dataset = small_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=small_dataset.column_names
    )
    
    print(f"Dataset prepared with {len(tokenized_dataset)} examples")
    return tokenized_dataset

def prepare_fine_tuning():
    """
    Main function to prepare and run fine-tuning:
    1. Loads model and tokenizer
    2. Prepares dataset
    3. Sets up training arguments
    4. Initializes the Trainer
    5. Trains the model
    6. Saves the final model
    """
    
    # Create a timestamp for the output directory for organization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./qwen_instruct_{timestamp}"
    
    print("Starting fine-tuning preparation...")
    
    # 1. Load model and tokenizer
    model_id = "Qwen/Qwen2.5-3B"  # The specific model name or path on Hugging Face
    print(f"Loading tokenizer from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Load the model in 4-bit precision for memory efficiency
    print(f"Loading model from {model_id} with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="auto",   # Automatically places model layers on the available GPU(s)/CPU
        load_in_4bit=True,   # Enables 4-bit quantization for lower memory usage
        torch_dtype=torch.float16  # Mixed-precision training with half-precision floats
    )

    # 2. Prepare dataset using the function above
    tokenized_dataset = prepare_dataset(tokenizer)

    # 3. Set up training arguments
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,            # Where to save model checkpoints and final model
        num_train_epochs=3,               # Number of epochs to train for
        per_device_train_batch_size=4,    # How many samples to process per device
        gradient_accumulation_steps=4,    # Accumulate gradients to effectively increase batch size
        save_steps=100,                   # When to save checkpoints (every 100 steps)
        save_total_limit=2,               # Keep only the last 2 checkpoints to save disk space
        logging_steps=10,                 # Log loss and metrics every 10 steps
        learning_rate=2e-5,               # The initial learning rate for the optimizer
        fp16=True,                        # Use automatic mixed precision (16-bit) for speed/memory
        warmup_steps=50,                  # Gradually increase the learning rate for the first 50 steps
        report_to="none",                 # Disable automated logging to WandB or other services
        logging_steps=10,        # See progress every 10 steps
        evaluation_strategy="steps",
        eval_steps=100,          # Evaluate every 100 steps
        load_best_model_at_end=True,
    )

    # 4. Create the Trainer object
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,                      # The model to train
        args=training_args,               # Training configurations
        train_dataset=tokenized_dataset,  # Our tokenized dataset to train on
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer, 
            mlm=False  # We set this to False because we are doing causal language modeling
        )
    )

    # 5. Train the model
    print("Starting training...")
    trainer.train()

    # 6. Save the final model to a separate folder
    final_output_dir = f"{output_dir}_final"
    print(f"Saving final model to {final_output_dir}")
    trainer.save_model(final_output_dir)
    print("Fine-tuning completed!")

# This ensures that the fine-tuning process runs only if this script is called as the main program.
if __name__ == "__main__":
    prepare_fine_tuning()