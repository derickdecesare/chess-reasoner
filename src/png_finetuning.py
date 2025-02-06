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

def prepare_pgn_dataset(tokenizer):
    """
    Loads a chess PGN dataset and tokenizes it.
    
    Expected:
      - The dataset (replace "your_chess_pgn_dataset" with your actual identifier)
      - Each example should have a key 'pgn' that contains the raw PGN moves.
    
    Returns the tokenized dataset.
    """
    print("Loading chess PGN dataset...")
    dataset = load_dataset("your_chess_pgn_dataset")  # Replace with your actual PGN dataset identifier

    def format_pgn(example):
        # Make sure the PGN string is in the 'text' field for tokenization.
        return {"text": example["pgn"]}
    
    # Map to standardize the dataset
    dataset = dataset.map(format_pgn)
    
    # For quick experimentation, select a subset (adjust the range as needed)
    if "train" in dataset:
        small_dataset = dataset["train"].select(range(1000))
    else:
        small_dataset = dataset.select(range(1000))
    
    print("Tokenizing the PGN dataset...")
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,  # Adjust depending on PGN length requirements
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
    Prepares and runs the domain-adaptive (chess-specific) fine-tuning using PGN next-token prediction.
    
    Key Points:
      - Uses raw PGN strings for next-token prediction (as opposed to structured instruction tuning).
      - Employs a lower learning rate (e.g., 1e-5) for gentle, full-model updates.
      - Avoids parameter-efficient methods like PEFT/LoRA since we wish to adjust the entire latent space.
      
    This setup will help enhance the model's understanding of chess while still retaining instruction-following capabilities.
    """
    
    # Create an output directory with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./instruct_model_pgn_finetuning_{timestamp}"
    
    print("Loading instruct model and tokenizer for chess PGN tuning...")
    # Using an instruct model that previously showed better PGN next-move prediction performance.
    model_id = "meta-llama/Llama-3.2-3B-Instruct"  # could also consider meta-llama/Llama-3.2-1B-Instruct if we want to save memory/compute
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="auto",   # Automatically allocates the model to available GPU(s)/CPU
        load_in_4bit=True,   # 4-bit quantization for memory efficiency
        torch_dtype=torch.float16  # Mixed precision for faster training.
    )
    
    # Prepare the tokenized PGN dataset
    tokenized_dataset = prepare_pgn_dataset(tokenizer)
    
    print("Setting up training arguments for PGN domain-adaptive pre-training...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        save_steps=100,
        save_total_limit=2,
        logging_steps=10,
        learning_rate=1e-5,  # Lower learning rate to ensure gentle updates
        fp16=True,
        warmup_steps=50,
        report_to="none",  # Disable additional logging (e.g., WandB)
        evaluation_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer, 
            mlm=False  # This is a causal language modeling task (next-token prediction)
        )
    )
    
    print("Starting PGN fine-tuning (domain-adaptive pre-training on chess PGNs)...")
    trainer.train()
    
    final_output_dir = f"{output_dir}_final"
    print(f"Saving the fine-tuned model to {final_output_dir}...")
    trainer.save_model(final_output_dir)
    print("Chess PGN fine-tuning completed!")

if __name__ == "__main__":
    prepare_pgn_finetuning()