from rl_training_DEPRECATED import ChessRLTrainer

def main():
    # Initialize trainer
    trainer = ChessRLTrainer(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",  # or your preferred model
        use_wandb=False,  # Set to False if you don't want to use wandb
        samples_per_position=3,  # Start with a smaller number for testing
        num_positions=4  # Start with a smaller number for testing
    )

    # Train for a specified number of steps
    num_steps = 5  # Start small for testing
    try:
        losses = trainer.train(num_steps)
        print(f"Training completed with final loss: {losses[-1]}")
    except Exception as e:
        print(f"Training failed with error: {e}")
    finally:
        # Clean up
        trainer.engine.quit()  # Close Stockfish engine

if __name__ == "__main__":
    main()




# loss_scale (should fluctuate between 2^16 and 2^24)
# gradient_norm (should stay < 1000)
# advantages.mean() (should center around 0)