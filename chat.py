# Import required libraries
from transformers import AutoModelForCausalLM, AutoTokenizer  # HuggingFace's transformer library for model and tokenizer
from peft import PeftModel  # Parameter Efficient Fine-Tuning library for LoRA
import torch  # PyTorch deep learning library
from transformers import TextIteratorStreamer  # HuggingFace's streaming utility
from threading import Thread  # Python's built-in threading for concurrent operations

class ChatBot:
    def __init__(self, 
                 base_model_id="Qwen/Qwen2.5-3B",  # HuggingFace model ID
                 lora_path="models/qwen_instruct_20250123_041812_final"):  # Local path to LoRA weights
        # Initialize model and tokenizer
        print("Loading model and tokenizer...")
        # Load tokenizer from HuggingFace hub, trust_remote_code needed for Qwen's custom code
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            trust_remote_code=True,  # Allow Qwen's custom model code
            device_map="auto",       # Automatically choose best device (CPU/GPU)
            # load_in_4bit=True,     # Commented out 4-bit quantization
            torch_dtype=torch.float16  # Use 16-bit floating point for efficiency
        )
        
        # Apply our LoRA weights to the base model
        self.model = PeftModel.from_pretrained(base_model, lora_path)
        # Initialize empty list to store conversation history
        self.chat_history = []

    def stream_response(self, instruction, input_text=None):
        # Format the prompt with special tokens and instruction format
        if input_text:
            # Format for instruction + input
            prompt = (
                f"{self.tokenizer.bos_token}### Instruction: {instruction}\n"  # Begin with special token
                f"### Input: {input_text}\n"
                f"### Response:"
            )
        else:
            # Format for instruction only
            prompt = (
                f"{self.tokenizer.bos_token}### Instruction: {instruction}\n"
                f"### Response:"
            )

        # Convert text to token IDs and move to same device as model
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Setup streaming configuration
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_special_tokens=True,  # Don't show special tokens like <s>, </s>
            spaces_between_special_tokens=True  # Add spaces between tokens for readability
        )
        
        # Setup generation parameters
        generation_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=512,      # Maximum length of generated response
            temperature=0.7,         # Controls randomness (0=deterministic, 1=random)
            do_sample=True,          # Use sampling instead of greedy decoding
            pad_token_id=self.tokenizer.pad_token_id  # ID for padding token
        )

        # Create separate thread for generation to allow streaming
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()  # Start generation in background

        # Collect and clean response tokens as they're generated
        collected_response = []
        for text in streamer:  # Iterate over generated tokens
            # Clean up various artifacts
            clean_chunk = text.replace("<|endoftext|>", "").strip()
            if "### Response:" in clean_chunk:  # Remove response marker
                clean_chunk = clean_chunk.split("### Response:")[-1].strip()
            if "### Instruction:" in clean_chunk:  # Remove instruction marker
                clean_chunk = clean_chunk.split("### Instruction:")[0].strip()
            if "None" in clean_chunk:  # Remove None artifacts
                clean_chunk = clean_chunk.replace("None", "").strip()
                
            if clean_chunk:  # Only process non-empty chunks
                print(clean_chunk, end=" ", flush=True)  # Print immediately with space
                collected_response.append(clean_chunk + " ")  # Store with space
    
        # Join all chunks and clean up extra spaces
        full_response = "".join(collected_response).strip()
        # Add to conversation history
        self.chat_history.append((instruction, full_response))
        return full_response

    def chat(self):
        # Simple command-line chat interface
        print("Chat started! (Type 'quit' to exit)")
        while True:  # Infinite loop for conversation
            user_input = input("\nYou: ")  # Get user input
            if user_input.lower() == 'quit':  # Check for quit command
                break
                
            print("\nBot: ", end=" ")  # Print bot prefix
            response = self.stream_response(user_input)  # Get and stream response
            print("\n")  # Add spacing after response

# Only run if script is run directly (not imported)
if __name__ == "__main__":
    # Initialize chatbot with our fine-tuned model
    bot = ChatBot(lora_path="models/qwen_instruct_20250123_041812_final")
    bot.chat()  # Start chat interface