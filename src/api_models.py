import os
from openai import OpenAI
from anthropic import Anthropic
import re
from typing import Generator, Optional
from dotenv import load_dotenv
from src.utils.formatting_utils import is_plausible_san

load_dotenv()

# Initialize API clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def generate_move_api(
    model_name: str,
    prompt: str,
    max_attempts: int = 5,
    base_temperature: float = 0.2,
    temperature_increment: float = 0.2
) -> Generator[str, None, None]:
    """
    Generate chess moves using API models (OpenAI GPT-4 or Anthropic Claude)
    with multiple attempts and temperature increments. Stops after finding first valid move.
    """
    
    all_moves_found = False
    
    for attempt in range(max_attempts):
        # Calculate temperature and clamp between 0 and 1 for Claude
        raw_temperature = base_temperature + (temperature_increment * attempt)
        temperature = min(1.0, max(0.0, raw_temperature))
        print(f"[API] Attempt {attempt+1}/{max_attempts}, temperature={temperature}")

        # Call the appropriate API:
        if "gpt-4" in model_name.lower() or "o3" in model_name.lower() or "o1" in model_name.lower():
            if "o3" in model_name.lower() or "o1" in model_name.lower():
                response = openai_client.chat.completions.create(
                    model=model_name,
                    # reasoning_effort="medium", # automatically defaults to this..
                    messages=[{"role": "user", "content": prompt}],
                    # temperature=temperature,  # OpenAI can handle higher temperatures
                )
            else:
                response = openai_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,  # OpenAI can handle higher temperatures
                    max_tokens=10,
                )
            move_text = response.choices[0].message.content.strip()
        elif "claude" in model_name.lower():
            response = anthropic_client.messages.create(
                model=model_name,
                max_tokens=10,
                temperature=temperature,  # Claude needs temperature between 0 and 1
                messages=[{"role": "user", "content": prompt}]
            )
            if isinstance(response.content, list) and len(response.content) > 0:
                move_text = response.content[0].text.strip()
            else:
                move_text = str(response.content).strip()
        else:
            raise ValueError(f"Unsupported API model: {model_name}")


        new_text_no_numbers = re.sub(r'\d+\.\s*', '', move_text) # remove move numbers
        # Split the response to find possible tokens
        tokens = new_text_no_numbers.split()
        # print(f"Tokens: {tokens}")
        move_count = 0

        for token in tokens:
            clean_move = token.strip().rstrip('+#')
            if len(clean_move) >= 2 and is_plausible_san(clean_move):
                print(f"Yielding: {clean_move}")
                yield clean_move
                move_count += 1
            else:
                print(f"Token not plausible, skipping: {clean_move}")
                continue

        # if move_count > 0:
        #     all_moves_found = True
        #     break

    if not all_moves_found:
        print("[API] No valid tokens found in all attempts; yielding ILLEGAL_MOVE.")
        yield "ILLEGAL_MOVE"

    # The ILLEGAL_MOVE will be yielded by baseline_eval.py if needed 