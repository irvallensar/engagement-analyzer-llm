from mlx_lm import load
import outlines
from pydantic import BaseModel
from typing import List

# 1. Define the Strict Pydantic Schema
class MarkerSpan(BaseModel):
    text: str
    label: str
    context_before: str

class ExtractionResult(BaseModel):
    thought_process: str
    spans: List[MarkerSpan]

model = None
generator = None

def call_local_llm(prompt_text):
    global model, generator
    
    # 2. Load the Model & Outlines Generator
    if model is None or generator is None:
        print("\n[SYSTEM] Loading Qwen 3 (32B) into Mac Studio memory...")
        
        # Switched to the user-requested Qwen 3 32B MLX model
        model_id = "Qwen/Qwen3-32B-MLX-4bit"
        
        # Use the correct new Outlines MLX API
        mlx_model, tokenizer = load(model_id, tokenizer_config={"trust_remote_code": True})
        model = outlines.models.mlx(mlx_model, tokenizer)
        
        # Build the structured JSON generator based on our Pydantic schema
        generator = outlines.generate.json(model, ExtractionResult)

    # 3. Format the prompt for Qwen's ChatML
    messages = [
        {"role": "system", "content": "You are an expert computational linguist."},
        {"role": "user", "content": prompt_text}
    ]
    
    # Convert messages into the exact ChatML string Qwen expects
    formatted_prompt = model.tokenizer.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # 4. Generate the response with increased max_tokens to prevent EOF cutoffs
    # Because Outlines forces JSON, native <think> tags are blocked.
    # It will only reason inside the "thought_process" JSON string.
    raw_result = generator(formatted_prompt, max_tokens=4096)
    
    # 5. Parse the validated JSON back into a Python object
    try:
        result = ExtractionResult.model_validate_json(raw_result)
        
        # RETURN THE PYDANTIC OBJECT DIRECTLY
        return result

    except Exception as e:
        print(f"  [!] Pydantic Parse Error: {e}")
        return None
