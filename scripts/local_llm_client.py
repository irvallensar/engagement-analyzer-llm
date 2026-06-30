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
        print("\n[SYSTEM] Loading Qwen 3 (32B) MLX Model into memory...")
        model_id = "Qwen/Qwen3-32B-MLX-4bit"
        
        # Using exactly the syntax you confirmed works.
        # This natively leverages MLX framework for Apple Silicon optimization.
        model = outlines.models.mlxlm(model_id)
        
        # Build the structured JSON generator based on our Pydantic schema
        generator = outlines.generate.json(model, ExtractionResult)

    # 3. Format the prompt for Qwen's ChatML
    messages = [
        {"role": "system", "content": "You are an expert computational linguist."},
        {"role": "user", "content": prompt_text}
    ]
    
    try:
        formatted_prompt = model.tokenizer.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except AttributeError:
        # Fallback if tokenizer lacks apply_chat_template
        formatted_prompt = f"<|im_start|>system\nYou are an expert computational linguist.<|im_end|>\n<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
    
    # 4. Generate the response 
    try:
        # Outlines uses highly optimized MLX sampling under the hood here
        raw_result = generator(formatted_prompt, max_tokens=4096)
        
        # THE FIX: Modern Outlines returns the Pydantic object directly. 
        # If we try to parse it with model_validate_json again, it crashes silently!
        if isinstance(raw_result, ExtractionResult):
            return raw_result
        elif isinstance(raw_result, str):
            return ExtractionResult.model_validate_json(raw_result)
        else:
            print(f"  [!] Unexpected type from outlines: {type(raw_result)}")
            return ExtractionResult(thought_process="Error", spans=[])

    except Exception as e:
        print(f"  [!] LLM Crash or Parsing Error: {e}")
        # Returning an empty ExtractionResult prevents evaluate_iob.py from crashing
        return ExtractionResult(thought_process="Error", spans=[])
