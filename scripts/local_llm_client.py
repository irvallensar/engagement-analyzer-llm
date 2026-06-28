import mlx_lm
import outlines
from outlines.inputs import Chat
from pydantic import BaseModel, Field
from typing import List

# 1. Define the Strict Pydantic Schema
class SpanExtraction(BaseModel):
    text: str = Field(description="Exact marker string from the sentence")
    label: str = Field(description="One of the 10 uppercase engagement labels")
    context_before: str = Field(description="1-3 words immediately preceding the marker (or empty string)")

class ExtractionResult(BaseModel):
    thought_process: str = Field(description="Logical reasoning for extracting the spans")
    spans: List[SpanExtraction] = Field(description="List of extracted engagement markers")

# Global variables
model = None

def call_local_llm(prompt_text):
    global model
    
    # 2. Load the model using the NEW Outlines API
    if model is None:
        print("\n[SYSTEM] Loading Qwen 2.5 (72B) into 256GB Mac Studio memory...")
        model_id = "mlx-community/Qwen2.5-72B-Instruct-4bit"
        
        # The new Outlines integration requires loading via mlx_lm first
        loaded_mlx = mlx_lm.load(model_id)
        model = outlines.from_mlxlm(*loaded_mlx)

    # 3. Format the chat prompt properly for Qwen 2.5's ChatML template
    chat_prompt = Chat([
        {"role": "system", "content": "You are an expert computational linguist."},
        {"role": "user", "content": prompt_text}
    ])

    # 4. Generate the response enforcing the Pydantic schema
    raw_result = model(chat_prompt, max_tokens=4096, output_type=ExtractionResult)
    
    # Outlines v0.1+ returns a JSON string, so we parse it back into our Pydantic object
    if isinstance(raw_result, str):
        result = ExtractionResult.model_validate_json(raw_result)
    else:
        result = raw_result
        
    return result
