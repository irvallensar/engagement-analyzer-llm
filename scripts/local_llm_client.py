import outlines
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

# Global variables to hold the massive model in memory
model = None
generator = None

def call_local_llm(prompt_text):
    global model, generator
    
    # 2. Load the model and initialize the Outlines generator
    if model is None or generator is None:
        print("\n[SYSTEM] Loading Qwen 2.5 (72B) into 256GB Mac Studio memory...")
        
        # Load the 72B Instruct model via Outlines' MLX integration
        model_id = "mlx-community/Qwen2.5-72B-Instruct-4bit"
        
        # FIX: The correct outlines attribute is .mlx(), not .mlxlm()
        model = outlines.models.mlx(model_id)
        
        # Initialize the structured JSON generator using our Pydantic schema
        generator = outlines.generate.json(model, ExtractionResult)

    # 3. Generate the response
    # Outlines forces the LLM to output a perfect ExtractionResult Pydantic object
    result = generator(prompt_text, max_tokens=1024)
    
    return result
