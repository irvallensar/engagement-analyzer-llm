import outlines
from pydantic import BaseModel, Field
from typing import List, Literal

# 1. THE PYDANTIC SCHEMA
# This mathematically forces the LLM to output exactly this structure. 
# It cannot hallucinate outside of these constraints.
class Span(BaseModel):
    text: str = Field(description="Exact marker string from the sentence.")
    label: Literal[
        "MONOGLOSS", "DENY", "COUNTER", "PROCLAIM", "ENTERTAIN", 
        "ATTRIBUTION", "CITATION", "SOURCES", "ENDOPHORIC", "JUSTIFYING"
    ]
    context_before: str = Field(description="1-3 words immediately preceding the marker (or empty string).")

class ExtractionResult(BaseModel):
    # Putting the thought process inside the schema guarantees Chain-of-Thought
    # executes BEFORE the model attempts to extract the spans!
    thought_process: str = Field(description="Brief step-by-step logic for extracting the engagement markers.")
    spans: List[Span]

# Global variables
model = None
generator = None

def call_local_llm(prompt_text):
    global model, generator
    
    # 2. Load the Model & Outlines Generator
    if model is None or generator is None:
        print("\n[SYSTEM] Loading Qwen 2.5 model...")
        
        # Download model
        model_id = "mlx-community/Qwen2.5-72B-Instruct-4bit"
        
        # Load via outlines MLX integration
        model = outlines.models.mlxlm(model_id)
        
        # Compiles the Pydantic schema into a strict Regex state-machine
        generator = outlines.generate.json(model, ExtractionResult)

    # 3. Format the prompt using standard ChatML
    formatted_prompt = f"<|im_start|>system\nYou are an expert computational linguist.<|im_end|>\n<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
    
    # 4. Generate the structured response
    # Because of Outlines, 'result' is a pure Python Pydantic Object, NOT a raw string!
    result = generator(formatted_prompt)
    
    return result
