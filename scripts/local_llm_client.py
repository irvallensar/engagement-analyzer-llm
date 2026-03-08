from llama_cpp import Llama
import os

# Global variable to hold the model in memory
# This ensures we don't reload the 10GB model for every single sentence!
MODEL_PATH = "./models/qwen2.5-9b-instruct-q8_0.gguf" 
# ^^^ MAKE SURE THIS MATCHES THE FILENAME YOU DOWNLOADED ^^^

print(f"Loading LLM from {MODEL_PATH}...")

try:
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,      # -1 means "Offload EVERYTHING to GPU"
        n_ctx=4096,           # Context window (fits in 15GB VRAM with Q8 model)
        verbose=False         # Set to True if you want to see the speed stats
    )
    print("LLM Loaded successfully on GPU!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise e

def get_completion(prompt):
    """
    Direct Python call to the GGUF model via llama-cpp-python
    """
    # Formatting the prompt for Qwen Instruct (ChatML format is safest)
    # Most GGUF models handle raw text fine, but this helps stability
    formatted_prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    output = llm(
        formatted_prompt,
        max_tokens=1024,  # Give it room to write the JSON
        stop=["<|im_end|>", "User:", "Observation:"], # Stop tokens
        echo=False,
        temperature=0.0   # Deterministic output
    )
    
    # Extract just the text response
    return output['choices'][0]['text']
