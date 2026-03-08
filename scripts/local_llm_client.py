import sys
from llama_cpp import Llama

# 1. GLOBAL MODEL LOADER
# We load this ONCE at the top level so it stays in GPU memory.
# If we put this inside the function, it would reload the 10GB file for every sentence (super slow).

MODEL_PATH = "./models/Qwen3.5-9B-Q8_0.gguf" 

print(f"Loading Qwen 3.5 from {MODEL_PATH}...")

try:
    # Initialize the model with GPU offloading
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,      # -1 = Offload ALL layers to GPU
        n_ctx=4096,           # 4k Context Window (Fits easily in 15GB VRAM)
        verbose=False         # Set True if you want to see layer-by-layer load stats
    )
    print("SUCCESS: Model loaded on Tesla T4 GPU.")
except Exception as e:
    print(f"CRITICAL ERROR LOADING MODEL: {e}")
    sys.exit(1)

def get_completion(prompt):
    """
    Direct Python call to the GGUF model via llama-cpp-python
    """
    # Qwen 3.5 uses ChatML format. We must wrap the prompt correctly.
    formatted_prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    # Generate response
    output = llm(
        formatted_prompt,
        max_tokens=1024,   # Space for the JSON response
        stop=["<|im_end|>"], # Stop generating when finished
        echo=False,        # Do not repeat the prompt
        temperature=0.0    # Deterministic (Scientific)
    )
    
    # Extract just the text
    return output['choices'][0]['text']
