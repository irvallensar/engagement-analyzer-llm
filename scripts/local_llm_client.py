from unsloth import FastLanguageModel
import torch
import sys
import re

# 1. GLOBAL LOAD (Runs once when script starts)
# This keeps the model in VRAM so we don't reload it for every sentence.
MODEL_NAME = "unsloth/Qwen3.5-9B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 4096 # Supports up to 128k, but 4k is safe for T4 VRAM
dtype = None # Auto detection
load_in_4bit = True # Force 4-bit (The Qiita method)

print(f"LOADING MODEL: {MODEL_NAME}...")

try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    print("SUCCESS: Qwen 3.5 9B Loaded on GPU.")
except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    sys.exit(1)

def call_local_llm(prompt):
    """
    Direct Python inference using Unsloth (Qwen 3.5).
    """
    # 2. Format Prompt (ChatML for Qwen)
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Output valid JSON only."},
        {"role": "user", "content": prompt}
    ]
    
    # Apply chat template
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True,
        return_tensors = "pt",
    ).to("cuda")

    # 3. Generate
    # Qwen 3.5 has "Thinking" enabled by default. We allow it to think,
    # then we extract the JSON from the result.
    outputs = model.generate(
        input_ids = inputs,
        max_new_tokens = 2048, # Give it room to think + answer
        use_cache = True,
        temperature = 0.0,      # Deterministic
    )
    
    # Decode
    response_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # 4. Cleaning: Remove the prompt and the "Thinking" block if present
    # Qwen 3.5 output often looks like: "user prompt... system... <think>...</think> JSON"
    # We just want the last part.
    
    # Simple strategy: The tokenizer usually returns the full conversation + answer.
    # We split by "assistant" header if present, or just grab the JSON.
    
    return response_text
