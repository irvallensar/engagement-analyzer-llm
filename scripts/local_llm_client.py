import torch
import json
import re
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ==========================================
# 1. LOAD MODEL GLOBALLY (Runs Once)
# ==========================================
# Target the Qwen 3.5 9B model directly from Hugging Face
MODEL_ID = "Qwen/Qwen3.5-9B-Instruct"

print(f"Downloading and Loading {MODEL_ID} into VRAM (4-bit)...")
try:
    # Force 4-bit quantization so the 9B model fits perfectly on the 15GB T4 GPU
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto", # Automatically maps layers to the T4 GPU
        quantization_config=quantization_config,
    )
    print("SUCCESS: Qwen 3.5 9B loaded natively on GPU!")
except Exception as e:
    print(f"CRITICAL ERROR LOADING MODEL: {e}")
    sys.exit(1)

# ==========================================
# 2. INFERENCE FUNCTION
# ==========================================
def call_local_llm(prompt):
    """
    Direct Python inference using Hugging Face Transformers.
    """
    # Format the prompt using Qwen's specific Chat Template
    messages = [
        {"role": "system", "content": "You are an expert linguistic annotator. Output ONLY valid JSON arrays. Do not include markdown formatting or explanations."},
        {"role": "user", "content": prompt}
    ]
    
    # Prepare inputs for the GPU
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    
    # Generate the response
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.0, # Deterministic for scientific consistency
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode only the newly generated text (ignore the prompt)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)]
    response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response_text    
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
