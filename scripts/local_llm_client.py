import torch
import json
import re
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# 1. LOAD PRE-COMPRESSED MODEL (Runs Once)
# ==========================================
# We add '-AWQ' to grab the 4-bit pre-compressed version of your model.
MODEL_ID = "Qwen/Qwen3.5-9B-AWQ"

print(f"Downloading Pre-Compressed Model: {MODEL_ID}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # Because it is already 4-bit, we just load it straight into the GPU.
    # No BitsAndBytes config needed!
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="cuda", # Force it strictly onto the GPU
    )
    print("SUCCESS: Qwen 3.5 9B (AWQ) loaded smoothly into VRAM!")
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
    messages = [
        {"role": "system", "content": "You are an expert linguistic annotator. Output ONLY valid JSON arrays. Do not include markdown formatting or explanations."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.0, 
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs["input_ids"], outputs)]
    response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response_text
