import os
# 1. CRITICAL: Fix PyTorch Memory Fragmentation before importing torch
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import json
import re
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ==========================================
# 2. LOAD MODEL GLOBALLY (Runs Once)
# ==========================================
MODEL_ID = "Qwen/Qwen3.5-9B"

print(f"Downloading and Loading {MODEL_ID} into VRAM (4-bit)...")
try:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # 3. CRITICAL: Memory Offloading Setup
    # We cap the GPU at 12GB to leave room for the attention fallback overhead.
    # The rest of the model layers will spill into the CPU RAM.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        max_memory={0: "12GB", "cpu": "10GB"}, 
        quantization_config=quantization_config,
    )
    print("SUCCESS: Qwen 3.5 9B loaded (GPU + CPU Spillover)!")
except Exception as e:
    print(f"CRITICAL ERROR LOADING MODEL: {e}")
    sys.exit(1)

# ==========================================
# 4. INFERENCE FUNCTION
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
