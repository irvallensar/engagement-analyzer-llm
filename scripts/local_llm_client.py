import mlx_lm

# ==========================================
# 1. GLOBAL SPACE (Runs ONLY ONCE at startup)
# ==========================================
print("Loading model and adapters into Unified Memory...")
model_id = "mlx-community/Qwen2.5-32B-Instruct-4bit"

# This loads the 32B base and your fine-tuned weights simultaneously
model, tokenizer = mlx_lm.load(model_id, adapter_path="adapters")
print("Model loaded successfully! Ready for inference.")


# ==========================================
# 2. FUNCTION SPACE (Runs 1,700 times)
# ==========================================
def call_local_llm(sentence_text):
    # Build the exact Chat dictionary
    messages = [
        {"role": "system", "content": "You are an expert annotator. Extract Engagement markers as a JSON array."},
        {"role": "user", "content": sentence_text}
    ]
    
    # Inject the special <|im_start|> tokens
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Generate the response using the ALREADY LOADED model
    response = mlx_lm.generate(
        model, 
        tokenizer, 
        prompt=formatted_prompt, 
        max_tokens=500, 
        verbose=False
    )
    
    return response
