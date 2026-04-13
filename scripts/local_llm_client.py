from mlx_lm import load, generate

# We initialize these as None globally. 
# This ensures the massive model only loads into the Mac's RAM once, 
# rather than reloading for every single sentence.
model = None
tokenizer = None

def call_local_llm(prompt_text):
    global model, tokenizer
    
    # 1. Load the model (Only happens on the first sentence)
    if model is None or tokenizer is None:
        print("\n[SYSTEM] Loading MLX model into Mac Studio memory... (This takes a moment)")
        
        model_id = "mlx-community/Qwen2.5-32B-Instruct-bf16"
        model, tokenizer = load(model_id)

    # 2. Format the prompt for Qwen 2.5 Instruct
    # Qwen models expect a specific ChatML format (System message -> User message)
    messages = [
        {"role": "system", "content": "You are an expert computational linguist."},
        {"role": "user", "content": prompt_text}
    ]
    
    # The tokenizer automatically wraps your prompt in Qwen's required syntax
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # 3. Generate the response
    response = generate(
        model, 
        tokenizer, 
        prompt=formatted_prompt, 
        max_tokens=1000,   # High enough to allow for the <thought_process> block
        verbose=False      # Keeps your terminal output clean
    )
    
    return response
