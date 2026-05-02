from mlx_lm import load, generate

# 1. Load the model globally so the function can see it.
# IMPORTANT: Change this path to the exact MLX model path you are using!
MODEL_PATH = "mlx-community/Qwen2.5-32B-Instruct-4bit" 

print(f"Loading MLX model from {MODEL_PATH}...")
model, tokenizer = load(MODEL_PATH)
print("Model loaded successfully! Ready for inference.")

# 2. The generation function
def generate_response(messages):
    """
    Accepts a list of message dictionaries and generates a response.
    """
    # Apply Qwen's chat template
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Generate the text
    response = generate(
        model, 
        tokenizer, 
        prompt=formatted_prompt, 
        max_tokens=1024, 
        verbose=False
    )
    
    return response
