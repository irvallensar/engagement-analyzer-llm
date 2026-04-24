import mlx_lm

print("Loading model and adapters into Unified Memory...")
model_id = "mlx-community/Qwen2.5-32B-Instruct-4bit"
model, tokenizer = mlx_lm.load(model_id, adapter_path="adapters")
print("Model loaded successfully! Ready for inference.")

def call_local_llm(sentence_text):
    system_prompt = (
    "You are an expert linguistic annotator. "
    "Extract Engagement markers and output them as a JSON array. "
    "Each item must follow this format: "
    "[{\"label\": \"CATEGORY\", \"span\": \"target text\", \"context_before\": \"preceding text\"}]. "
    "If there are no Engagement markers, output []."
)
    messages = [ # must match training prompt
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Analyze this sentence:\n\n{sentence_text}"}
    ]
    
    formatted_prompt = tokenizer.apply_chat_template( # Converts messages into model input format
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    response = mlx_lm.generate(
        model, 
        tokenizer, 
        prompt=formatted_prompt, 
        max_tokens=1024,  # output length
        verbose=False # clean output
    )
    
    return response
