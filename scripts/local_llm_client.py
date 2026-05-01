import mlx_lm

print("Loading model and adapters into Unified Memory...")
model_id = "mlx-community/Qwen2.5-32B-Instruct-bf16"
model, tokenizer = mlx_lm.load(model_id, adapter_path="adapters")
print("Model loaded successfully! Ready for inference.")

def call_local_llm(sentence_text):
    SYSTEM_PROMPT = (
        "You are an expert linguistic annotator. Analyze the sentence and extract all Engagement markers. "
        "Output a JSON array of dictionaries with 'label' and 'span' keys. "
        "The 10 valid tags are: ATTRIBUTION, CITATION, COUNTER, DENY, ENDOPHORIC, ENTERTAIN, JUSTIFYING, MONOGLOSS, PROCLAIM, SOURCES. "
        "Example Input: I do not believe this approach works. "
        "Example Output: [{\"label\": \"DENY\", \"span\": \"not\"}, {\"label\": \"ENTERTAIN\", \"span\": \"believe\"}] "
        "If there are no markers, output []."
    )
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Analyze this sentence:\n\n{sentence_text}"}
    ]
    
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    response = mlx_lm.generate(
        model, 
        tokenizer, 
        prompt=formatted_prompt, 
        max_tokens=1024, 
        verbose=False
    )
    
    return response # Return raw XML string to the evaluator
