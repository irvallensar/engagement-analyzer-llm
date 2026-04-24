import mlx_lm

print("Loading model and adapters into Unified Memory...")
model_id = "mlx-community/Qwen2.5-32B-Instruct-4bit"
model, tokenizer = mlx_lm.load(model_id, adapter_path="adapters")
print("Model loaded successfully! Ready for inference.")

def call_local_llm(sentence_text):
    system_prompt = (
        "You are an expert linguistic annotator. "
        "Rewrite the provided sentence and wrap all Engagement markers in XML tags corresponding to their category. "
        "The 10 valid tags are: <ATTRIBUTION>, <CITATION>, <COUNTER>, <DENY>, <ENDOPHORIC>, <ENTERTAIN>, <JUSTIFYING>, <MONOGLOSS>, <PROCLAIM>, <SOURCES>. "
        "Example Input: I do not believe this approach works. "
        "Example Output: I do <DENY>not</DENY> <ENTERTAIN>believe</ENTERTAIN> this approach works. "
        "If there are no markers, simply output the original sentence exactly as written."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
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
