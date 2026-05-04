def generate_response(messages):
    """
    Generic LLM caller for Experiment 2 (Data Augmentation).
    Accepts a custom list of message dictionaries (system and user prompts)
    rather than hardcoding the extraction prompt.
    """
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
    
    return response
