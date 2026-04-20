import mlx_lm

print("Loading model and adapters into Unified Memory...")
model_id = "mlx-community/Qwen2.5-32B-Instruct-4bit"
model, tokenizer = mlx_lm.load(model_id, adapter_path="adapters")
print("Model loaded successfully! Ready for inference.")

def call_local_llm(sentence_text):
    system_prompt = (
        "You are an expert linguistic annotator specializing in Engagement analysis (Appraisal Theory). "
        "Extract Engagement markers using ONLY these 10 labels: "
        "ATTRIBUTION, CITATION, COUNTER, DENY, ENDOPHORIC, ENTERTAIN, JUSTIFYING, MONOGLOSS, PROCLAIM, SOURCES. "
        "\n\nLABEL DEFINITIONS:"
        "\n- ENTERTAIN: hedges, epistemic uncertainty (e.g. 'might', 'perhaps', 'seems', 'I think')"
        "\n- ATTRIBUTION: attributing a position to an external voice (e.g. 'X argues that', 'according to X')"
        "\n- CITATION: direct reference to a specific source or work"
        "\n- COUNTER: concessive or counter-expectational (e.g. 'although', 'however', 'while', 'despite')"
        "\n- DENY: explicit negation of a position (e.g. 'this is not', 'contrary to', 'fails to')"
        "\n- ENDOPHORIC: reference to another part of the same text (e.g. 'as shown above', 'see Figure 3')"
        "\n- JUSTIFYING: providing evidence or reasoning (e.g. 'because', 'given that', 'since', 'therefore')"
        "\n- MONOGLOSS: bare assertion with no dialogic acknowledgment"
        "\n- PROCLAIM: emphatic assertion (e.g. 'clearly', 'obviously', 'of course', 'undeniably')"
        "\n- SOURCES: reference to a data source or corpus (e.g. 'the data shows', 'our corpus reveals')"
        "\n\nOutput format: JSON array only, no other text."
        "\n[{\"label\": \"CATEGORY\", \"span\": \"exact text\", \"context_before\": \"preceding 3 words\"}]"
        "\nIf no markers: []"
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
    
    return response
