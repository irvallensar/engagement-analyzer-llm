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
    
        "Examples:\n\n"
    
        "Input: However , this approach has significant flaws .\n"
        "Output: <COUNTER>However</COUNTER> , this approach has significant flaws .\n\n"
    
        "Input: I do not believe this approach works .\n"
        "Output: I do <DENY>not</DENY> <ENTERTAIN>believe</ENTERTAIN> this approach works .\n\n"
    
        "Input: The results might suggest a different interpretation .\n"
        "Output: The results <ENTERTAIN>might suggest</ENTERTAIN> a different interpretation .\n\n"
    
        "Input: This is clearly the most important factor .\n"
        "Output: This is <PROCLAIM>clearly</PROCLAIM> the most important factor .\n\n"
    
        "Input: Water boils at 100 degrees Celsius .\n"
        "Output: Water boils at 100 degrees Celsius .\n\n"
    
        "Input: According to Descartes , God was a supreme being .\n"
        "Output: <ATTRIBUTION>According to</ATTRIBUTION> <SOURCES>Descartes</SOURCES> , God was a supreme being .\n\n"
    
        "Input: Researchers have found that coverage of sport is biased .\n"
        "Output: <SOURCES>Researchers</SOURCES> have found that coverage of sport is biased .\n\n"
    
        "Input: As noted in Smith ( 2019 ) , the results were inconclusive .\n"
        "Output: <CITATION>As noted in Smith ( 2019 )</CITATION> , the results were inconclusive .\n\n"
    
        "Input: The results are shown in Table 3 .\n"
        "Output: The results are shown <ENDOPHORIC>in Table 3</ENDOPHORIC> .\n\n"
    
        "Input: Thus the evidence suggests that the policy failed .\n"
        "Output: <JUSTIFYING>Thus</JUSTIFYING> the evidence <ENTERTAIN>suggests</ENTERTAIN> that the policy failed .\n\n"
    
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
