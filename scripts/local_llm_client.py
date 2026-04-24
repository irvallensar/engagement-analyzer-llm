import mlx_lm
import json
import re

print("Loading model and adapters into Unified Memory...")
model_id = "mlx-community/Qwen2.5-32B-Instruct-4bit"
model, tokenizer = mlx_lm.load(model_id, adapter_path="adapters")
print("Model loaded successfully! Ready for inference.")

VALID_LABELS = {'ATTRIBUTION', 'CITATION', 'COUNTER', 'DENY', 'ENDOPHORIC', 'ENTERTAIN', 'JUSTIFYING', 'MONOGLOSS', 'PROCLAIM', 'SOURCES'}

def xml_to_json_spans(xml_text):
    """
    Validation Layer: Catches invalid XML, auto-fixes mismatched closing tags,
    and converts to the JSON structure expected by the evaluator.
    """
    # Regex catches <OPEN_TAG>content</CLOSE_TAG> even if they don't match
    pattern = re.compile(r'<([A-Z_]+)>(.*?)</([A-Z_]+)>')
    spans = []
    
    for match in pattern.finditer(xml_text):
        open_tag = match.group(1)
        content = match.group(2).strip()
        close_tag = match.group(3)
        
        # AUTO-FIX LOGIC: If tags are mismatched (e.g. <DENY>not</PROCLAIM>), 
        # we trust the opening tag as long as it is a valid label.
        if open_tag in VALID_LABELS:
            spans.append({
                "label": open_tag,
                "span": content
            })
            
    return json.dumps(spans)

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
    
    # Pass the LLM's raw XML response through your validation layer
    return xml_to_json_spans(response)
