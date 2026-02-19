import json
import re

def parse_llm_json(text: str):
    # 1. Regex Extraction
    # Finds the part of the string starting with [ and ending with ]
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        raise ValueError("No JSON array found in LLM output")

    json_text = match.group(0)

    try:
        # 2. JSON Decoding
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON:\n{json_text}") from e

    # 3. Validation Loop
    valid_items = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        
        # FIX: Accept either 'id' or 'text'
        if "id" not in item and "text" not in item:
             print(f"Warning: Skipping item {i} (missing 'id' and 'text'): {item}")
             continue

        if "label" not in item:
            continue
            
        valid_items.append(item)

    return valid_items
