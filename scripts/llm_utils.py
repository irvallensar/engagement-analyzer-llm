import json
import re

def parse_llm_json(text: str):
    # 1. Regex Extraction
    # Finds the part of the string starting with [ and ending with ]
    match = re.search(r"\[[\s\S]*\]", text)    #the first thing that looks like a JSON array # anything between [ and ]
    # re.search = a match object (if found); None (if not found)

    # r"\[[\s\S]*\]" means Find:
    # [
    # then anything
    # then ]
    if not match:
        raise ValueError("No JSON array found in LLM output")

    json_text = match.group(0) #it finds a clean string, "[ {...} ]"

    try:
        # 2. JSON Decoding
        data = json.loads(json_text) #Converts JSON string → Python object.
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON:\n{json_text}") from e

    # 3. Validation Loop
    valid_items = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        
        # FIX: Accept if it has text (Ignore the 'id' requirement)
        if "text" not in item:
             print(f"Warning: Skipping item {i} (missing 'span'): {item}")
             continue

        if "label" not in item:
            continue
            
        valid_items.append(item) #requirees a label

    return valid_items # returns only clean data
