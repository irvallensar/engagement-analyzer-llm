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
    for i, item in enumerate(data): # enumerate gives index + value
        if not isinstance(item, dict): #rejects strings, numbers, lists
            continue
        
        # FIX: Accept either 'id' or 'text'
        if "id" not in item and "text" not in item: #requires at least one item or one text
             print(f"Warning: Skipping item {i} (missing 'id' and 'text'): {item}")
             continue

        if "label" not in item: #requires label
            continue
            
        valid_items.append(item) 

    return valid_items #Returns only clean data.
