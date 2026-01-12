import json
import re

def parse_llm_json(text: str):
    """
    Extract and parse the first JSON array from LLM output.
    Enforces schema: each item must have 'id' and 'label'.
    """
    # Extract JSON array
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        raise ValueError("No JSON array found in LLM output")

    json_text = match.group(0)

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON:\n{json_text}") from e

    if not isinstance(data, list):
        raise ValueError("LLM output is not a JSON array")

    valid_items = []
    for i, item in enumerate(data):
        # CHANGE: Check for validity and skip instead of raising error
        if not isinstance(item, dict):
            print(f"Warning: Skipping item {i} (not a dictionary): {item}")
            continue
        
        # Check for malformed keys (like id0) or missing keys
        if "id" not in item:
            print(f"Warning: Skipping item {i} (missing 'id'): {item}")
            continue
        if "label" not in item:
            print(f"Warning: Skipping item {i} (missing 'label'): {item}")
            continue
            
        valid_items.append(item)

    return valid_items
