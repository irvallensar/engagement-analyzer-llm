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

    # Schema validation
    if not isinstance(data, list):
        raise ValueError("LLM output is not a JSON array")

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {i} is not an object: {item}")
        if "id" not in item:
            raise ValueError(f'Missing "id" in item {i}: {item}')
        if "label" not in item:
            raise ValueError(f'Missing "label" in item {i}: {item}')

    return data
