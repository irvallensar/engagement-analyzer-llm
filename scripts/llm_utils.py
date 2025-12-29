import json
import re

def parse_llm_json(response_text):
    """
    Safely parse LLM output into Python objects.
    """
    try:
        # Extract first JSON array from the response
        match = re.search(r"\[\s*{.*?}\s*\]", response_text, re.DOTALL)
        if not match:
            raise ValueError("No JSON array found in LLM output")

        json_text = match.group(0)
        return json.loads(json_text)

    except Exception as e:
        raise ValueError(
            f"Invalid LLM JSON output:\n{response_text}"
        ) from e

def build_labeling_prompt(sentence, candidates):
    prompt = f"""
Sentence:
\"{sentence}\"

Candidates:
"""
    for i, c in enumerate(candidates):
        prompt += f'{i}: "{c["text"]}"\n'

    prompt += """
Label each candidate using one of:
[ENTERTAIN, PROCLAIM, DENY, O]

Return JSON exactly in this format:
[
"""
    for i in range(len(candidates)):
        prompt += f'  {{"id": {i}, "label": "O"}},\n'
    prompt += "]"

    return prompt
