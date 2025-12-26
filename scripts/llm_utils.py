import json

def parse_llm_json(response_text):
    """
    Safely parse LLM output into Python objects.
    """
    try:
        data = json.loads(response_text)
        assert isinstance(data, list)
        for item in data:
            assert "id" in item
            assert "label" in item
        return data
    except Exception as e:
        raise ValueError(
            f"Invalid LLM JSON output:\n{response_text}\nError: {e}"
        )

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
