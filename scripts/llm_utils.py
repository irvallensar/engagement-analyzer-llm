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
