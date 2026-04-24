import re
import json

VALID_LABELS = {
    'ATTRIBUTION', 'CITATION', 'COUNTER', 'DENY', 'ENDOPHORIC', 
    'ENTERTAIN', 'JUSTIFYING', 'MONOGLOSS', 'PROCLAIM', 'SOURCES'
}

def parse_llm_json(raw_response):
    """
    Validation Layer: Parses XML tags from the LLM response and converts 
    them into the Python list of dictionaries expected by the evaluator.
    """
    # This is the XML regex. It hunts for <TAG>text</TAG> instead of brackets.
    pattern = re.compile(r'<([A-Z_]+)>(.*?)</([A-Z_]+)>')
    spans = []
    
    # We scan the raw text for XML tags
    for match in pattern.finditer(raw_response):
        open_tag = match.group(1)
        content = match.group(2).strip()
        
        # AUTO-FIX: We only trust the opening tag, and only if it's a valid label.
        if open_tag in VALID_LABELS:
            spans.append({
                "label": open_tag,
                "span": content
            })
            
    return spans
