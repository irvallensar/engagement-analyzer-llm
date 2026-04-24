import re

VALID_LABELS = {
    'ATTRIBUTION', 'CITATION', 'COUNTER', 'DENY', 'ENDOPHORIC', 
    'ENTERTAIN', 'JUSTIFYING', 'MONOGLOSS', 'PROCLAIM', 'SOURCES'
}

def parse_llm_json(raw_response):
    # 1. THIS WILL PRINT EXACTLY WHAT THE LLM WROTE
    print(f"\n[DEBUG] LLM RAW OUTPUT:\n{raw_response}")
    
    # 2. Bulletproof Regex: catches newlines and upper/lowercase
    pattern = re.compile(r'<([A-Za-z_]+)>(.*?)</([A-Za-z_]+)>', re.DOTALL)
    spans = []
    
    for match in pattern.finditer(raw_response):
        open_tag = match.group(1).upper() # Force uppercase to match VALID_LABELS
        content = match.group(2).strip()
        
        if open_tag in VALID_LABELS:
            spans.append({
                "label": open_tag,
                "span": content
            })
            
    # 3. THIS WILL PRINT WHAT THE REGEX ACTUALLY EXTRACTED
    print(f"[DEBUG] REGEX EXTRACTED: {spans}\n")
        
    return spans
