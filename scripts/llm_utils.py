import re
import json

VALID_LABELS = {
    'ATTRIBUTION', 'CITATION', 'COUNTER', 'DENY', 'ENDOPHORIC', 
    'ENTERTAIN', 'JUSTIFYING', 'MONOGLOSS', 'PROCLAIM', 'SOURCES'
}

def parse_llm_json(raw_response):
    print(f"\n[DEBUG] LLM RAW OUTPUT:\n{raw_response}")
    
    # 1. Clean up Markdown artifacts
    clean_response = raw_response.replace('```json', '').replace('```', '').strip()
    
    try:
        # 2. Hunt for the JSON array brackets
        match = re.search(r'\[.*\]', clean_response, re.DOTALL)
        if match:
            clean_json = match.group(0)
            extracted_spans = json.loads(clean_json)
            
            # 3. Validation: Only keep dicts with valid labels and a 'span' key
            valid_spans = []
            for item in extracted_spans:
                # Ensure it's a dict and has the required keys
                if isinstance(item, dict) and 'label' in item and 'span' in item:
                    # Force uppercase just in case, and check against valid set
                    label = str(item['label']).upper().strip()
                    span = str(item['span']).strip()
                    
                    if label in VALID_LABELS:
                        valid_spans.append({
                            "label": label,
                            "span": span
                        })
                        
            print(f"[DEBUG] JSON EXTRACTED: {valid_spans}\n")
            return valid_spans
        else:
            print("[DEBUG] No JSON array found in output.\n")
            return []
            
    except json.JSONDecodeError as e:
        print(f"[DEBUG] JSON Decode Error: {e}\n")
        return []
