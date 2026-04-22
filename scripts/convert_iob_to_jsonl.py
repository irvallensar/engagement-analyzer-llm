import json

# --- Configuration ---
INPUT_IOB = "data/train.iob"
OUTPUT_JSONL = "data/train.jsonl"

SYSTEM_PROMPT = (
    "You are an expert linguistic annotator. "
    "Extract Engagement markers and output them as a JSON array. "
    "You MUST strictly use ONLY the following 10 labels: "
    "['ATTRIBUTION', 'CITATION', 'COUNTER', 'DENY', 'ENDOPHORIC', 'ENTERTAIN', 'JUSTIFYING', 'MONOGLOSS', 'PROCLAIM', 'SOURCES']. "
    "CRITICAL BOUNDARY EXTRACTION RULES:\n"
    "1. CITATIONS: You must include the encapsulating parentheses and all authors/years within them (e.g., '(Smith, 2020; Doe, 2021)').\n"
    "2. MARKER ISOLATION: For COUNTER, JUSTIFYING, and MONOGLOSS, isolate the specific transition word or phrase (e.g., 'However', 'since'). DO NOT highlight the entire surrounding clause.\n"
    "3. MODIFIERS & AUXILIARIES: You must include adjacent adverbs, modals, and auxiliaries that modify the marker (e.g., 'strongly asserts', 'could not enforce', 'is also represented').\n"
    "4. EXISTENTIALS: For PROCLAIM or DENY phrases using 'there is' or 'there can be', include those starting words (e.g., 'there is no doubt').\n"
    "Each item must follow this format: "
    "[{\"label\": \"CATEGORY\", \"span\": \"target text\", \"context_before\": \"preceding text\"}]. "
    "If there are no Engagement markers, output []."
)
def process_iob_to_jsonl():
    print(f"Reading from {INPUT_IOB}...")
    
    with open(INPUT_IOB, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    sentences = []
    current_words = []
    current_tags_matrix = []
    
    # 1. Parse the IOB into a sentence matrix
    for line in lines:
        line = line.strip()
        if not line:
            if current_words:
                sentences.append({"words": current_words, "tags_matrix": current_tags_matrix})
                current_words = []
                current_tags_matrix = []
            continue
            
        if "-DOCSTART-" in line or line == "-X-" or line == "O":
            continue
            
        parts = line.split()
        if len(parts) >= 2:
            current_words.append(parts[0])
            current_tags_matrix.append(parts[1:]) 

    if current_words:
        sentences.append({"words": current_words, "tags_matrix": current_tags_matrix})

    # 2. Extract overlapping spans from every column
    sentences_processed = 0
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as out_f:
        for entry in sentences:
            words = entry["words"]
            tags_matrix = entry["tags_matrix"]
            sentence_text = " ".join(words)
            markers = []
            
            if tags_matrix:
                num_cols = len(tags_matrix[0])
                for col_idx in range(num_cols):
                    current_label = None
                    start_idx = -1
                    
                    for i, row in enumerate(tags_matrix):
                        tag = row[col_idx] if col_idx < len(row) else "O"
                        
                        if tag.startswith("B-"):
                            if current_label:
                                save_span(words, start_idx, i, current_label, markers)
                            current_label = tag[2:]
                            start_idx = i
                        elif tag.startswith("I-") and current_label == tag[2:]:
                            continue
                        else:
                            if current_label:
                                save_span(words, start_idx, i, current_label, markers)
                                current_label = None
                                start_idx = -1
                                
                    if current_label:
                        save_span(words, start_idx, len(tags_matrix), current_label, markers)
                        
            # Format into Chat ML with thought process
            assistant_response = f"<thought_process>\nAnalyzing the sentence for heteroglossic and monoglossic markers...\n</thought_process>\n{json.dumps(markers)}"
            
            chat_dict = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Analyze this sentence:\n\n{sentence_text}"},
                    {"role": "assistant", "content": assistant_response}
                ]
            }
            out_f.write(json.dumps(chat_dict) + "\n")
            sentences_processed += 1

    print(f"\n[SUCCESS] Conversion complete! {sentences_processed} sentences saved to {OUTPUT_JSONL}")

def save_span(words, start_idx, end_idx, label, markers):
    span_words = words[start_idx:end_idx]
    span_text = " ".join(span_words)
    
    context_start = max(0, start_idx - 3)
    context_words = words[context_start:start_idx]
    context_text = " ".join(context_words)
    
    markers.append({
        "label": label.upper(),
        "span": span_text,
        "context_before": context_text
    })

if __name__ == "__main__":
    process_iob_to_jsonl()
