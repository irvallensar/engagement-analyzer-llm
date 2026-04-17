import json

INPUT_IOB = "data/train.iob"
OUTPUT_JSONL = "data/train.jsonl"
SYSTEM_PROMPT = "You are an expert annotator. Extract Engagement markers as a JSON array."

def convert_iob_to_jsonl():
    with open(INPUT_IOB, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    dataset = []
    current_words = []
    current_markers = []
    
    # Track character offsets for strict span extraction
    current_char_idx = 0 
    current_span_start = None
    current_label = None

    print(f"Reading from {INPUT_IOB}...")

    for line in lines:
        line = line.strip()
        if not line: # End of sentence
            if current_words:
                sentence_text = " ".join(current_words)
                
                # Format exactly like Run 1 (No CoT, just JSON)
                message_dict = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": sentence_text},
                        {"role": "assistant", "content": json.dumps(current_markers)}
                    ]
                }
                dataset.append(message_dict)
                
            # Reset for next sentence
            current_words = []
            current_markers = []
            current_char_idx = 0
            continue

        parts = line.split('\t')
        if len(parts) >= 2:
            word = parts[0]
            tag = parts[-1]

            # Calculate start/end indices for the marker
            start_idx = current_char_idx
            end_idx = start_idx + len(word)

            if tag.startswith("B-"):
                # Save previous marker if exists
                if current_label:
                    current_markers.append({"label": current_label, "start": current_span_start, "end": start_idx - 1})
                current_label = tag[2:]
                current_span_start = start_idx
            elif tag.startswith("I-") and current_label == tag[2:]:
                pass # Continue the span
            else:
                if current_label:
                    current_markers.append({"label": current_label, "start": current_span_start, "end": start_idx - 1})
                    current_label = None

            current_words.append(word)
            current_char_idx = end_idx + 1 # +1 for the space

    # Save the pure jsonl
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as out_f:
        for item in dataset:
            out_f.write(json.dumps(item) + '\n')

    print(f"[SUCCESS] Saved {len(dataset)} sentences to {OUTPUT_JSONL}")

if __name__ == "__main__":
    convert_iob_to_jsonl()
