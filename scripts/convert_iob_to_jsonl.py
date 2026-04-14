import json
from pathlib import Path

# --- Configuration ---
INPUT_IOB = "data/train.iob"
OUTPUT_JSONL = "data/valid.jsonl"

# The hyper-concise system prompt for fine-tuning
SYSTEM_PROMPT = "You are an expert annotator. Extract Engagement markers as a JSON array."

def process_iob_to_jsonl():
    print(f"Reading from {INPUT_IOB}...")
    
    with open(INPUT_IOB, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    sentences_processed = 0
    current_words = []
    current_labels = []
    
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as out_f:
        
        for line in lines:
            line = line.strip()
            
            # If the line is empty, we have reached the end of a sentence
            if not line:
                if current_words:
                    process_sentence(current_words, current_labels, out_f)
                    sentences_processed += 1
                    current_words = []
                    current_labels = []
                continue
            
            # Split the line into word and label (handles both spaces and tabs)
            parts = line.split()
            if len(parts) >= 2:
                word = parts[0]
                label = parts[-1] # Label is usually the last item
                current_words.append(word)
                current_labels.append(label)

        # Catch the last sentence if the file doesn't end with an empty line
        if current_words:
            process_sentence(current_words, current_labels, out_f)
            sentences_processed += 1

    print(f"\n[SUCCESS] Conversion complete! {sentences_processed} sentences saved to {OUTPUT_JSONL}")

def process_sentence(words, labels, out_f):
    sentence_text = " ".join(words)
    markers = []
    
    i = 0
    while i < len(words):
        label = labels[i]
        
        # Clean up B- and I- tags if they exist in your IOB file
        clean_label = label.replace("B-", "").replace("I-", "")
        
        if clean_label != "O":
            start_idx = i
            # Look ahead to find the full multi-word span
            while i + 1 < len(words) and labels[i+1].replace("B-", "").replace("I-", "") == clean_label:
                # Standard IOB rules: A 'B-' tag means a new marker starts, even if it's the same label
                if labels[i+1].startswith("B-"):
                    break
                i += 1
            
            span_words = words[start_idx : i+1]
            span_text = " ".join(span_words)
            
            # Grab context before (up to 3 words)
            context_start = max(0, start_idx - 3)
            context_words = words[context_start:start_idx]
            context_text = " ".join(context_words)
            
            markers.append({
                "text": span_text,
                "label": clean_label.upper(),
                "context_before": context_text
            })
            
        i += 1

    # Format into the exact Chat ML dictionary MLX requires
    chat_dict = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sentence_text},
            # We convert the markers list into a strict JSON string for the assistant's response
            {"role": "assistant", "content": json.dumps(markers)}
        ]
    }
    
    # Write as a single line in the JSONL file
    out_f.write(json.dumps(chat_dict) + "\n")

if __name__ == "__main__":
    process_iob_to_jsonl()
