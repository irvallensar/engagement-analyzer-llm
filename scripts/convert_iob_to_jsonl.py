import json
import os
import random

# PURE JSON PROMPT - NO CONTEXT_BEFORE
SYSTEM_PROMPT = (
    "You are an expert linguistic annotator. Analyze the sentence and extract all Engagement markers. "
    "Output a JSON array of dictionaries with 'label' and 'span' keys. "
    "The 10 valid tags are: ATTRIBUTION, CITATION, COUNTER, DENY, ENDOPHORIC, ENTERTAIN, JUSTIFYING, MONOGLOSS, PROCLAIM, SOURCES. "
    "Example Input: I do not believe this approach works. "
    "Example Output: [{\"label\": \"DENY\", \"span\": \"not\"}, {\"label\": \"ENTERTAIN\", \"span\": \"believe\"}] "
    "If there are no markers, output []."
)

def iob_to_jsonl(iob_file_path, output_file_path):
    dataset = []
    
    # 1. Process the Real Data
    with open(iob_file_path, 'r', encoding='utf-8') as f:
        tokens, labels = [], []
        
        for line in f:
            line = line.strip()
            if not line or line.startswith("-DOCSTART-"):
                if tokens:
                    entry = process_sentence(tokens, labels)
                    if entry is not None:
                        dataset.append(entry)
                tokens, labels = [], []
                continue
                
            parts = line.split()
            if len(parts) >= 2:
                tokens.append(parts[0])
                labels.append(parts[1].upper()) # Safely pulling from column 1
                
        if tokens:
            entry = process_sentence(tokens, labels)
            if entry is not None:
                dataset.append(entry)

    # 2. Process the Synthetic Data (Reading from RAW to strip context_before)
    synthetic_file = 'data/synthetic_json_raw.jsonl'
    synthetic_count = 0
    if os.path.exists(synthetic_file):
        with open(synthetic_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                # Force the pure schema
                spans = [{"label": data["label"], "span": data["span"]}]
                dataset.append(format_chatml(data["sentence"], json.dumps(spans, ensure_ascii=False)))
                synthetic_count += 1

    # 3. The Crucial Shuffle
    random.shuffle(dataset)

    # 4. Save to Master File
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print(f"[SUCCESS] Merged and SHUFFLED {len(dataset) - synthetic_count} real and {synthetic_count} synthetic sentences to {output_file_path}!")

def process_sentence(tokens, tags):
    clean = [(w, t) for w, t in zip(tokens, tags) if w != "-DOCSTART-"]
    if not clean:
        return None
    tokens, tags = zip(*clean)
    raw_sentence = " ".join(tokens)
    spans = []
    current_span = []
    current_label = None

    
    for word, tag in zip(tokens, tags):
        if tag.startswith("B-"):
            if current_label:
                spans.append({"label": current_label, "span": " ".join(current_span)})
            current_label = tag[2:]
            current_span = [word]
        elif tag.startswith("I-") and current_label == tag[2:]:
            current_span.append(word)
        else:
            if current_label:
                spans.append({"label": current_label, "span": " ".join(current_span)})
                current_label = None
                current_span = []
                
    if current_label:
        spans.append({"label": current_label, "span": " ".join(current_span)})
        
    return format_chatml(raw_sentence, json.dumps(spans, ensure_ascii=False))

def format_chatml(raw_text, json_spans):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyze this sentence:\n\n{raw_text}"},
            {"role": "assistant", "content": json_spans}
        ]
    }

if __name__ == "__main__":
    iob_to_jsonl('data/train.iob', 'data/train.jsonl')
    iob_to_jsonl('data/dev.iob', 'data/valid.jsonl')
    iob_to_jsonl('data/test.iob', 'data/test.jsonl')
    
