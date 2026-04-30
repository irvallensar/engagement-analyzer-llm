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

def iob_to_jsonl(iob_file_path, output_file_path, include_synthetic=False):
    dataset = []
    
    with open(iob_file_path, 'r', encoding='utf-8') as f:
        tokens, labels1, labels2 = [], [], []
        
        for line in f:
            line = line.strip()
            if not line or line.startswith("-DOCSTART-"):
                if tokens:
                    entry = process_sentence(tokens, labels1, labels2)
                    if entry is not None:
                        dataset.append(entry)
                tokens, labels1, labels2 = [], [], []
                continue
                
            parts = line.split()
            if len(parts) >= 2:
                tokens.append(parts[0])
                labels1.append(parts[1].upper())
                labels2.append(parts[2].upper() if len(parts) >= 3 else 'O')
                
        if tokens:
            entry = process_sentence(tokens, labels1, labels2)
            if entry is not None:
                dataset.append(entry)

    # 2. Only merge synthetic data for training split
    synthetic_count = 0
    if include_synthetic:
        synthetic_file = 'data/synthetic_json_raw.jsonl'
        if os.path.exists(synthetic_file):
            with open(synthetic_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    spans = [{"label": data["label"], "span": data["span"]}]
                    dataset.append(format_chatml(data["sentence"], json.dumps(spans, ensure_ascii=False)))
                    synthetic_count += 1

    # 3. Shuffle
    random.shuffle(dataset)

    # 4. Save
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print(f"[SUCCESS] Merged and SHUFFLED {len(dataset) - synthetic_count} real and {synthetic_count} synthetic sentences to {output_file_path}!")


def process_sentence(tokens, tags1, tags2):
    raw_sentence = " ".join(tokens)
    spans = []
    
    # Process both label columns independently
    for tag_sequence in [tags1, tags2]:
        current_span = []
        current_label = None
        
        for word, tag in zip(tokens, tag_sequence):
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
    iob_to_jsonl('data/train.iob', 'data/train.jsonl', include_synthetic=True)
    iob_to_jsonl('data/dev.iob', 'data/valid.jsonl', include_synthetic=False)
    iob_to_jsonl('data/test.iob', 'data/test.jsonl', include_synthetic=False)
