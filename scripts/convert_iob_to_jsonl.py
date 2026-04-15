import json
import spacy
from spacy.tokens import Doc
from pathlib import Path

# Load spaCy
nlp = spacy.load("en_core_web_sm")

# We will change these manually for train vs dev
INPUT_IOB = "data/train.iob"
OUTPUT_JSONL = "data/train.jsonl"

# NEW: Updated System Prompt to demand reasoning first
SYSTEM_PROMPT = "You are an expert annotator. First, analyze the sentence and explicitly identify any Engagement markers. Then, output the final result as a strict JSON array."

def process():
    print(f"Reading from {INPUT_IOB}...")
    with open(INPUT_IOB, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    sentences = []
    current_tokens = []
    current_tags_matrix = []

    for line in lines:
        line = line.strip()
        if not line:
            if current_tokens:
                sentences.append({"tokens": current_tokens, "tags_matrix": current_tags_matrix})
                current_tokens = []
                current_tags_matrix = []
            continue
        
        if "-DOCSTART-" in line or line == "-X-" or line == "O":
            continue

        parts = line.split()
        if len(parts) >= 2:
            current_tokens.append(parts[0])
            current_tags_matrix.append(parts[1:])

    if current_tokens:
        sentences.append({"tokens": current_tokens, "tags_matrix": current_tags_matrix})

    print(f"Found {len(sentences)} sentences. Extracting markers and generating CoT reasoning...")

    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as out_f:
        total_markers_found = 0
        
        for entry in sentences:
            doc = Doc(nlp.vocab, words=entry["tokens"])
            text = doc.text
            gold_spans = set()

            # THE MATRIX DECODER
            if entry["tags_matrix"]:
                num_cols = len(entry["tags_matrix"][0])
                for col_idx in range(num_cols):
                    current_label = None
                    start_idx = -1
                    for i, row in enumerate(entry["tags_matrix"]):
                        tag = row[col_idx] if col_idx < len(row) else "O"
                        if tag.startswith("B-"):
                            if current_label: gold_spans.add((current_label, start_idx, i))
                            current_label = tag[2:]
                            start_idx = i
                        elif tag.startswith("I-") and current_label == tag[2:]: continue
                        else:
                            if current_label:
                                gold_spans.add((current_label, start_idx, i))
                                current_label = None
                                start_idx = -1
                    if current_label: gold_spans.add((current_label, start_idx, len(entry["tags_matrix"])))
            
            markers = []
            reasoning_lines = []
            
            for label, start_idx, end_idx in gold_spans:
                span = doc[start_idx:end_idx]
                context_start = max(0, start_idx - 3)
                context_words = doc[context_start:start_idx].text.strip()
                
                markers.append({
                    "text": span.text,
                    "label": label.upper(),
                    "context_before": context_words
                })
                
                # SYNTHETIC REASONING GENERATION
                reasoning_lines.append(f"- Found the span '{span.text}'. In this context, it functions as an {label.upper()} marker.")
                total_markers_found += 1

            # Build the Assistant's CoT Response
            if markers:
                thought_process = "Analyzing the sentence for Engagement markers...\n" + "\n".join(reasoning_lines) + "\n\nExtraction complete. Generating JSON array:\n"
            else:
                thought_process = "Analyzing the sentence for Engagement markers...\n- No Engagement markers were found in this sentence.\n\nExtraction complete. Generating JSON array:\n"

            # Combine reasoning with the strict JSON block
            final_assistant_content = thought_process + "```json\n" + json.dumps(markers, indent=2) + "\n```"

            # Format into Chat ML
            chat_dict = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": final_assistant_content}
                ]
            }
            out_f.write(json.dumps(chat_dict) + "\n")

    print(f"\n[SUCCESS] CoT Conversion complete!")
    print(f"Saved to {OUTPUT_JSONL}")

if __name__ == "__main__":
    process()
