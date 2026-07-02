import json
import sys

def verify_extracted_sentences(jsonl_path="pseudo_labeled_corpus.jsonl", num_samples=15):
    print(f"=== Verifying First {num_samples} Pseudo-Labeled Sentences ===")
    print("Format: [LABEL: extracted text segment]\n" + "="*60)
    
    count = 0
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if count >= num_samples:
                    break
                
                data = json.loads(line)
                text = data["text"]
                spans = data["spans"]
                
                # Sort spans in reverse order (back to front) 
                # This prevents character offsets from breaking when modifying the string length!
                sorted_spans = sorted(spans, key=lambda x: x["start_char"], reverse=True)
                
                formatted_text = text
                for span in sorted_spans:
                    start = span["start_char"]
                    end = span["end_char"]
                    label = span["label"]
                    
                    # Wrap the target text with clear visual indicators
                    tagged_segment = f" **[{label} → {formatted_text[start:end]}]** "
                    formatted_text = formatted_text[:start] + tagged_segment + formatted_text[end:]
                    
                print(f"\n[Sample #{count + 1}] ID: {data.get('sentence_id', 'UNKNOWN')}")
                print(f"Rendered: {formatted_text}")
                print("-" * 60)
                count += 1
                
    except FileNotFoundError:
        print(f"Error: Could not find '{jsonl_path}'. Make sure your extraction script ran completely.")
        sys.exit(1)

if __name__ == "__main__":
    # Adjust num_samples to check more or fewer sentences
    verify_extracted_sentences("pseudo_labeled_corpus.jsonl", num_samples=30)
