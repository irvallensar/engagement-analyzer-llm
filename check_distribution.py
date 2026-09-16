import json
import re
from collections import Counter

def check_distribution(file_path):
    label_counts = Counter()
    total_sentences = 0
    sentences_with_markers = 0

    print(f"Scanning {file_path}...\n")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                total_sentences += 1
                try:
                    data = json.loads(line.strip())
                    
                    # The assistant's response is the third message in the array
                    assistant_content = data['messages'][2]['content']
                    
                    # Extract all labels using regex to bypass the thought_process text safely
                    matches = re.findall(r'"label":\s*"([^"]+)"', assistant_content)
                    
                    if matches:
                        sentences_with_markers += 1
                        label_counts.update(matches)
                        
                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    print(f"Warning: Could not parse structure on line {total_sentences}: {e}")
                    
    except FileNotFoundError:
        print(f"Error: Could not find {file_path}")
        return

    print("--- DATASET STATISTICS ---")
    print(f"Total Sentences Processed: {total_sentences}")
    print(f"Sentences with Markers: {sentences_with_markers}")
    print("\n--- LABEL DISTRIBUTION ---")
    
    # Print the counts sorted from highest to lowest
    for label, count in label_counts.most_common():
        print(f"  {label.ljust(15)}: {count}")

if __name__ == "__main__":
    # Ensure this points to your active training file
    check_distribution('data/train.jsonl')
