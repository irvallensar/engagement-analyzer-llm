import json
from collections import Counter
import random
import os

# We will read from your original unbalanced backup!
INPUT_FILE = "data/train_unbalanced.jsonl" 
OUTPUT_FILE = "data/train_balanced_gentle.jsonl"
TARGET_MINIMUM = 3000  # Your suggested 2x cap!

def process():
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: {INPUT_FILE} not found. Please rename your original run 1 dataset to train_unbalanced.jsonl")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    label_counts = Counter()
    parsed_data = []

    for line in lines:
        data = json.loads(line)
        assistant_msg = data["messages"][2]["content"]
        markers = json.loads(assistant_msg)
        
        labels_in_sentence = set()
        for marker in markers:
            label = marker['label']
            label_counts[label] += 1
            labels_in_sentence.add(label)
            
        parsed_data.append({"original_line": line, "labels": labels_in_sentence})

    balanced_dataset = []
    for item in parsed_data:
        balanced_dataset.append(item["original_line"]) # Add original
        
        if item["labels"]:
            rarest_label = min(item["labels"], key=lambda l: label_counts[l])
            rarest_count = label_counts[rarest_label]
            
            # Gentle Oversampling: Only boost if under 3000
            if rarest_count < TARGET_MINIMUM:
                # Max multiplier is basically 2x or 3x, not 5x
                multiplier = min(int(TARGET_MINIMUM / rarest_count), 2) 
                for _ in range(multiplier):
                    balanced_dataset.append(item["original_line"])

    random.seed(42)
    random.shuffle(balanced_dataset)

    print(f"Original Dataset Size: {len(lines)} sentences")
    print(f"New Gentle Dataset Size: {len(balanced_dataset)} sentences")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for line in balanced_dataset:
            f.write(line)

if __name__ == "__main__":
    process()
