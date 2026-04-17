import json
from collections import Counter
import random

INPUT_IOB = "data/train.iob"
OUTPUT_JSONL = "data/train.jsonl"
# SOFT CAP: We only boost severely underrepresented classes up to ~3500
TARGET_MINIMUM = 3500 

def process():
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    label_counts = Counter()
    parsed_data = []

    for line in lines:
        data = json.loads(line)
        markers = json.loads(data["messages"][2]["content"])
        
        labels_in_sentence = set()
        for marker in markers:
            label = marker['label']
            label_counts[label] += 1
            labels_in_sentence.add(label)
            
        parsed_data.append({"original_line": line, "labels": labels_in_sentence})

    print("--- ORIGINAL DISTRIBUTION ---")
    for label, count in label_counts.most_common():
        print(f"{label}: {count}")

    balanced_dataset = []
    for item in parsed_data:
        balanced_dataset.append(item["original_line"])
        
        if item["labels"]:
            rarest_label = min(item["labels"], key=lambda l: label_counts[l])
            rarest_count = label_counts[rarest_label]
            
            # Only multiply if it's below our soft cap of 3500
            if rarest_count < TARGET_MINIMUM:
                multiplier = int(TARGET_MINIMUM / rarest_count)
                # Keep it strictly <= 2x max (only add 1 extra duplicate)
                added_copies = min(multiplier - 1, 1) 
                for _ in range(added_copies):
                    balanced_dataset.append(item["original_line"])

    random.seed(42)
    random.shuffle(balanced_dataset)

    new_counts = Counter()
    for line in balanced_dataset:
        markers = json.loads(json.loads(line)["messages"][2]["content"])
        for m in markers: new_counts[m['label']] += 1

    print("\n--- NEW SOFT-BALANCED DISTRIBUTION ---")
    for label, count in new_counts.most_common():
        print(f"{label}: {count}")

    print(f"\nBalanced Dataset Size: {len(balanced_dataset)} sentences")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for line in balanced_dataset: f.write(line)

if __name__ == "__main__":
    process()
