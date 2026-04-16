import json
from collections import Counter
import random

INPUT_FILE = "data/train.jsonl"
OUTPUT_FILE = "data/train_balanced.jsonl"
# This is the minimum number of times the model needs to see a label to not "forget" it
TARGET_MINIMUM = 1500 

def process():
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 1. Count the global distribution
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
            
        parsed_data.append({
            "original_line": line,
            "labels": labels_in_sentence
        })

    print("--- ORIGINAL DISTRIBUTION ---")
    for label, count in label_counts.most_common():
        print(f"{label}: {count}")

    # 2. Oversample minority sentences
    balanced_dataset = []
    for item in parsed_data:
        # Always add the original sentence at least once
        balanced_dataset.append(item["original_line"])
        
        if item["labels"]:
            # Find the rarest label in this specific sentence
            rarest_label = min(item["labels"], key=lambda l: label_counts[l])
            rarest_count = label_counts[rarest_label]
            
            # If the rarest label is under the target, duplicate the sentence!
            if rarest_count < TARGET_MINIMUM:
                # Calculate the exact multiplier needed
                multiplier = int(TARGET_MINIMUM / rarest_count)
                # We already added it once, so add the remainder
                for _ in range(multiplier - 1):
                    balanced_dataset.append(item["original_line"])

    # 3. Shuffle so the duplicates are spread evenly across the training epoch
    random.seed(42)
    random.shuffle(balanced_dataset)

    # 4. Prove the new math
    new_counts = Counter()
    for line in balanced_dataset:
        markers = json.loads(json.loads(line)["messages"][2]["content"])
        for m in markers:
            new_counts[m['label']] += 1

    print("\n--- NEW BALANCED DISTRIBUTION ---")
    for label, count in new_counts.most_common():
        print(f"{label}: {count}")

    print(f"\nOriginal Dataset Size: {len(lines)} sentences")
    print(f"Balanced Dataset Size: {len(balanced_dataset)} sentences")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for line in balanced_dataset:
            f.write(line)

if __name__ == "__main__":
    process()
