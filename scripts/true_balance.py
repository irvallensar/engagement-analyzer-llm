import json
import random

def true_balance():
    with open('data/train.jsonl', 'r') as f:
        data = [json.loads(line) for line in f]

    # The rare classes we desperately need the model to learn
    minority_classes = ["CITATION", "COUNTER", "DENY", "ENDOPHORIC", "JUSTIFYING", "PROCLAIM", "SOURCES", "ATTRIBUTION"]
    majority_classes = ["ENTERTAIN", "MONOGLOSS"]

    balanced_data = []
    majority_only_data = []

    for entry in data:
        content = entry['messages'][2]['content']
        
        # Check if the sentence contains ANY rare class
        has_minority = any(f'"label": "{mc}"' in content for mc in minority_classes)
        
        if has_minority:
            balanced_data.append(entry)
        else:
            # If it only has majority classes, quarantine it
            has_majority = any(f'"label": "{mc}"' in content for mc in majority_classes)
            if has_majority:
                majority_only_data.append(entry)

    # Aggressively downsample the majority-only sentences to 600
    random.seed(42)
    random.shuffle(majority_only_data)
    downsampled_majority = majority_only_data[:600] 

    final_dataset = balanced_data + downsampled_majority
    random.shuffle(final_dataset)

    # Save over the old file so MLX picks it up automatically
    with open('data/train.jsonl', 'w') as f:
        for d in final_dataset:
            f.write(json.dumps(d) + '\n')

    print(f"Original Dataset Size: {len(data)}")
    print(f"Rare Class Sentences Preserved: {len(balanced_data)}")
    print(f"Majority Class Sentences Kept: {len(downsampled_majority)}")
    print(f"[SUCCESS] New Highly-Concentrated Dataset Size: {len(final_dataset)}")

if __name__ == "__main__":
    true_balance()
