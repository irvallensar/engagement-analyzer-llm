import json
import random
from collections import Counter

def eguchi_balance():
    with open('data/train.jsonl', 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    # Target multipliers aligned with Table 9 of Eguchi & Kyle (2024)
    moderate_minority = ["COUNTER", "DENY", "PROCLAIM", "ATTRIBUTE"]
    rare_minority = ["CITATION", "ENDOPHORIC", "JUSTIFYING", "SOURCES", "CONCUR", "ENDORSE"]
    majority = ["ENTERTAIN", "MONOGLOSS"]

    balanced_data = []
    majority_only = []

    for entry in data:
        content = entry['messages'][2]['content'] # To detect which labels exist
        
        # 1. Rare Classes (Multiply by 4 to looke like Table 9 scaling)
        if any(f'"label": "{mc}"' in content for mc in rare_minority):
            balanced_data.extend([entry] * 4)
            
        # 2. Moderate Minority (Multiply by 2)
        elif any(f'"label": "{mc}"' in content for mc in moderate_minority):
            balanced_data.extend([entry] * 2)
            
        # 3. Majority Only (Stored separately)
        else:
            majority_only.append(entry)

    # from overwhelming an LLM 
    random.seed(42)
    random.shuffle(majority_only)
    capped_majority = majority_only[:4000]

    final_dataset = balanced_data + capped_majority
    random.shuffle(final_dataset) # Prevents training bias and batch clustering of same class

    with open('data/train_balanced.jsonl', 'w', encoding='utf-8') as f:
        for d in final_dataset:
            f.write(json.dumps(d) + '\n')

    print(f"Original Base Sentences: {len(data)}")
    print(f"[SUCCESS] Eguchi-Aligned Dataset Size: {len(final_dataset)}")

if __name__ == "__main__":
    eguchi_balance()
