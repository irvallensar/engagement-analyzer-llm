import json
import random

def aggressive_boost():
    with open('data/train.jsonl', 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    boosted_data = []
    
    for entry in data:
        boosted_data.append(entry) # Keep the original sentence
        
        content = entry['messages'][2]['content']
        
        # Multiply the extremely low classes by 30
        if '"label": "JUSTIFYING"' in content or '"label": "ENDOPHORIC"' in content:
            boosted_data.extend([entry] * 30)
            
        # If CITATION or SOURCES actually exist but were missed, multiply them heavily too
        elif '"label": "CITATION"' in content or '"label": "SOURCES"' in content:
            boosted_data.extend([entry] * 30)

    # Shuffle to prevent the model from memorizing duplicate batches
    random.seed(42)
    random.shuffle(boosted_data)

    with open('data/train_aggressively_boosted.jsonl', 'w', encoding='utf-8') as f:
        for d in boosted_data:
            f.write(json.dumps(d) + '\n')

    print(f"Original Dataset Size: {len(data)}")
    print(f"[SUCCESS] Aggressively Boosted Size: {len(boosted_data)}")
    print("Run your check_distribution.py on 'data/train_aggressively_boosted.jsonl' to verify.")

if __name__ == "__main__":
    aggressive_boost()
