import json
import random

def boost_zero_classes():
    with open('data/train.jsonl', 'r') as f:
        data = [json.loads(line) for line in f]

    # The categories that scored 0.0000
    zero_classes = ["CITATION", "ENDOPHORIC", "JUSTIFYING", "SOURCES"]
    
    boosted_data = []
    
    for entry in data:
        boosted_data.append(entry) # Keep the original sentence
        
        content = entry['messages'][2]['content']
        
        # If the sentence contains a zero-class, duplicate it 4 times
        if any(f'"label": "{zc}"' in content for zc in zero_classes):
            boosted_data.extend([entry] * 4)

    # Shuffle to prevent the model from seeing identical sentences back-to-back
    random.seed(42)
    random.shuffle(boosted_data)

    with open('data/train_boosted.jsonl', 'w') as f:
        for d in boosted_data:
            f.write(json.dumps(d) + '\n')

    print(f"Original Dataset Size: {len(data)}")
    print(f"[SUCCESS] Boosted Dataset Size: {len(boosted_data)}")
    print("Replace your train.jsonl with train_boosted.jsonl for Run 12.")

if __name__ == "__main__":
    boost_zero_classes()
