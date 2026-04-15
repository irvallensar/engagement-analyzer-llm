import json
import random

INPUT_FILE = "data/train.jsonl"
OUTPUT_FILE = "data/train_balanced.jsonl"
# This forces a perfect 50/50 split of sentences with markers vs empty sentences
EMPTY_RATIO = 1.0 

def balance():
    print(f"Reading from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    positives = []
    empties = []

    for line in lines:
        data = json.loads(line)
        # The assistant's response is the 3rd message in the ChatML format
        assistant_content = data["messages"][2]["content"]
        
        if assistant_content == "[]":
            empties.append(line)
        else:
            positives.append(line)

    print(f"\nOriginal Dataset Breakdown:")
    print(f"  Sentences WITH markers: {len(positives)}")
    print(f"  Sentences WITHOUT markers (Empty): {len(empties)}")

    # Calculate how many empty sentences to keep
    keep_amount = int(len(positives) * EMPTY_RATIO)
    
    # Randomly select the empty sentences
    random.seed(42) # Keeps the random selection consistent if you run it twice
    sampled_empties = random.sample(empties, min(keep_amount, len(empties)))

    print(f"\nDownsampling empty sentences to {len(sampled_empties)}...")

    # Combine and shuffle so the model doesn't learn a repeating pattern
    balanced_dataset = positives + sampled_empties
    random.shuffle(balanced_dataset)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for line in balanced_dataset:
            f.write(line)

    print(f"\n[SUCCESS] Balanced dataset created: {len(balanced_dataset)} total sentences saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    balance()
