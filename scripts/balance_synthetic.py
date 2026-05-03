import json
from collections import defaultdict

input_file = "data/synthetic_data_clean.jsonl"
output_file = "data/synthetic_balanced.jsonl"
TARGET = 2000

counts = defaultdict(int)
balanced_data = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        label = data["label"]
        
        # Only keep it if we haven't hit 2000 for this label yet
        if counts[label] < TARGET:
            balanced_data.append(data)
            counts[label] += 1

with open(output_file, "w", encoding="utf-8") as f:
    for item in balanced_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("Data successfully balanced!")
for label, count in counts.items():
    print(f"{label}: {count}")
