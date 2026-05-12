# save as count_iob.py
from collections import Counter

counter = Counter()
with open('data/test.iob', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        for part in parts:
            if part.startswith('B-'):
                counter[part[2:]] += 1

print(f"{'Category':<15} Count")
print("-" * 25)
for label, count in sorted(counter.items(), key=lambda x: -x[1]):
    print(f"{label:<15} {count:,}")
print(f"\nTotal: {sum(counter.values()):,}")
