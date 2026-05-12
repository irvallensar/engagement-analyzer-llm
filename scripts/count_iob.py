from collections import Counter

for split in ["train", "dev", "test"]:
    counter = Counter()
    with open(f'data/{split}.iob', 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            for part in parts:
                if part.startswith('B-'):
                    counter[part[2:]] += 1
    
    print(f"\n=== {split.upper()} ===")
    for label, count in sorted(counter.items(), key=lambda x: -x[1]):
        print(f"{label:<15} {count:,}")
    print(f"Total: {sum(counter.values()):,}")
