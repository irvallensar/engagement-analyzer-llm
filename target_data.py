import os

def filter_iob(input_file, output_file, target_classes):
    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        blocks = f.read().split('\n\n')

    kept_blocks = []
    counts = {c: 0 for c in target_classes}
    discarded = 0

    for block in blocks:
        if not block.strip(): continue
        
        tag = None
        for line in block.split('\n'):
            if '\t' in line or ' ' in line:
                parts = line.replace('\t', ' ').split(' ')
                if len(parts) > 1 and parts[-1] != 'O':
                    tag = parts[-1].replace('B-', '').replace('I-', '').strip()
                    break
        
        if tag in target_classes:
            kept_blocks.append(block)
            counts[tag] += 1
        else:
            discarded += 1

    print("Writing targeted data...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(kept_blocks) + '\n\n')

    print("=== SURGICAL DATA EXTRACTION COMPLETE ===")
    for k, v in counts.items():
        print(f"KEPT -> {k}: {v} spans")
    print(f"DISCARDED -> {discarded} spans (Citations, Justifying, etc.)")

if __name__ == "__main__":
    # Ensure it reads from your FULL synthetic file, not the half-sized one
    filter_iob('data/synthetic_train.iob', 'data/synthetic_targeted.iob', ["SOURCES", "ENDOPHORIC"])
