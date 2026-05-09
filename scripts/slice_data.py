import os

def reduce_iob(input_file, output_file, max_per_class=1000):
    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        # Split by double newline (which separates sentences in IOB)
        blocks = f.read().split('\n\n')

    counts = {"CITATION": 0, "ENDOPHORIC": 0, "JUSTIFYING": 0, "SOURCES": 0}
    kept_blocks = []

    for block in blocks:
        if not block.strip(): continue
        
        # Find which tag is in this sentence
        tag = None
        for line in block.split('\n'):
            if '\t' in line or ' ' in line:
                parts = line.replace('\t', ' ').split(' ')
                if len(parts) > 1 and parts[-1] != 'O':
                    tag = parts[-1].replace('B-', '').replace('I-', '').strip()
                    break
        
        if tag and tag in counts:
            if counts[tag] < max_per_class:
                kept_blocks.append(block)
                counts[tag] += 1
        else:
            # If it's somehow untagged, keep it (though there shouldn't be any)
            kept_blocks.append(block)

    print("Writing sliced data...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(kept_blocks) + '\n\n')

    print("=== NEW SYNTHETIC DATA DIET ===")
    for k, v in counts.items():
        print(f"{k}: {v} spans kept")

if __name__ == "__main__":
    reduce_iob('data/synthetic_only.iob', 'data/synthetic_half.iob')
