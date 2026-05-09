import os
from collections import defaultdict
import random
random.seed(42)

def stratify_iob(input_file, num_folds=5):
    print(f"Reading {input_file} for stratification...")
    with open(input_file, 'r', encoding='utf-8') as f:
        blocks = f.read().strip().split('\n\n')

    # Step 1: Score each sentence based on its RAREST tag
    # The rarer the tag, the higher priority it has to be distributed evenly
    tag_priority = {
        "ENDOPHORIC": 10, "SOURCES": 9, "PROCLAIM": 8, "ATTRIBUTION": 7, 
        "DENY": 6, "JUSTIFYING": 5, "ENTERTAIN": 4, "MONOGLOSS": 3, 
        "COUNTER": 2, "CITATION": 1
    }
    
    scored_blocks = []
    for block in blocks:
        if not block.strip(): continue
        highest_priority = 0
        for line in block.split('\n'):
            if '\t' in line or ' ' in line:
                parts = line.replace('\t', ' ').split(' ')
                if len(parts) > 1 and parts[-1] != 'O':
                    tag = parts[-1].replace('B-', '').replace('I-', '').strip()
                    if tag in tag_priority and tag_priority[tag] > highest_priority:
                        highest_priority = tag_priority[tag]
        
        # If it has no tags, give it priority 0
        scored_blocks.append((highest_priority, block))

    # Step 2: Sort sentences so the rarest ones are dealt first
    scored_blocks.sort(key=lambda x: x[0], reverse=True)

    # Step 3: Deal into 5 buckets (Round-Robin)
    buckets = [[] for _ in range(num_folds)]
    for i, (_, block) in enumerate(scored_blocks):
        buckets[i % num_folds].append(block)

    # Step 4: Write the new Train and Test files for each fold
    print("Writing perfectly stratified folds...")
    for fold_idx in range(num_folds):
        test_bucket = buckets[fold_idx]
        random.shuffle(test_bucket)
        train_buckets = [b for i, b in enumerate(buckets) if i != fold_idx]
        
        # Flatten the train buckets
        train_flat = [block for sublist in train_buckets for block in sublist]
        random.shuffle(train_flat)
        # Write to files
        with open(f'data/5_fold_exp/strat_train{fold_idx+1}.iob', 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(train_flat) + '\n\n')
            
        with open(f'data/5_fold_exp/strat_test{fold_idx+1}.iob', 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(test_bucket) + '\n\n')
            
    print("=== STRATIFICATION COMPLETE ===")
    print(f"Generated strat_train1-5.iob and strat_test1-5.iob")

if __name__ == "__main__":
    stratify_iob('data/5_fold_exp/master_organic_full.iob')
