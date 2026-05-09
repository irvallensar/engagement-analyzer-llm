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

    print("Writing perfectly stratified folds...")
    for fold_idx in range(num_folds):
        # Test = current fold
        test_bucket = buckets[fold_idx]
    
        # Dev = next fold (rotating)
        dev_idx = (fold_idx + 1) % num_folds
        dev_bucket = buckets[dev_idx]
    
        # Train = remaining 3 folds
        train_flat = [block 
                    for i, sublist in enumerate(buckets) 
                    if i != fold_idx and i != dev_idx 
                    for block in sublist]
    
        random.shuffle(test_bucket)
        random.shuffle(dev_bucket)
        random.shuffle(train_flat)
    
        # Write IOB files
        with open(f'data/5_fold_exp/strat_train{fold_idx+1}.iob', 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(train_flat) + '\n\n')
        
        with open(f'data/5_fold_exp/strat_dev{fold_idx+1}.iob', 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(dev_bucket) + '\n\n')
        
        with open(f'data/5_fold_exp/strat_test{fold_idx+1}.iob', 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(test_bucket) + '\n\n')

    print("=== STRATIFICATION COMPLETE ===")
    print(f"Generated strat_train1-5.iob, strat_dev1-5.iob, strat_test1-5.iob")

if __name__ == "__main__":
    stratify_iob('data/5_fold_exp/master_organic_full.iob')
