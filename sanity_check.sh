#!/bin/bash

# Pointing to your new diet-sized data
SYNTHETIC_DATA="data/synthetic_targeted.iob" 

echo "=== SANITY CHECK: FOLD 1 ONLY ==="

# 1. Merge the organic Fold 1 with the new diet synthetic data
cat data/5_fold_exp/train1.iob "$SYNTHETIC_DATA" > data/5_fold_exp/combined_train1.iob

# 2. Convert to spaCy format
python3 scripts/rebuild_spacy.py data/5_fold_exp/combined_train1.iob data/5_fold_exp/da_train1.spacy

# 3. Train JUST Fold 1 (Using your updated config.cfg with 0.2 dropout)
echo "Training Sanity Model..."
caffeinate -i python3 -m spacy train config.cfg --output ./models/sanity_da_1 --paths.train data/5_fold_exp/da_train1.spacy --paths.dev data/5_fold_exp/dev1.spacy --gpu-id 0

# 4. Evaluate
echo "=== SANITY CHECK RESULTS ===" > sanity_results.txt
python3 -m spacy evaluate ./models/sanity_da_1/model-best data/5_fold_exp/test1.spacy --gpu-id 0 >> sanity_results.txt

echo "ALL DONE. CHECK sanity_results.txt"
