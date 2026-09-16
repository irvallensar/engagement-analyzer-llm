#!/bin/bash

# Point exactly to the FULL synthetic data (all 4 classes)
SYNTHETIC_DATA="data/synthetic_train.iob" 

echo "=== STEP 1: PREPPING ORIGINAL FOLDS WITH FULL DA ==="
for i in {1..5}; do
    python3 scripts/rebuild_spacy.py data/5_fold_exp/test${i}.iob data/5_fold_exp/test${i}.spacy
    python3 scripts/rebuild_spacy.py data/5_fold_exp/dev${i}.iob data/5_fold_exp/dev${i}.spacy
    python3 scripts/rebuild_spacy.py data/5_fold_exp/train${i}.iob data/5_fold_exp/organic_train${i}.spacy
    
    # Surgical Injection: Add FULL synthetic ONLY to the organic train fold
    cat data/5_fold_exp/train${i}.iob "$SYNTHETIC_DATA" > data/5_fold_exp/combined_train${i}.iob
    
    python3 scripts/rebuild_spacy.py data/5_fold_exp/combined_train${i}.iob data/5_fold_exp/da_train${i}.spacy
done

echo "=== STEP 2: TRAINING ORGANIC BASELINES ==="
for i in {1..5}; do
    python3 -m spacy train config.cfg --output ./models/full_organic_${i} --paths.train data/5_fold_exp/organic_train${i}.spacy --paths.dev data/5_fold_exp/dev${i}.spacy --gpu-id 0
done

echo "=== STEP 3: TRAINING FULL DA-RoBERTa ==="
for i in {1..5}; do
    python3 -m spacy train config.cfg --output ./models/full_da_${i} --paths.train data/5_fold_exp/da_train${i}.spacy --paths.dev data/5_fold_exp/dev${i}.spacy --gpu-id 0
done

echo "=== STEP 4: EVALUATION ==="
echo "FULL ORGANIC RESULTS" > full_da_results.txt
for i in {1..5}; do
    python3 -m spacy evaluate ./models/full_organic_${i}/model-best data/5_fold_exp/test${i}.spacy --gpu-id 0 >> full_da_results.txt
done

echo "FULL DA-ROBERTA RESULTS" >> full_da_results.txt
for i in {1..5}; do
    python3 -m spacy evaluate ./models/full_da_${i}/model-best data/5_fold_exp/test${i}.spacy --gpu-id 0 >> full_da_results.txt
done

echo "ALL DONE. CHECK full_da_results.txt"
