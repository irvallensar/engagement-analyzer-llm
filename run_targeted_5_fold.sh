#!/bin/bash

SYNTHETIC_DATA="data/synthetic_targeted.iob" 

echo "=== STEP 1: PREPPING ORIGINAL FOLDS WITH TARGETED DA ==="
for i in {1..5}; do
    python3 scripts/rebuild_spacy.py data/5_fold_exp/test${i}.iob data/5_fold_exp/test${i}.spacy
    python3 scripts/rebuild_spacy.py data/5_fold_exp/dev${i}.iob data/5_fold_exp/dev${i}.spacy
    python3 scripts/rebuild_spacy.py data/5_fold_exp/train${i}.iob data/5_fold_exp/organic_train${i}.spacy
    cat data/5_fold_exp/train${i}.iob "$SYNTHETIC_DATA" > data/5_fold_exp/combined_train${i}.iob
    python3 scripts/rebuild_spacy.py data/5_fold_exp/combined_train${i}.iob data/5_fold_exp/da_train${i}.spacy
done

echo "=== STEP 2: TRAINING ORGANIC BASELINES ==="
for i in {1..5}; do
    python3 -m spacy train config.cfg --output ./models/target_organic_${i} --paths.train data/5_fold_exp/organic_train${i}.spacy --paths.dev data/5_fold_exp/dev${i}.spacy --gpu-id 0
done

echo "=== STEP 3: TRAINING TARGETED DA-RoBERTa ==="
for i in {1..5}; do
    python3 -m spacy train config.cfg --output ./models/target_da_${i} --paths.train data/5_fold_exp/da_train${i}.spacy --paths.dev data/5_fold_exp/dev${i}.spacy --gpu-id 0
done

echo "=== STEP 4: EVALUATION ==="
echo "TARGETED ORGANIC RESULTS" > final_results.txt
for i in {1..5}; do
    python3 -m spacy evaluate ./models/target_organic_${i}/model-best data/5_fold_exp/test${i}.spacy --gpu-id 0 >> final_results.txt
done

echo "TARGETED DA-ROBERTA RESULTS" >> final_results.txt
for i in {1..5}; do
    python3 -m spacy evaluate ./models/target_da_${i}/model-best data/5_fold_exp/test${i}.spacy --gpu-id 0 >> final_results.txt
done

echo "ALL DONE. CHECK final_results.txt"
