#!/bin/bash

echo "=== STARTING RESULTS ==="
echo "ORGANIC RESULTS" > 5fold_results.txt

for i in {1..5}; do
    echo "Grading Organic Fold $i..."
    python3 -m spacy evaluate ./models/5fold_organic_${i}/model-best data/5_fold_exp/test${i}.spacy --gpu-id 0 >> 5fold_results.txt
done

echo "" >> 5fold_results.txt
echo "DA-ROBERTA RESULTS" >> 5fold_results.txt

for i in {1..5}; do
    echo "Grading DA-RoBERTa Fold $i..."
    python3 -m spacy evaluate ./models/5fold_da_${i}/model-best data/5_fold_exp/test${i}.spacy --gpu-id 0 >> 5fold_results.txt
done

echo "=== ALL DONE! ==="
