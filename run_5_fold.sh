#!/bin/bash

# Configuration - update this path if your synthetic file is named differently
SYNTHETIC_DATA="data/synthetic_half.iob" 

# Ensure the synthetic file exists before starting
if [ ! -f "$SYNTHETIC_DATA" ]; then
    echo "CRITICAL ERROR: Synthetic booster pack $SYNTHETIC_DATA not found!"
    exit 1
fi

echo "=== STEP 1: CONVERTING IOB TO SPACY ==="
for i in {1..5}; do
    echo "Processing Fold $i..."
    # 1. Convert Organic Files
    python3 scripts/rebuild_spacy.py data/5_fold_exp/train${i}.iob data/5_fold_exp/organic_train${i}.spacy
    python3 scripts/rebuild_spacy.py data/5_fold_exp/dev${i}.iob data/5_fold_exp/dev${i}.spacy
    python3 scripts/rebuild_spacy.py data/5_fold_exp/test${i}.iob data/5_fold_exp/test${i}.spacy
    
    # 2. Create the Augmented IOB (Merge Organic + Synthetic)
    cat data/5_fold_exp/train${i}.iob "$SYNTHETIC_DATA" > data/5_fold_exp/combined_train${i}.iob
    
    # 3. Convert Augmented IOB to Spacy
    python3 scripts/rebuild_spacy.py data/5_fold_exp/combined_train${i}.iob data/5_fold_exp/da_train${i}.spacy

    # VERIFICATION: Stop if any file was not created
    if [ ! -f "data/5_fold_exp/organic_train${i}.spacy" ] || [ ! -f "data/5_fold_exp/da_train${i}.spacy" ]; then
        echo "ERROR: Conversion failed for Fold $i. Check rebuild_spacy.py logic."
        exit 1
    fi
done

echo "=== STEP 2: TRAINING (ESTIMATED 30-40 HOURS) ==="
# We only get here if Step 1 succeeded for ALL folds
for i in {1..5}; do
    echo "Starting Training Marathon for Fold $i..."
    # Train Organic
    # caffeinate -i python3 -m spacy train config.cfg --output ./models/5fold_organic_${i} --paths.train data/5_fold_exp/organic_train${i}.spacy --paths.dev data/5_fold_exp/dev${i}.spacy --gpu-id 0
    
    # Train DA-RoBERTa
    caffeinate -i python3 -m spacy train config.cfg --output ./models/5fold_da_${i} --paths.train data/5_fold_exp/da_train${i}.spacy --paths.dev data/5_fold_exp/dev${i}.spacy --gpu-id 0
done

echo "=== STEP 3: EVALUATION ==="
# Final evaluation logic remains the same...
