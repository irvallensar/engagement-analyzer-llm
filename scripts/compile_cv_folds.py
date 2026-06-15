import spacy
from spacy.tokens import DocBin, Span
import os

# Create a blank English vocabulary used when reconstructing Docs
# from IOB files and DocBins.
nlp = spacy.blank("en")

def parse_multicolumn_iob(input_path):
    # Converts a multi-column IOB file into a spaCy DocBin.
    # Each annotation column after the token column is treated
    # as an independent BIO annotation layer.
    db = DocBin()

    with open(input_path, 'r', encoding='utf-8') as f:
        # Sentences/documents are separated by blank lines.
        blocks = f.read().strip().split('\n\n')
        
    # Process one sentence/document block at a time.
    for block in blocks:
        lines = block.split('\n')

        # Skip empty blocks if encountered.
        if not lines: continue
        # Column 0 contains the token text.
        words = [line.split()[0] for line in lines if line.strip()]
        # Reconstruct a spaCy Doc from tokenized words.
        doc = spacy.tokens.Doc(nlp.vocab, words=words)
        # Collect spans from all annotation columns.
        spans_list = []
        
        # Determine how many annotation columns exist in this block.
        max_cols = max(len(line.split()) for line in lines)
        # Column 0 is the token itself, so annotation columns begin at 1.
        for col_idx in range(1, max_cols):
            start = None
            label = None
            # Walk through tokens and rebuild entity spans
            # from BIO tags.
            for i, line in enumerate(lines):
                parts = line.split()
                # Missing annotation columns are treated as outside ("0").
                tag = parts[col_idx] if col_idx < len(parts) else "0"
                # Start of a new entity.
                if tag.startswith("B-"):
                    if start is not None:
                        spans_list.append(Span(doc, start, i, label=label))
                    start = i
                    label = tag[2:]
                # Continue current entity if label matches.
                elif tag.startswith("I-") and start is not None and tag[2:] == label:
                    pass
                # Any non-matching tag closes the current entity.
                else:
                    if start is not None:
                        spans_list.append(Span(doc, start, i, label=label))
                        start = None
                        label = None
            # Handle entities that extend to the final token.
            if start is not None:
                spans_list.append(Span(doc, start, len(lines), label=label))
        # Store all recovered spans in the spancat key.
        doc.spans["sc"] = spans_list
        # Add processed document to DocBin.
        db.add(doc)
    # Return compiled dataset.
    return db

def main():
    # Load the 8,000 synthetic sentences once
    print("Loading synthetic data...")
    # Synthetic documents are reused for every fold's
    # augmented training dataset.
    synthetic_db = DocBin().from_disk("data/synthetic_pseudo_labeled_few_shot_v3.spacy")
    synthetic_docs = list(synthetic_db.get_docs(nlp.vocab))

    print(f"Loaded {len(synthetic_docs)} synthetic sentences.\n")
    # Build all 5 cross-validation folds.
    for fold in range(1, 6):
        print(f"=== Processing Fold {fold} ===")
        
        # 1. Compile Dev and Test directly
        # Convert dev fold from IOB format into spaCy binary format.
        print(f"  Compiling dev{fold}.iob...")
        dev_db = parse_multicolumn_iob(f"data/5_fold_exp/dev{fold}.iob")
        dev_db.to_disk(f"data/5_fold_exp/dev{fold}.spacy")
        
        # Convert test fold from IOB format into spaCy binary format.
        print(f"  Compiling test{fold}.iob...")
        test_db = parse_multicolumn_iob(f"data/5_fold_exp/test{fold}.iob")
        test_db.to_disk(f"data/5_fold_exp/test{fold}.spacy")
        
        # 2. Compile Baseline Train
        # Baseline training set contains only human annotations.
        print(f"  Compiling train{fold}.iob (Baseline)...")
        train_db = parse_multicolumn_iob(f"data/5_fold_exp/train{fold}.iob")
        train_db.to_disk(f"data/5_fold_exp/train{fold}.spacy")
        
        # 3. Create Augmented Train (DA-Train)

        # NOTE:
        # DA = Data Augmentation.
        # This dataset combines human training examples
        # with synthetic pseudo-labeled examples.
        print(f"  Injecting synthetic data into da_train{fold}.spacy...")
        da_train_db = DocBin()
        
        # Add human train docs
        # First copy all original human training documents.
        for doc in train_db.get_docs(nlp.vocab):
            da_train_db.add(doc)

        # Add synthetic docs
        # Then append all synthetic documents.
        for doc in synthetic_docs:
            da_train_db.add(doc)
        # Save augmented training set for this fold.
        da_train_db.to_disk(f"data/5_fold_exp/da_train{fold}.spacy")
        print(f"  Fold {fold} complete.\n")

    # At this point all baseline and augmented datasets
    # have been generated for every fold.
    print("[SUCCESS] All 5 folds compiled and augmented!")

if __name__ == "__main__":
    main()
