import spacy
from spacy.tokens import DocBin, Span
import os

nlp = spacy.blank("en")

def parse_multicolumn_iob(input_path):
    db = DocBin()
    with open(input_path, 'r', encoding='utf-8') as f:
        blocks = f.read().strip().split('\n\n')
        
    for block in blocks:
        lines = block.split('\n')
        if not lines: continue
        
        words = [line.split()[0] for line in lines if line.strip()]
        doc = spacy.tokens.Doc(nlp.vocab, words=words)
        spans_list = []
        
        max_cols = max(len(line.split()) for line in lines)
        
        for col_idx in range(1, max_cols):
            start = None
            label = None
            for i, line in enumerate(lines):
                parts = line.split()
                tag = parts[col_idx] if col_idx < len(parts) else "0"
                
                if tag.startswith("B-"):
                    if start is not None:
                        spans_list.append(Span(doc, start, i, label=label))
                    start = i
                    label = tag[2:]
                elif tag.startswith("I-") and start is not None and tag[2:] == label:
                    pass
                else:
                    if start is not None:
                        spans_list.append(Span(doc, start, i, label=label))
                        start = None
                        label = None
            if start is not None:
                spans_list.append(Span(doc, start, len(lines), label=label))
                
        doc.spans["sc"] = spans_list
        db.add(doc)
    return db

def main():
    # Load the 8,000 synthetic sentences once
    print("Loading synthetic data...")
    synthetic_db = DocBin().from_disk("data/synthetic_pseudo_labeled_few_shot.spacy")
    synthetic_docs = list(synthetic_db.get_docs(nlp.vocab))
    print(f"Loaded {len(synthetic_docs)} synthetic sentences.\n")

    for fold in range(1, 6):
        print(f"=== Processing Fold {fold} ===")
        
        # 1. Compile Dev and Test directly
        print(f"  Compiling dev{fold}.iob...")
        dev_db = parse_multicolumn_iob(f"data/5_fold_exp/dev{fold}.iob")
        dev_db.to_disk(f"data/5_fold_exp/dev{fold}.spacy")
        
        print(f"  Compiling test{fold}.iob...")
        test_db = parse_multicolumn_iob(f"data/5_fold_exp/test{fold}.iob")
        test_db.to_disk(f"data/5_fold_exp/test{fold}.spacy")
        
        # 2. Compile Baseline Train
        print(f"  Compiling train{fold}.iob (Baseline)...")
        train_db = parse_multicolumn_iob(f"data/5_fold_exp/train{fold}.iob")
        train_db.to_disk(f"data/5_fold_exp/train{fold}.spacy")
        
        # 3. Create Augmented Train (DA-Train)
        print(f"  Injecting synthetic data into da_train{fold}.spacy...")
        da_train_db = DocBin()
        
        # Add human train docs
        for doc in train_db.get_docs(nlp.vocab):
            da_train_db.add(doc)
        # Add synthetic docs
        for doc in synthetic_docs:
            da_train_db.add(doc)
            
        da_train_db.to_disk(f"data/5_fold_exp/da_train{fold}.spacy")
        print(f"  Fold {fold} complete.\n")

    print("[SUCCESS] All 5 folds compiled and augmented!")

if __name__ == "__main__":
    main()
