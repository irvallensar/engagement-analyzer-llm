import spacy
from spacy.tokens import DocBin
import os

def mine_leak_proof_gold():
    nlp = spacy.blank("en")
    master_train_path = "data/train.spacy"
    fold_dir = "data/5_fold_exp"
    
    if not os.path.exists(master_train_path):
        print(f"Could not find master human data at {master_train_path}")
        return
        
    # 1. Load all test set texts into a blacklist set to guarantee zero data leakage
    test_blacklist = set()
    for fold in range(1, 6):
        test_path = f"{fold_dir}/test{fold}.spacy"
        if os.path.exists(test_path):
            test_db = DocBin().from_disk(test_path)
            for doc in test_db.get_docs(nlp.vocab):
                test_blacklist.add(doc.text.strip().lower())
                
    print(f"Loaded {len(test_blacklist)} test sentences into the leakage blacklist.")
    
    # 2. Open the master train file
    master_db = DocBin().from_disk(master_train_path)
    docs = list(master_db.get_docs(nlp.vocab))
    
    diversity_keywords = ["although", "however", "despite", "not", "but", "while", "failed"]
    targets = ["ENDOPHORIC", "JUSTIFYING", "SOURCES", "CITATION"]
    
    print("\n=== MINING LEAK-PROOF COMPLEX HUMAN EXAMPLES ===")
    for target in targets:
        print(f"\n--- Leak-Proof Candidates for {target} ---")
        count = 0
        
        for doc in docs:
            cleaned_text = doc.text.strip()
            text_lower = cleaned_text.lower()
            
            # Skip if the sentence leaks into any test fold
            if text_lower in test_blacklist:
                continue
                
            has_target = any(span.label_ == target for span in doc.spans.get("sc", []))
            has_diversity = any(word in text_lower for word in diversity_keywords)
            
            if has_target and has_diversity:
                for span in doc.spans.get("sc", []):
                    if span.label_ == target:
                        print(f'"{cleaned_text} | {span.text.strip()}"')
                        count += 1
                        break
                        
            if count >= 3: # We only need 3 solid examples per class
                break

if __name__ == "__main__":
    mine_leak_proof_gold()
