import spacy
from spacy.tokens import DocBin
import os

def mine_generic_sources():
    nlp = spacy.blank("en")
    master_db = DocBin().from_disk("data/train.spacy")
    docs = list(master_db.get_docs(nlp.vocab))
    
    # Load the leakage blacklist
    test_blacklist = set()
    for fold in range(1, 6):
        test_path = f"data/5_fold_exp/test{fold}.spacy"
        if os.path.exists(test_path):
            test_db = DocBin().from_disk(test_path)
            for doc in test_db.get_docs(nlp.vocab):
                test_blacklist.add(doc.text.strip().lower())

    diversity_words = ["although", "however", "despite", "not", "but", "while", "failed"]
    # These force the span to be a generic group, avoiding 'he', 'his', 'it'
    generic_nouns = ["research", "studies", "scholars", "critics", "literature", "authors", "analysts", "evidence"]
    
    print("\n--- Leak-Proof Generic Candidates for SOURCES ---")
    count = 0
    for doc in docs:
        text = doc.text.strip()
        if text.lower() in test_blacklist: continue
        
        has_div = any(w in text.lower() for w in diversity_words)
        if not has_div: continue
            
        for span in doc.spans.get("sc", []):
            if span.label_ == "SOURCES":
                # Only accept the span if it contains a generic noun
                if any(noun in span.text.lower() for noun in generic_nouns):
                    # Clean DOCSTART for output
                    clean_text = text.replace("-DOCSTART-", "").strip()
                    print(f'"{clean_text} | {span.text.strip()}"')
                    count += 1
                    break
        if count >= 3:
            break

if __name__ == "__main__":
    mine_generic_sources()
