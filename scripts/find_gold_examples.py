import spacy
from spacy.tokens import DocBin
import os

def mine_abundant_gold():
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

    diversity_words = ["although", "however", "despite", "not", "but", "while", "failed", "even though"]
    targets = ["ENDOPHORIC", "JUSTIFYING", "SOURCES", "CITATION"]
    
    # Specific keywords to force GOOD SOURCES and ban BAD SOURCES
    good_source_nouns = ["research", "studies", "scholars", "critics", "literature", "analysts", "evidence", "reports", "findings"]
    bad_source_words = ["his", "her", "he", "she", "their", "my", "our", "the authors"]

    print("\n=== MINING 15 UNIQUE LEAK-PROOF CANDIDATES PER CLASS ===")
    
    for target in targets:
        print(f"\n--- Candidates for {target} ---")
        seen_texts = set()
        count = 0
        
        for doc in docs:
            # Clean text right away
            text = doc.text.replace("-DOCSTART-", "").strip()
            text_lower = text.lower()
            
            # 1. Deduplication and Leakage Check
            if text_lower in test_blacklist or text_lower in seen_texts:
                continue
                
            # 2. Diversity Check (Must contain background argumentation)
            has_div = any(w in text_lower for w in diversity_words)
            if not has_div:
                continue
                
            # 3. Target Extraction
            for span in doc.spans.get("sc", []):
                if span.label_ == target:
                    span_lower = span.text.lower()
                    
                    # Extra filtering specifically for SOURCES
                    if target == "SOURCES":
                        if not any(noun in span_lower for noun in good_source_nouns):
                            continue
                        # Ensure no bad pronouns exist as standalone words in the span
                        if any(bad in span_lower.split() for bad in bad_source_words):
                            continue
                    
                    print(f'{count + 1}. "{text} | {span.text.strip()}"')
                    seen_texts.add(text_lower)
                    count += 1
                    break # Move to next doc once we find a valid span
                    
            if count >= 15:
                break

if __name__ == "__main__":
    mine_abundant_gold()
