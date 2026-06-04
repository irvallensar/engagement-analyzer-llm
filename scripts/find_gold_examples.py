import spacy
from spacy.tokens import DocBin
import os

def mine_complex_gold():
    nlp = spacy.blank("en")
    # Load your actual human training fold
    file_path = "data/5_fold_exp/train1.spacy"
    
    if not os.path.exists(file_path):
        print(f"Could not find human data at {file_path}")
        return
        
    doc_bin = DocBin().from_disk(file_path)
    docs = list(doc_bin.get_docs(nlp.vocab))
    
    # Argumentative keywords we want to find in the background text
    diversity_keywords = ["although", "however", "despite", "not", "but", "while", "failed"]
    
    targets = ["ENDOPHORIC", "JUSTIFYING", "SOURCES", "CITATION"]
    
    print("=== MINING REAL COMPLEX HUMAN EXAMPLES ===")
    for target in targets:
        print(f"\n--- Real Human Candidates for {target} ---")
        count = 0
        
        for doc in docs:
            # Check if this sentence contains our target label
            has_target = any(span.label_ == target for span in doc.spans.get("sc", []))
            
            # Check if the text contains any of our background diversity words
            text_lower = doc.text.lower()
            has_diversity = any(word in text_lower for word in diversity_keywords)
            
            if has_target and has_diversity:
                # Find the target span text to format it as: Sentence | Span
                for span in doc.spans.get("sc", []):
                    if span.label_ == target:
                        print(f'"{doc.text.strip()} | {span.text.strip()}"')
                        count += 1
                        break # Just show the first matching span in the doc
                        
            if count >= 5: # Give us 5 options per class
                break

if __name__ == "__main__":
    mine_complex_gold()
