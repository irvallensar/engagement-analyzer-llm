import spacy
from spacy.tokens import DocBin
import os

def clean_file(filename):
    if not os.path.exists(filename):
        return
    
    print(f"Cleaning {filename}...")
    nlp = spacy.blank("en")
    doc_bin = DocBin().from_disk(filename)
    docs = list(doc_bin.get_docs(nlp.vocab))
    
    # KEEP ONLY docs that have actual text/letters in them
    clean_docs = [doc for doc in docs if len(doc.text.strip()) > 0]
    
    DocBin(docs=clean_docs).to_disk(filename)
    print(f"  Removed {len(docs) - len(clean_docs)} ghost documents.")

clean_file("data/train.spacy")
clean_file("data/dev.spacy")
print("All files sanitized!")
