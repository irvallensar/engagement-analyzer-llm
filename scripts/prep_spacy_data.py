import spacy
from spacy.training.converters import conll_ner_to_docs
import os
import re

def clean_iob_file(filepath):
    """Strips out multiple consecutive blank lines that cause empty documents (size 0)."""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Replace 2 or more consecutive newlines with exactly 2 newlines 
    # (which equals exactly 1 blank line between sentences)
    cleaned_text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Strip leading/trailing whitespace
    cleaned_text = cleaned_text.strip() + "\n\n"
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

def combine_and_convert():
    print("1. Merging organic and synthetic data...")
    with open("data/train.iob", "r", encoding="utf-8") as f1, \
         open("data/synthetic_train.iob", "r", encoding="utf-8") as f2, \
         open("data/combined_train.iob", "w", encoding="utf-8") as out:
        out.write(f1.read().strip())
        out.write("\n\n") # Clean break
        out.write(f2.read().strip())
        out.write("\n\n")

    print("2. Sanitizing IOB files to remove empty documents...")
    clean_iob_file("data/combined_train.iob")
    
    if os.path.exists("data/valid.iob"):
        clean_iob_file("data/valid.iob")
    elif os.path.exists("data/dev.iob"):
        clean_iob_file("data/dev.iob")

    print("3. Converting to .spacy binary format...")
    nlp = spacy.blank("en")
    
    def convert_file(iob_path, spacy_path):
        if not os.path.exists(iob_path):
            return
            
        print(f"  Converting {iob_path} to {spacy_path}...")
        with open(iob_path, "r", encoding="utf-8") as f:
            file_data = f.read()
            
        docs = conll_ner_to_docs(file_data, nlp=nlp)
        
        # Double-check: filter out any docs that somehow still have 0 length
        valid_docs = [doc for doc in docs if len(doc) > 0]
        
        doc_bin = spacy.tokens.DocBin(docs=valid_docs)
        doc_bin.to_disk(spacy_path)

    convert_file("data/combined_train.iob", "data/train.spacy")
    
    if os.path.exists("data/valid.iob"):
        convert_file("data/valid.iob", "data/dev.spacy")
    elif os.path.exists("data/dev.iob"):
        convert_file("data/dev.iob", "data/dev.spacy")
    
    print("\n[SUCCESS] Data sanitized and preparation complete! Ready for RoBERTa.")

if __name__ == "__main__":
    combine_and_convert()
