import spacy
from spacy.training.converters import conll_ner_to_docs
import os

def combine_and_convert():
    print("1. Merging organic and synthetic data...")
    # Make sure these filenames match exactly what is in your data folder!
    with open("data/train.iob", "r", encoding="utf-8") as f1, \
         open("data/synthetic_train.iob", "r", encoding="utf-8") as f2, \
         open("data/combined_train.iob", "w", encoding="utf-8") as out:
        out.write(f1.read())
        out.write("\n\n") # Clean break between datasets
        out.write(f2.read())

    print("2. Converting to .spacy binary format...")
    nlp = spacy.blank("en")
    
    def convert_file(iob_path, spacy_path):
        if not os.path.exists(iob_path):
            print(f"  [SKIPPED] Could not find {iob_path}")
            return
            
        print(f"  Converting {iob_path} to {spacy_path}...")
        with open(iob_path, "r", encoding="utf-8") as f:
            file_data = f.read()
            
        docs = conll_ner_to_docs(file_data, nlp=nlp)
        doc_bin = spacy.tokens.DocBin(docs=docs)
        doc_bin.to_disk(spacy_path)

    # Convert the new massive combined training set
    convert_file("data/combined_train.iob", "data/train.spacy")
    
    # Convert your validation set (Eguchi & Kyle usually named this valid.iob or dev.iob)
    # If yours is named differently, change "data/valid.iob" below to match!
    if os.path.exists("data/valid.iob"):
        convert_file("data/valid.iob", "data/dev.spacy")
    elif os.path.exists("data/dev.iob"):
        convert_file("data/dev.iob", "data/dev.spacy")
    
    print("\n[SUCCESS] Data preparation complete! Ready for RoBERTa.")

if __name__ == "__main__":
    combine_and_convert()
