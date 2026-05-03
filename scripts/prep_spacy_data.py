import spacy
from spacy.training.converters import conll_ner_to_docs

def combine_and_convert():
    # 1. Merge organic and synthetic IOB files
    print("Merging organic and synthetic data...")
    with open("data/train.iob", "r", encoding="utf-8") as f1, \
         open("data/synthetic_train.iob", "r", encoding="utf-8") as f2, \
         open("data/combined_train.iob", "w", encoding="utf-8") as out:
        out.write(f1.read())
        out.write("\n") # Ensure a clean break
        out.write(f2.read())

    # 2. Convert to .spacy format
    nlp = spacy.blank("en")
    
    def convert_file(iob_path, spacy_path):
        print(f"Converting {iob_path} to {spacy_path}...")
        with open(iob_path, "r", encoding="utf-8") as f:
            file_data = f.read()
        docs = conll_ner_to_docs(file_data, nlp=nlp)
        doc_bin = spacy.tokens.DocBin(docs=docs)
        doc_bin.to_disk(spacy_path)

    convert_file("data/combined_train.iob", "data/train.spacy")
    convert_file("data/valid.iob", "data/dev.spacy")
    convert_file("data/test.iob", "data/test.spacy") # If you have a test set
    
    print("\nData preparation complete! Ready for RoBERTa.")

if __name__ == "__main__":
    combine_and_convert()
