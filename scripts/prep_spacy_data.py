import spacy
from spacy.training.converters import conll_ner_to_docs

def convert_iob_to_spacy(iob_file_path, output_spacy_path):
    nlp = spacy.blank("en")
    
    with open(iob_file_path, "r", encoding="utf-8") as f:
        file_data = f.read()
        
    # Convert IOB to spaCy Doc objects
    docs = conll_ner_to_docs(file_data, nlp=nlp)
    
    # Save as binary .spacy file
    doc_bin = spacy.tokens.DocBin(docs=docs)
    doc_bin.to_disk(output_spacy_path)
    print(f"Saved {output_spacy_path}")

# Run this for your combined train file, your dev file, and your test file.
convert_iob_to_spacy("data/organic_plus_synthetic_train.iob", "data/train.spacy")
convert_iob_to_spacy("data/valid.iob", "data/dev.spacy")
convert_iob_to_spacy("data/test.iob", "data/test.spacy")
