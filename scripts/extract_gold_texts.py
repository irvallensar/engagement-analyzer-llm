import spacy
from spacy.tokens import DocBin

# Replace this with the actual paths to your Gold Standard .spacy files
# If you used all 5 folds for your Gold Standard, list all 5 test/train files here.
gold_spacy_files = [
    "data/5_fold_exp/test1.spacy",
    "data/5_fold_exp/test2.spacy",
    "data/5_fold_exp/test3.spacy",
    "data/5_fold_exp/test4.spacy",
    "data/5_fold_exp/test5.spacy"
]

gold_texts = set()
nlp = spacy.blank("en")

print("Extracting Gold Standard texts to prevent data leakage...")

for file_path in gold_spacy_files:
    try:
        doc_bin = DocBin().from_disk(file_path)
        docs = list(doc_bin.get_docs(nlp.vocab))
        for doc in docs:
            # Clean up whitespace and save the exact text
            gold_texts.add(doc.text.strip())
        print(f"Loaded {len(docs)} sentences from {file_path}")
    except Exception as e:
        print(f"Skipping {file_path} (Not found or error: {e})")

# Export to a simple text file
output_file = "gold_texts.txt"
with open(output_file, "w", encoding="utf-8") as f:
    for text in gold_texts:
        f.write(text + "\n")

print(f"\nSuccessfully saved {len(gold_texts)} unique Gold sentences to {output_file}.")
