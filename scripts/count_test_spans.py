import spacy
from spacy.tokens import DocBin
from collections import Counter
import os

def main():
    nlp = spacy.blank("en")
    grand_total_counts = Counter()

    print("\n" + "="*40)
    print("      TEST SET SPAN COUNTS")
    print("="*40)

    # Loop through all 5 test folds
    for fold in range(1, 6):
        file_path = f"data/5_fold_exp/test{fold}.spacy"

        if not os.path.exists(file_path):
            print(f"[WARNING] {file_path} not found.")
            continue

        doc_bin = DocBin().from_disk(file_path)
        docs = list(doc_bin.get_docs(nlp.vocab))

        fold_counts = Counter()

        for doc in docs:
            # Tally every span in the 'sc' (spancat) component
            for span in doc.spans.get("sc", []):
                fold_counts[span.label_] += 1
                grand_total_counts[span.label_] += 1

        print(f"\n--- {file_path} ---")
        for label, count in fold_counts.most_common():
            print(f"{label}: {count}")

    print("\n" + "="*40)
    print("  GRAND TOTAL SUPPORT (ALL 5 FOLDS)")
    print("="*40)
    for label, count in grand_total_counts.most_common():
        print(f"{label}: {count}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
