import spacy
from spacy.tokens import DocBin
from collections import Counter

nlp = spacy.blank("en")

for split in ["train", "dev", "test"]:
    db = DocBin().from_disk(f"data/{split}.spacy")
    docs = list(db.get_docs(nlp.vocab))
    
    counter = Counter()
    for doc in docs:
        for ent in doc.ents:
            counter[ent.label_] += 1
    
    print(f"\n=== {split.upper()} ===")
    for label, count in sorted(counter.items()):
        print(f"{label}: {count}")
