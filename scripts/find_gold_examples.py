import spacy
from spacy.tokens import DocBin

nlp = spacy.blank("en")
doc_bin = DocBin().from_disk("data/train.spacy")
docs = list(doc_bin.get_docs(nlp.vocab))

# Find sentences with 3+ distinct labels — ideal for few-shot
good_examples = []
for doc in docs:
    spans = doc.spans.get("sc", [])
    labels = set(s.label_ for s in spans)
    if len(labels) >= 3:
        good_examples.append((doc.text, spans, labels))

# Sort by label diversity
good_examples.sort(key=lambda x: len(x[2]), reverse=True)

# Print top candidates
for text, spans, labels in good_examples[:20]:
    print(f"\nSentence: {text}")
    for s in spans:
        print(f"  [{s.label_}] '{s.text}'")
    print(f"  Labels: {labels}")
