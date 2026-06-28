import spacy
from spacy.tokens import DocBin

nlp = spacy.blank("en")
doc_bin = DocBin().from_disk("data/train.spacy")
docs = list(doc_bin.get_docs(nlp.vocab))

target = {"MONOGLOSS", "DENY", "PROCLAIM", "ENTERTAIN"}

candidates = []
for doc in docs:
    spans = doc.spans.get("sc", [])
    labels = set(s.label_ for s in spans)
    overlap = labels & target
    if len(overlap) >= 3:  # hits at least 3 of the 4 target labels
        candidates.append((len(overlap), len(labels), doc.text, spans, labels))

candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

for score, diversity, text, spans, labels in candidates[:5]:
    print(f"\nScore {score}/4 | Total labels: {diversity}")
    print(f"Sentence: {text[:200]}")
    for s in spans:
        print(f"  [{s.label_}] '{s.text}'")
    print(f"  Labels: {labels}")
