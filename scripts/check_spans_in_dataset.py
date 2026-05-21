import spacy
from spacy.tokens import DocBin
from collections import defaultdict

nlp = spacy.blank("en")
db = DocBin().from_disk("data/train.spacy")
docs = list(db.get_docs(nlp.vocab))

overlapping = 0
nested = 0
multiple = 0
total_sentences = len(docs)

for doc in docs:
    ents = sorted(doc.ents, key=lambda e: e.start)
    
    if len(ents) > 1:
        multiple += 1
    
    has_overlap = False
    has_nested = False
    
    for i in range(len(ents)):
        for j in range(i + 1, len(ents)):
            a, b = ents[i], ents[j]
            
            # Nested: b is completely inside a
            if a.start <= b.start and b.end <= a.end:
                has_nested = True
                
            # Overlapping: partial overlap (not nested)
            elif a.start < b.end and b.start < a.end:
                has_overlap = True
    
    if has_overlap:
        overlapping += 1
    if has_nested:
        nested += 1

print(f"Total sentences:              {total_sentences}")
print(f"Sentences with multiple spans: {multiple}")
print(f"Sentences with nested spans:   {nested}")
print(f"Sentences with overlapping spans: {overlapping}")
