import spacy
from typing import List, Dict

nlp = spacy.load("en_core_web_sm")

def ngram_suggester(
    text: str,
    min_n: int = 1,
    max_n: int = 4
) -> List[Dict]:
    """
    Generate candidate spans with token indices.
    """
    doc = nlp(text)
    candidates = []
    cid = 0

    for n in range(min_n, max_n + 1):
        for i in range(len(doc) - n + 1):
            span = doc[i:i+n]
            candidates.append({
                "id": cid,
                "text": span.text,
                "start": span.start,
                "end": span.end - 1
            })
            cid += 1

    return candidates
