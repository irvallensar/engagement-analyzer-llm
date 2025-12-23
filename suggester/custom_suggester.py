# Custom candidate span suggester

import spacy

class CandidateSuggester:
    def __init__(self, nlp):
        self.nlp = nlp

    def get_candidates(self, text):
        doc = self.nlp(text)
        candidates = []

        # simple ngram-based spans (2–5 tokens)
        for i in range(len(doc)):
            for j in range(i + 1, min(i + 6, len(doc) + 1)):
                span = doc[i:j]
                candidates.append({
                    "text": span.text,
                    "start_token": span.start,
                    "end_token": span.end - 1,
                    "start_char": span.start_char,
                    "end_char": span.end_char
                })

        return candidates
