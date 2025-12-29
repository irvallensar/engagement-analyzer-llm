# Custom candidate span suggester

import spacy
import string

class CandidateSuggester:
    def __init__(self, nlp, max_width=6):
        self.nlp = nlp
        self.max_width = max_width

    def get_candidates(self, text):
        doc = self.nlp(text)
        candidates = []
        cid = 0

        for start in range(len(doc)):
            for end in range(start + 1, min(start + self.max_width + 1, len(doc) + 1)):
                span = doc[start:end]

                # -------- FILTERS --------
                if not has_min_length(span, min_tokens=2):
                    continue

                if is_punctuation_only(span):
                    continue

                if is_all_stopwords(span):
                    continue
                # -------------------------

                candidates.append({
                    "id": cid,
                    "text": span.text,
                    "start_token": span.start,
                    "end_token": span.end - 1,
                })
                cid += 1

        return candidates
