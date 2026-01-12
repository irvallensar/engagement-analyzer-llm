# Custom candidate span suggester

import spacy
import string

# ---------- helper functions ----------
def is_punctuation_only(span):
    return all(token.is_punct for token in span)

def is_all_stopwords(span):
    return all(token.is_stop for token in span)

def contains_finite_verb(span):
    return any(
        tok.pos_ == "VERB" and tok.morph.get("VerbForm") == ["Fin"]
        for tok in span
    )
    
def is_contentless_np(span):
    return (
        span.root.pos_ in {"NOUN", "PRON"} and
        not any(tok.pos_ == "VERB" for tok in span)
    )
# -------------------------------------

seen = set()

def add_candidate(c, candidates):
    if c["id"] not in seen:
        candidates.append(c)
        seen.add(c["id"])
        
class CandidateSuggester:
    def __init__(self, nlp, max_width=6):
        self.nlp = nlp
        self.max_width = max_width

    def get_candidates(self, text):
        doc = self.nlp(text)
        candidates = []
        cid = 0

        # ==================================================
        # STEP 1: High-priority atomic candidates (VERBS)
        # ==================================================
        for token in doc:
            # Keep this as is
            if token.pos_ == "VERB" and token.morph.get("VerbForm") == ["Fin"]:
                candidates.append({
                    "id": f"{token.i}-{token.i+1}",
                    "text": token.text,
                    "start_token": token.i,
                    "end_token": token.i + 1,
                })
                cid += 1

        # ==================================================
        # STEP 2: Phrase candidates
        # ==================================================
        for start in range(len(doc)):
            for end in range(start + 1, min(start + self.max_width + 1, len(doc) + 1)):
                span = doc[start:end]

                # ---- HARD FILTERS ----
                if is_punctuation_only(span):
                    continue

                if is_all_stopwords(span):
                    continue

                # REMOVED: The check that blocks finite verbs in phrases
                # if len(span) > 1 and contains_finite_verb(span):
                #    continue

                # ❌ Drop long clause-like spans (Keep this to prevent entire sentences)
                if len(span) > 5: # Increased slightly to allow "It is often believed that"
                    continue

                # ❌ Remove bare noun phrases (Keep this)
                if is_contentless_np(span):
                    continue
                    
                add_candidate({
                    "id": f"{span.start}-{span.end}",
                    "text": span.text,
                    "start_token": span.start,
                    "end_token": span.end,
                }, candidates)
                cid += 1

        return candidates
