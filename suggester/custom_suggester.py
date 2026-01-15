import spacy

class CandidateSuggester:
    def __init__(self, nlp, max_width=5):  # CHANGED: Reduced max_width from 6 to 5
        self.nlp = nlp
        self.max_width = max_width

    def filter_contained_candidates(self, candidates):
        """
        Deduplication Rule:
        If we have "It is often believed" (0-4), we remove "believed" (3-4).
        """
        # Sort by length (longest first)
        candidates.sort(key=lambda x: (x["end_token"] - x["start_token"]), reverse=True)
        
        final_list = []
        for c in candidates:
            is_inside = False
            for existing in final_list:
                # Check if 'c' is strictly inside 'existing'
                if (c["start_token"] >= existing["start_token"] and 
                    c["end_token"] <= existing["end_token"] and
                    c["id"] != existing["id"]):
                    is_inside = True
                    break
            
            if not is_inside:
                final_list.append(c)
        
        # Sort back by position
        final_list.sort(key=lambda x: x["start_token"])
        return final_list

    def get_candidates(self, text):
        doc = self.nlp(text)
        raw_candidates = []
        seen_ids = set()

        def add(c):
            if c["id"] not in seen_ids:
                raw_candidates.append(c)
                seen_ids.add(c["id"])

        # --- STEP 1: Single Verbs ---
        for token in doc:
            if token.pos_ == "VERB" and token.morph.get("VerbForm") == ["Fin"]:
                add({
                    "id": f"{token.i}-{token.i+1}",
                    "text": token.text,
                    "start_token": token.i,
                    "end_token": token.i + 1,
                })

        # --- STEP 2: Phrases ---
        # Strictly Forbidden Start/End POS tags
        BAD_START = {"DET", "SCONJ", "CCONJ", "PUNCT", "PART", "ADV", "ADP"} 
        BAD_END =   {"DET", "SCONJ", "CCONJ", "PRON", "PUNCT", "ADP"}

        for start in range(len(doc)):
            # Loop up to max_width (default 5)
            for end in range(start + 1, min(start + self.max_width + 1, len(doc) + 1)):
                span = doc[start:end]
                
                # 1. Skip if span contains 'that', 'if', 'whether' (SCONJ)
                # Stance markers usually precede 'that', they don't include it.
                if any(t.pos_ == "SCONJ" for t in span):
                    continue

                # 2. Basic content filters
                if all(t.is_punct for t in span): continue
                if all(t.is_stop for t in span): continue

                # 3. Strict Boundary Checks
                # Exception: Allow "It" (PRON) to start a span (e.g. "It is...")
                if span[0].pos_ in BAD_START and span[0].text.lower() != "it":
                    continue
                if span[-1].pos_ in BAD_END:
                    continue

                # 4. Remove contentless Noun Phrases (e.g. "the language")
                if span.root.pos_ in {"NOUN", "PRON"} and not any(t.pos_ in {"VERB", "AUX"} for t in span):
                    continue

                add({
                    "id": f"{span.start}-{span.end}",
                    "text": span.text,
                    "start_token": span.start,
                    "end_token": span.end,
                })

        # --- STEP 3: Deduplicate ---
        clean_candidates = self.filter_contained_candidates(raw_candidates)
        
        return clean_candidates
