import spacy

class CandidateSuggester:
    def __init__(self, nlp, max_width=6):
        self.nlp = nlp
        self.max_width = max_width

    def filter_contained_candidates(self, candidates):
        """
        Simplification Rule: If we have "It is often believed", 
        we do not need "is often believed" or "believed".
        Keep the longest spans; drop spans that are strictly inside another.
        """
        # Sort by length (longest first)
        candidates.sort(key=lambda x: (x["end_token"] - x["start_token"]), reverse=True)
        
        final_list = []
        for c in candidates:
            is_inside = False
            for existing in final_list:
                # Check if 'c' is inside 'existing'
                if (c["start_token"] >= existing["start_token"] and 
                    c["end_token"] <= existing["end_token"]):
                    is_inside = True
                    break
            
            if not is_inside:
                final_list.append(c)
        
        # Sort back by position for the LLM
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
        # Invalid start/end tokens (Garbage filters)
        BAD_START = {"DET", "SCONJ", "CCONJ", "PUNCT", "PART", "ADV"} # Added ADV to stop "often believed"
        BAD_END =   {"DET", "SCONJ", "CCONJ", "PRON", "PUNCT", "ADP"}

        for start in range(len(doc)):
            for end in range(start + 1, min(start + self.max_width + 1, len(doc) + 1)):
                span = doc[start:end]
                
                # 1. Length check
                if len(span) > 6: continue

                # 2. Basic content check
                if all(t.is_punct for t in span): continue
                if all(t.is_stop for t in span): continue

                # 3. Clean Boundaries (The simplified "Bad Start/End" rule)
                if span[0].pos_ in BAD_START and span[0].text.lower() != "it": # Exception for "It"
                    continue
                if span[-1].pos_ in BAD_END:
                    continue

                # 4. Filter contentless Noun Phrases (e.g., "the language")
                if span.root.pos_ in {"NOUN", "PRON"} and not any(t.pos_ == "VERB" for t in span):
                    continue

                add({
                    "id": f"{span.start}-{span.end}",
                    "text": span.text,
                    "start_token": span.start,
                    "end_token": span.end,
                })

        # --- STEP 3: Deduplicate ---
        # This removes the "messy" overlaps automatically
        clean_candidates = self.filter_contained_candidates(raw_candidates)
        
        return clean_candidates
