import spacy

class CandidateSuggester:
    def __init__(self, nlp, max_width=5):
        self.nlp = nlp
        self.max_width = max_width

    def filter_contained_candidates(self, candidates):
        """
        Deduplication Rule:
        Removes smaller candidates that are strictly contained inside larger ones.
        e.g., removes "believed" if "It is often believed" exists.
        """
        # Sort by length (longest first)
        candidates.sort(key=lambda x: (x["end_token"] - x["start_token"]), reverse=True)
        
        final_list = []
        for c in candidates:
            is_inside = False
            for existing in final_list:
                # Check if 'c' is strictly inside 'existing'
                # (and not the exact same span)
                if (c["start_token"] >= existing["start_token"] and 
                    c["end_token"] <= existing["end_token"] and
                    c["id"] != existing["id"]):
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

        # STEP 1: Atomic Verbs
        for token in doc:
            if token.pos_ == "VERB" and token.morph.get("VerbForm") == ["Fin"]:
                add({
                    "id": f"{token.i}-{token.i+1}",
                    "text": token.text,
                    "start_token": token.i,
                    "end_token": token.i + 1,
                })

        # STEP 2: Phrases
        # 1. Useless Starts (Function words)
        BAD_START_ALWAYS = {"DET", "SCONJ", "CCONJ", "PUNCT", "PART", "ADP"}
        
        # 2. Useless Ends
        BAD_END = {"DET", "SCONJ", "CCONJ", "PRON", "PUNCT", "ADP"}

        for start in range(len(doc)):
            for end in range(start + 1, min(start + self.max_width + 1, len(doc) + 1)):
                span = doc[start:end]
                
                # FILTER 1: Strict Inner Content
                # Stance markers rarely contain sub-conjunctions like 'that'
                if any(t.pos_ == "SCONJ" for t in span):
                    continue

                # FILTER 2: Start Tokens
                first_pos = span[0].pos_
                
                # Reject if starts with function word (e.g. "The", "That")
                # Exception: "It" is allowed (PRON)
                if first_pos in BAD_START_ALWAYS:
                    continue
                
                # Reject Multi-word spans starting with NOUN/PROPN
                # This removes "language you speak..." (Starts with NOUN, len > 1)
                # But keeps "possibility" (Starts with NOUN, len == 1)
                if len(span) > 1 and first_pos in {"NOUN", "PROPN"}:
                    continue

                # FILTER 3: End Tokens 
                if span[-1].pos_ in BAD_END:
                    continue

                # FILTER 4: Contentless NPs 
                # e.g. "the language" -> Root is NOUN, no verb.
                if span.root.pos_ in {"NOUN", "PRON"} and not any(t.pos_ in {"VERB", "AUX", "ADJ", "ADV"} for t in span):
                    continue

                # FILTER 5: Basic Noise 
                if all(t.is_punct for t in span): continue
                if all(t.is_stop for t in span): continue

                add({
                    "id": f"{span.start}-{span.end}",
                    "text": span.text,
                    "start_token": span.start,
                    "end_token": span.end,
                })

        # STEP 3: Deduplicate
        clean_candidates = self.filter_contained_candidates(raw_candidates)
        
        return clean_candidates
