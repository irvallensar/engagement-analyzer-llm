import spacy

class CandidateSuggester:
    # 1. Initialization
    def __init__(self, nlp, max_width=5):
        self.nlp = nlp              # Stores the spaCy model (passed from run_evaluation.py)
        self.max_width = max_width  # Sets the max phrase length (default 5 words)

    # 2. Deduplication Logic
    def filter_contained_candidates(self, candidates):
        """
        Removes 'small' candidates that are inside 'big' ones.
        Goal: To remove smaller phrases that are inside bigger phrases (like removing "believed" if "It is believed" already exists).
        This was originally to solve the "It is believed..." sentence with tow "believes" in that same sentence.
        """
        # Sort candidates by LENGTH (Longest first).
        # lambda x: (end - start) calculates the length.
        # reverse=True means prioritize big spans first, then reject the small ones later.
        candidates.sort(key=lambda x: (x["end_token"] - x["start_token"]), reverse=True)
        
        final_list = []
        for c in candidates:
            is_inside = False
            for existing in final_list:
                # Check if current candidate 'c' fits strictly inside 'existing'.
                # c.start >= existing.start AND c.end <= existing.end
                if (c["start_token"] >= existing["start_token"] and # Does c start at or after the bigger span starts?
                    c["end_token"] <= existing["end_token"] and # Does c end at or before the bigger span ends?
                    c["id"] != existing["id"]): # Ensure the span don't compare to itself
                    is_inside = True
                    break # A parent is found (bigger than c). Reject c. 
                    # e.g. "It is believed" = parent ; "believed" = child ; meaning the "believed" inside the "it is believed" is ignored.

            
            # If no parent found, keep this candidate.
            if not is_inside:
                final_list.append(c)
        
        # Sort back by POSITION (Start Token 0, 1, 2...) so the LLM reads them in order.
        final_list.sort(key=lambda x: x["start_token"])
        return final_list

    # 3. The Main Candidate Generation Function
    def get_candidates(self, text):
        doc = self.nlp(text) # Run spaCy pipeline (POS tagging, etc.)
        raw_candidates = []
        seen_ids = set()     # A set to prevent duplicate IDs. A memory of which the candidates are already added.

        # Helper function to append candidates safely
        def add(c):
            if c["id"] not in seen_ids:
                raw_candidates.append(c)
                seen_ids.add(c["id"])

        # STEP 1: Atomic Verbs (Single words)
        # Catches single verbs like "demonstrates" or "implies".
        for token in doc:
            # Check if POS is VERB and it is FINITE (has tense, e.g., "runs" vs "running").
            if token.pos_ == "VERB" and token.morph.get("VerbForm") == ["Fin"]:
                add({
                    "id": f"{token.i}-{token.i+1}", # ID Format: "Start-End"
                    "text": token.text,
                    "start_token": token.i,
                    "end_token": token.i + 1,
                })

        # STEP 2: Phrases (Multi-words)
        # Definitions of "Bad" starting/ending POS tags
        BAD_START_ALWAYS = {"DET", "SCONJ", "CCONJ", "PUNCT", "PART", "ADP"} # POS tags that are not allowed at the start of a phrase.
        BAD_END = {"DET", "SCONJ", "CCONJ", "PRON", "PUNCT", "ADP"} #unwanted POS tags that produce incomplete information

        # Sliding Window Loop
        for start in range(len(doc)):
            # Inner loop goes from start+1 up to max_width
            for end in range(start + 1, min(start + self.max_width + 1, len(doc) + 1)):
                span = doc[start:end] # The actual phrase being tested
                
                # FILTER 1: No sub-conjunctions inside (e.g., rejects "believed that the")
                if any(t.pos_ == "SCONJ" for t in span): #t.pos_ is the part-of-speech tag of a token, used to...                              
                    continue                             #...identify its grammatical role such as noun, verb, or conjunction.

                # FILTER 2: Start Token Rules
                first_pos = span[0].pos_
                
                # Reject if starts with bad words (e.g., "The", "And")
                if first_pos in BAD_START_ALWAYS:
                    continue
                
                # Reject Noun-starts IF length > 1 (e.g., rejects "Language determines")
                # But allows Noun-starts if length == 1 (e.g., allows "Possibility")
                if len(span) > 1 and first_pos in {"NOUN", "PROPN"}:
                    continue

                # FILTER 3: End Token Rules (Don't end with "the" or "of")
                if span[-1].pos_ in BAD_END:
                    continue

                # FILTER 4: Contentless Noun Phrases
                # If root is Noun, it MUST contain a Verb/Adj/Adv.
                # Rejects: "the language" (Noun root, no modifiers).
                if span.root.pos_ in {"NOUN", "PRON"} and not any(t.pos_ in {"VERB", "AUX", "ADJ", "ADV"} for t in span):
                    continue

                # FILTER 5: Noise (All punctuation or all stopwords)
                if all(t.is_punct for t in span): continue
                if all(t.is_stop for t in span): continue

                # If it manages to go over all the filters, add it to the list. 
                add({
                    "id": f"{span.start}-{span.end}",
                    "text": span.text,
                    "start_token": span.start,
                    "end_token": span.end,
                })

        # STEP 3: Deduplicate
        clean_candidates = self.filter_contained_candidates(raw_candidates)
        
        return clean_candidates
