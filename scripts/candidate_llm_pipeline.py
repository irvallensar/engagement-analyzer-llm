# Candidate → LLM → Tuple pipeline

import json
from suggester.custom_suggester import CandidateSuggester
import spacy

def run_pipeline(text, llm_response):
    nlp = spacy.load("en_core_web_sm")
    suggester = CandidateSuggester(nlp)
    candidates = suggester.get_candidates(text)

    results = []
    for item in llm_response:
        if item["label"] != "O":
            cand = candidates[item["id"]]
            results.append((
                item["label"],
                cand["start_token"],
                cand["end_token"]
            ))

    return results
