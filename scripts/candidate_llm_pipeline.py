# Candidate → LLM → Tuple pipeline

import json
from suggester.custom_suggester import CandidateSuggester
import spacy

def run_pipeline(text, llm_response):
    """
    Candidate → LLM → Tuple pipeline.

    Input:
      - text: raw sentence
      - llm_response: [{"id": int, "label": str}, ...]

    Output:
      - List of (label, start_token, end_token)
      - Compatible with spaCy SpanCat and custom_eval.py
    """
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
