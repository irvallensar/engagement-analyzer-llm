import spacy
from pathlib import Path

from suggester.custom_suggester import CandidateSuggester
from scripts.local_llm_client import call_local_llm
from scripts.llm_utils import parse_llm_json

PROMPT_PATH = Path("prompts/candidate_labeling.txt")


def load_prompt():
    return PROMPT_PATH.read_text()


def build_candidates_block(candidates):
    return "\n".join(f'{c["id"]}: "{c["text"]}"' for c in candidates)


# ---------- POST-PROCESSING RULES ----------

def suppress_complement_proclaim(llm_items, candidates):
    """Suppress PROCLAIM inside / after MONOGLOSS spans"""
    monogloss_spans = []

    for item in llm_items:
        if item["label"] == "MONOGLOSS":
            c = next(c for c in candidates if c["id"] == item["id"])
            monogloss_spans.append((c["start_token"], c["end_token"]))

    cleaned = []
    for item in llm_items:
        if item["label"] == "PROCLAIM":
            c = next(c for c in candidates if c["id"] == item["id"])
            for m_start, m_end in monogloss_spans:
                if c["start_token"] >= m_start and c["end_token"] <= m_end + 2:
                    item = {**item, "label": "O"}
        cleaned.append(item)

    return cleaned


# In scripts/run_evaluation.py

def force_monogloss(candidates, llm_items):
    """Demote spurious ENTERTAIN only if no stance markers are present"""
    # Expanded list to include mental verbs, adverbs, and attribution signals
    uncertainty_markers = {
        "may", "might", "could", "probably", "possibly", "likely", "unlikely",
        "seem", "appear", "suggest", "believe", "think", "guess", "assume",
        "claim", "argue", "contend", "often", "usually", "generally", "opinion"
    }

    fixed = []
    for item in llm_items:
        c = next(c for c in candidates if c["id"] == item["id"])
        text = c["text"].lower()

        if item["label"] == "ENTERTAIN":
            # Check if any marker exists in the text
            if not any(w in text for w in uncertainty_markers):
                item = {**item, "label": "MONOGLOSS"}
        
        fixed.append(item)

    return fixed


# ---------- MAIN PIPELINE ----------

def run_sentence(text):
    nlp = spacy.load("en_core_web_sm")
    suggester = CandidateSuggester(nlp)
    candidates = suggester.get_candidates(text)
    
    prompt = (
        load_prompt()
        .replace("{sentence}", text)
        .replace("{candidates}", build_candidates_block(candidates))
    )

    llm_raw = call_local_llm(prompt)
    print("RAW LLM OUTPUT:")
    print(repr(llm_raw))
    print("------")

    llm_items = parse_llm_json(llm_raw)
    llm_items = suppress_complement_proclaim(llm_items, candidates)
    llm_items = force_monogloss(candidates, llm_items)

    pred_spans = []
    for item in llm_items:
        if item["label"] == "O":
            continue

        c = next(c for c in candidates if c["id"] == item["id"])
        pred_spans.append((
            item["label"],
            c["start_token"],
            c["end_token"]
        ))

    return pred_spans


if __name__ == "__main__":
    sentence = "It is often believed that the language you speak determines your thoughts."

    preds = run_sentence(sentence)

    print("Sentence:")
    print(sentence)
    print("\nPredicted spans:")
    for p in preds:
        print(p)
