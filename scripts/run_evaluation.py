import spacy
from pathlib import Path

from suggester.custom_suggester import CandidateSuggester
from scripts.local_llm_client import call_local_llm
from scripts.llm_utils import parse_llm_json

PROMPT_PATH = Path("prompts/candidate_labeling.txt")


def load_prompt():
    return PROMPT_PATH.read_text()


def build_candidates_block(candidates):
    lines = []
    for c in candidates:
        lines.append(f'{c["id"]}: "{c["text"]}"')
    return "\n".join(lines)

def force_monogloss(candidates, llm_items):
    fixed = []
    for item in llm_items:
        c = next(c for c in candidates if c["id"] == item["id"])
        text = c["text"].lower()

        if item["label"] == "ENTERTAIN":
            if not any(w in text for w in ["may", "might", "could", "probably"]):
                item["label"] = "MONOGLOSS"

        fixed.append(item)    
    return fixed
    
def run_sentence(text):
    nlp = spacy.load("en_core_web_sm")
    suggester = CandidateSuggester(nlp)
    candidates = suggester.get_candidates(text)

    prompt_template = load_prompt()
    prompt = (
    prompt_template
        .replace("{sentence}", text)
        .replace("{candidates}", build_candidates_block(candidates))
    )

    
    llm_raw = call_local_llm(prompt)
    print("RAW LLM OUTPUT:")
    print(repr(llm_raw))
    print("------")

    if not llm_raw.strip():
        raise RuntimeError("LLM returned empty output")
    llm_items = parse_llm_json(llm_raw)
    llm_items = suppress_complement_proclaim(llm_items, candidates)
    llm_items = force_monogloss(candidates, llm_items)

    def suppress_complement_proclaim(llm_items, candidates):
    """
    If a MONOGLOSS span exists, suppress PROCLAIM labels
    that fall inside or immediately follow it.
    """
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
    sentence = "The language you speak determines your thoughts."

    preds = run_sentence(sentence)

    print("Sentence:")
    print(sentence)
    print("\nPredicted spans:")
    for p in preds:
        print(p)

