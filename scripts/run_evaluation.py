import spacy
from suggester.custom_suggester import CandidateSuggester
from scripts.local_llm_client import call_local_llm
from scripts.llm_utils import parse_llm_json
from pathlib import Path


from pathlib import Path

def is_valid_entertain(span_text: str) -> bool:
    uncertainty_markers = {
        "might", "may", "could", "probably", "possibly", "perhaps"
    }
    text = span_text.lower()
    return any(w in text for w in uncertainty_markers)


def is_valid_deny(span_text: str) -> bool:
    negation_markers = {
        "not", "no", "never", "n't", "none", "without"
    }
    text = span_text.lower()
    return any(n in text for n in negation_markers)

PROMPT_PATH = Path("prompts/candidate_labeling.txt")

def load_prompt():
    return PROMPT_PATH.read_text()

def build_candidates_block(candidates):
    lines = []
    for i, c in enumerate(candidates):
        lines.append(f"{i}: \"{c['text']}\"")
    return "\n".join(lines)

nlp = spacy.load("en_core_web_sm")

def suppress_spurious_entertain(span_text):
    uncertainty_markers = {"might", "may", "could", "probably", "possibly"}
    return any(w in span_text.lower() for w in uncertainty_markers)
    
    pred_spans = []
    for item in llm_items:
        if item["label"] != "O":
            c = candidates[item["id"]]
            pred_spans.append((
                item["label"],
                c["start_token"],
                c["end_token"]
            ))
        if item["label"] == "ENTERTAIN":
            if not suppress_spurious_entertain(c["text"]):
                continue

def run_sentence(text, gold_spans=None):
    nlp = spacy.load("en_core_web_sm")
    suggester = CandidateSuggester(nlp)
    candidates = suggester.get_candidates(text)

    prompt_template = load_prompt()
    prompt = prompt_template.format(
        sentence=text,
        candidates=build_candidates_block(candidates)
    )

    llm_raw = call_local_llm(prompt)
    llm_items = parse_llm_json(llm_raw)

    pred_spans = []

    for item in llm_items:
        label = item["label"]
        if label == "O":
            continue

        c = candidates[item["id"]]
        span_text = c["text"]

        # 🔒 theory constraints
        if label == "ENTERTAIN" and not is_valid_entertain(span_text):
            continue

        if label == "DENY" and not is_valid_deny(span_text):
            continue

        pred_spans.append((
            label,
            c["start_token"],
            c["end_token"]
        ))

    return pred_spans

if __name__ == "__main__":
    sentence = "The language you speak determines your thoughts."
    preds = run_sentence(sentence, gold_spans=None)

    print("Sentence:")
    print(sentence)
    print("\nPredicted spans:")
    for p in preds:
        print(p)
