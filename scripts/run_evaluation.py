import spacy
from suggester.custom_suggester import CandidateSuggester
from scripts.openrouter_client import call_openrouter
from scripts.llm_utils import parse_llm_json


from pathlib import Path

PROMPT_PATH = Path("prompts/candidate_labeling.txt")

def load_prompt():
    return PROMPT_PATH.read_text()

def build_candidates_block(candidates):
    lines = []
    for c in candidates:
        lines.append(f"{c['id']}: \"{c['text']}\"")
    return "\n".join(lines)

def run_sentence(text, gold_spans):
    nlp = spacy.load("en_core_web_sm")
    suggester = CandidateSuggester(nlp)
    candidates = suggester.get_candidates(text)

    prompt_template = load_prompt()
    prompt = prompt_template.format(
        sentence=text,
        candidates=build_candidates_block(candidates)
    )

    llm_raw = call_openrouter(prompt)
    llm_items = parse_llm_json(llm_raw)

    pred_spans = set()
    for item in llm_items:
        if item["label"] != "O":
            c = candidates[item["id"]]
            pred_spans.add((
                item["label"],
                c["start_token"],
                c["end_token"]
            ))

    return score_set(pred_spans, gold_spans)
