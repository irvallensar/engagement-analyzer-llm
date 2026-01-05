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


def run_sentence(text):
    nlp = spacy.load("en_core_web_sm")
    suggester = CandidateSuggester(nlp)
    candidates = suggester.get_candidates(text)

    prompt_template = load_prompt()
    prompt = prompt_template.format(
        sentence=text,
        candidates=build_candidates_block(candidates),
    )

    llm_raw = call_local_llm(prompt)
    print("RAW LLM OUTPUT:")
    print(llm_raw)
    print("------")
    llm_items = parse_llm_json(llm_raw)

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

