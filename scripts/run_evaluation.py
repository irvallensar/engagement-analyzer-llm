import spacy
from pathlib import Path

# Imports your custom tools from other files
from suggester.custom_suggester import CandidateSuggester
from scripts.local_llm_client import call_local_llm
from scripts.llm_utils import parse_llm_json

# Defines where your prompt template lives
PROMPT_PATH = Path("prompts/candidate_labeling.txt")

# Helper: Reads the text from the prompt file
def load_prompt():
    return PROMPT_PATH.read_text()

# Helper: Converts the list of candidates into a string for the LLM
# Input: [{"id": "0-1", "text": "word"}, ...]
# Output: "0-1: "word"\n..."
def build_candidates_block(candidates):
    return "\n".join(f'{c["id"]}: "{c["text"]}"' for c in candidates)


# --- POST-PROCESSING RULES (The Python Logic fixes) ---

def suppress_complement_proclaim(llm_items, candidates):
    """
    Fixes a specific logic error where "Proclaim" overlaps with "Monogloss".
    """
    monogloss_spans = []

    # 1. Find all candidates the LLM labeled "MONOGLOSS"
    for item in llm_items:
        if item["label"] == "MONOGLOSS":
            # Find the original candidate data to get start/end tokens
            c = next(c for c in candidates if c["id"] == item["id"])
            monogloss_spans.append((c["start_token"], c["end_token"]))

    cleaned = []
    for item in llm_items:
        # 2. Check every "PROCLAIM" label
        if item["label"] == "PROCLAIM":
            c = next(c for c in candidates if c["id"] == item["id"])
            # If this PROCLAIM starts inside (or right after) a MONOGLOSS span...
            for m_start, m_end in monogloss_spans:
                if c["start_token"] >= m_start and c["end_token"] <= m_end + 2:
                    # Change it to "O" (Outside/Delete it)
                    item = {**item, "label": "O"}
        cleaned.append(item)

    return cleaned

def force_monogloss(candidates, llm_items):
    """
    Fixes the issue where the LLM calls everything "ENTERTAIN".
    """
    # A set of words that imply doubt/opinion.
    uncertainty_markers = {
        "may", "might", "could", "believe", "think", ... # (list truncated for brevity)
    }

    fixed = []
    for item in llm_items:
        c = next(c for c in candidates if c["id"] == item["id"])
        text = c["text"].lower()

        # Logic: If LLM says "ENTERTAIN" BUT there are no uncertainty words...
        if item["label"] == "ENTERTAIN":
            if not any(w in text for w in uncertainty_markers):
                # Force it to be "MONOGLOSS" (Fact) instead.
                item = {**item, "label": "MONOGLOSS"}
        
        fixed.append(item)

    return fixed


# --- MAIN PIPELINE ---

def run_sentence(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    
    # 1. Load prompt and ONLY replace {sentence}
    prompt = load_prompt().replace("{sentence}", text)

    # 2. Call the LLM
    llm_raw = call_local_llm(prompt)
    print("RAW LLM OUTPUT:")
    print(repr(llm_raw))
    print("------")

    # 3. Parse JSON
    llm_items = parse_llm_json(llm_raw)

    pred_spans = []
    
    # 4. Map the LLM's text back to spaCy token indices
    for item in llm_items:
        if item["label"] == "O":
            continue
            
        span_text = item.get("text", "")
        if not span_text: 
            continue

        # Find where the string starts in the original sentence
        start_char = text.find(span_text)
        
        if start_char != -1:
            end_char = start_char + len(span_text)
            
            # Use spaCy to convert character positions back to Token IDs (0, 1, 2)
            span = doc.char_span(start_char, end_char, alignment_mode="expand")
            
            if span:
                pred_spans.append((
                    item["label"],
                    span.start,
                    span.end
                ))
            else:
                print(f"Warning: Could not align tokens for text: '{span_text}'")
        else:
            print(f"Warning: LLM hallucinated text not in sentence: '{span_text}'")

    return pred_spans

# Ensures this runs only with "python run_evaluation.py"
if __name__ == "__main__":
    sentence = "The language you speak might influence your thoughts."
    preds = run_sentence(sentence)
    # Print results
