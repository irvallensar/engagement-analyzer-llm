import spacy
from pathlib import Path

# Imports your custom tools from other files
from suggester.custom_suggester import CandidateSuggester # generates candidate phrases
from scripts.local_llm_client import call_local_llm # sends prompt to LLM
from scripts.llm_utils import parse_llm_json # cleans LLM output

# Defines where your prompt template lives
PROMPT_PATH = Path("prompts/candidate_labeling.txt")

# Helper: Reads the text from the prompt file
def load_prompt():
    return PROMPT_PATH.read_text()

# Helper: Converts the list of candidates into a string for the LLM
# Input: [{"id": "0-1", "text": "word"}, ...]
# Output: "0-1: "word"\n..."
def build_candidates_block(candidates):
    return "\n".join(f'{c["id"]}: "{c["text"]}"' for c in candidates) #LLM must expect text not python object


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
    prompt = load_prompt().replace("{sentence}", text) #"Template: "Classify: {sentence}"

    # 2. Call the LLM
    llm_raw = call_local_llm(prompt) #sends prompt to local model.
    print("RAW LLM OUTPUT:")
    print(repr(llm_raw))
    print("------")

    # 3. Parse JSON
    llm_items = parse_llm_json(llm_raw) #Converts messy text into clean Python list.

    pred_spans = [] #create empty list of prediction spans
    
    # 4. Map the LLM's text back to spaCy token indices
    for item in llm_items: #loop through each LLM prediction
        if item["label"] == "O":
            continue
            
        span_text = item.get("text", "") #get predicted text
        if not span_text: 
            continue

        # Find where the string (text) starts in the original sentence
        start_char = text.find(span_text) #scans the text from left-to-right, and gets the first match
        
        if start_char != -1:
            end_char = start_char + len(span_text)
            
            # Use spaCy to convert character positions back to Token IDs (0, 1, 2)
            span = doc.char_span(start_char, end_char, alignment_mode="expand") #character positions → token indices
            
            if span:
                pred_spans.append((
                    item["label"],
                    span.start,
                    span.end
                )) #making it the final output, (label, start_token, end_token)
            else:
                print(f"Warning: Could not align tokens for text: '{span_text}'")
        else:
            print(f"Warning: LLM hallucinated text not in sentence: '{span_text}'")

    return pred_spans

# Ensures this runs only with "python run_evaluation.py"
if __name__ == "__main__":
    sentence = "It is often believed that the language you speak determines your thoughts."
    preds = run_sentence(sentence)
    # Print results
