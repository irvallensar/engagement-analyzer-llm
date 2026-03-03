import spacy
from pathlib import Path

# Import your existing tools
from scripts.local_llm_client import call_local_llm
from scripts.llm_utils import parse_llm_json

nlp = spacy.load("en_core_web_sm")
PROMPT_PATH = Path("prompts/candidate_labeling.txt")

def load_prompt():
    return PROMPT_PATH.read_text()

# --- OPTION 2: TEXT-BASED EXTRACTION WITH CONTEXT ANCHORING ---
def run_sentence_option2(text, doc):
    prompt = load_prompt().replace("{sentence}", text)
    
    # Call the LLM
    llm_raw = call_local_llm(prompt)
    
    try:
        llm_items = parse_llm_json(llm_raw)
    except Exception as e:
        print(f"  [!] JSON Parse Error (LLM Hallucinated bad syntax): {e}")
        return []

    pred_spans = []
    
    for item in llm_items:
        if item.get("label", "O") == "O":
            continue
            
        span_text = item.get("text", "")
        context_before = item.get("context_before", "").strip()
        
        if not span_text: 
            continue

        start_char = -1
        
        # 1. Try Context Anchoring first (solves duplicate words)
        if context_before:
            search_string = f"{context_before} {span_text}"
            combo_start = text.find(search_string)
            if combo_start != -1:
                start_char = combo_start + len(context_before) + 1 # +1 for the space
        
        # 2. Fallback if context anchoring fails or wasn't provided
        if start_char == -1:
            start_char = text.find(span_text)
            
        # 3. Map back to spaCy tokens
        if start_char != -1:
            end_char = start_char + len(span_text)
            span = doc.char_span(start_char, end_char, alignment_mode="expand")
            
            if span:
                pred_spans.append((item["label"], span.start, span.end))
            else:
                print(f"  [!] Warning: Could not align tokens for text: '{span_text}'")
        else:
            print(f"  [!] Warning: LLM hallucinated text not in sentence: '{span_text}'")

    return pred_spans


# --- IOB PARSER ---
def parse_iob_file(filepath):
    """Reads the IOB file and extracts sentences and Gold Spans."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    sentences = []
    current_tokens = []
    current_tags = []

    for line in lines:
        line = line.strip()
        if not line:
            if current_tokens:
                sentences.append({"tokens": current_tokens, "tags": current_tags})
                current_tokens = []
                current_tags = []
            continue
        
        parts = line.split()
        if len(parts) >= 2:
            current_tokens.append(parts[0])
            current_tags.append(parts[-1]) # The last column is usually the IOB tag

    # Convert IOB tags to strict span tuples: (Label, Start, End)
    dataset = []
    for entry in sentences:
        text = " ".join(entry["tokens"])
        doc = nlp(text)
        
        gold_spans = []
        current_label = None
        start_idx = -1
        
        for i, tag in enumerate(entry["tags"]):
            if tag.startswith("B-"):
                if current_label:
                    gold_spans.append((current_label, start_idx, i))
                current_label = tag[2:]
                start_idx = i
            elif tag.startswith("I-") and current_label == tag[2:]:
                continue
            else:
                if current_label:
                    gold_spans.append((current_label, start_idx, i))
                    current_label = None
                    start_idx = -1
        if current_label:
            gold_spans.append((current_label, start_idx, len(entry["tags"])))
            
        dataset.append({
            "text": text,
            "doc": doc,
            "gold_spans": gold_spans
        })
        
    return dataset


# --- EVALUATION ENGINE ---
def evaluate(filepath):
    print(f"Loading dataset from {filepath}...")
    dataset = parse_iob_file(filepath)
        
    print(f"Found {len(dataset)} sentences to evaluate.\n")

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for i, data in enumerate(dataset):
        print(f"Evaluating Sentence {i+1}/{len(dataset)}...")
        
        gold_spans = set(data["gold_spans"])
        
        # Get predictions from LLM
        pred_list = run_sentence_option2(data["text"], data["doc"])
        pred_spans = set(pred_list)

        # Calculate Strict Matches
        tp = len(gold_spans.intersection(pred_spans))
        fp = len(pred_spans - gold_spans)
        fn = len(gold_spans - pred_spans)

        true_positives += tp
        false_positives += fp
        false_negatives += fn
        
        # Print a mini-report for this sentence so you can see where it failed
        if fp > 0 or fn > 0:
            print(f"  Sentence: {data['text']}")
            print(f"  Gold Spans: {gold_spans}")
            print(f"  Pred Spans: {pred_spans}")
            print(f"  -> Errors: {fp} False Positives, {fn} False Negatives\n")

    # --- FINAL MATH ---
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("========================================")
    print("FINAL STRICT SPAN EVALUATION RESULTS")
    print("========================================")
    print(f"True Positives (Exact Matches) : {true_positives}")
    print(f"False Positives (Hallucinations) : {false_positives}")
    print(f"False Negatives (Missed Markers) : {false_negatives}")
    print("----------------------------------------")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print("========================================")

if __name__ == "__main__":
    evaluate("data/dev.iob")
