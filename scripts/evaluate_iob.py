from spacy.tokens import Doc
import spacy
from pathlib import Path
from collections import defaultdict

# Import your existing tools
from scripts.local_llm_client import call_local_llm
from scripts.llm_utils import parse_llm_json

nlp = spacy.load("en_core_web_sm")
PROMPT_PATH = Path("prompts/candidate_labeling.txt")

def load_prompt():
    return PROMPT_PATH.read_text(encoding='utf-8')

def run_sentence_option2(text, doc):
    prompt = load_prompt().replace("{sentence}", text)
    
    llm_raw = call_local_llm(prompt)
    
    try:
        llm_items = parse_llm_json(llm_raw)
    except Exception as e:
        return []

    pred_spans = []
    
    for item in llm_items:
        label = item.get("label", "O")
        if label == "O" or not label.strip():
            continue
            
        span_text = item.get("text", "")
        context_before = item.get("context_before", "").strip()
        
        if not span_text: 
            continue

        start_char = -1
        
        if context_before:
            search_string = f"{context_before} {span_text}"
            combo_start = text.find(search_string)
            if combo_start != -1:
                start_char = combo_start + len(context_before) + 1 
        
        if start_char == -1:
            start_char = text.find(span_text)
            
        if start_char != -1:
            end_char = start_char + len(span_text)
            span = doc.char_span(start_char, end_char, alignment_mode="expand")
            
            if span:
                pred_spans.append((label, span.start, span.end))
                
    return pred_spans


def parse_iob_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    sentences = []
    current_tokens = []
    current_tags_matrix = []

    for line in lines:
        line = line.strip()
        if not line:
            if current_tokens:
                sentences.append({"tokens": current_tokens, "tags_matrix": current_tags_matrix})
                current_tokens = []
                current_tags_matrix = []
            continue
        
        if "-DOCSTART-" in line or line == "-X-" or line == "O":
            continue

        parts = line.split()
        if len(parts) >= 2:
            current_tokens.append(parts[0])
            current_tags_matrix.append(parts[1:]) 

    dataset = []
    for entry in sentences:
        doc = Doc(nlp.vocab, words=entry["tokens"])
        text = doc.text
        gold_spans = set()
        
        if entry["tags_matrix"]:
            num_cols = len(entry["tags_matrix"][0])
            for col_idx in range(num_cols):
                current_label = None
                start_idx = -1
                
                for i, row in enumerate(entry["tags_matrix"]):
                    tag = row[col_idx] if col_idx < len(row) else "O"
                    
                    if tag.startswith("B-"):
                        if current_label:
                            gold_spans.add((current_label, start_idx, i))
                        current_label = tag[2:]
                        start_idx = i
                    elif tag.startswith("I-") and current_label == tag[2:]:
                        continue
                    else:
                        if current_label:
                            gold_spans.add((current_label, start_idx, i))
                            current_label = None
                            start_idx = -1
                            
                if current_label:
                    gold_spans.add((current_label, start_idx, len(entry["tags_matrix"])))
            
        dataset.append({
            "text": text,
            "doc": doc,
            "gold_spans": list(gold_spans)
        })
        
    return dataset


def evaluate(filepath, max_samples=None):
    print(f"Loading dataset from {filepath}...")
    dataset = parse_iob_file(filepath)
    
    if max_samples is not None:
        dataset = dataset[:max_samples]
        print(f"*** QUICK TEST MODE: Limiting to first {max_samples} sentences ***")
        
    print(f"Found {len(dataset)} sentences to evaluate.\n")

    # Strict Counters
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Token-Level Counters (New)
    token_tp = 0
    token_fp = 0
    token_fn = 0

    for i, data in enumerate(dataset):
        gold_spans = set(data["gold_spans"])
        pred_list = run_sentence_option2(data["text"], data["doc"])
        pred_spans = set(pred_list)

        # 1. Calculate Strict Matches
        tp_set = gold_spans.intersection(pred_spans)
        fp_set = pred_spans - gold_spans
        fn_set = gold_spans - pred_spans

        true_positives += len(tp_set)
        false_positives += len(fp_set)
        false_negatives += len(fn_set)
        
        # 2. Calculate Token-Level (Partial) Matches
        gold_tokens = set()
        for label, start, end in gold_spans:
            for idx in range(start, end):
                gold_tokens.add((idx, label))
                
        pred_tokens = set()
        for label, start, end in pred_spans:
            for idx in range(start, end):
                pred_tokens.add((idx, label))
                
        tok_tp = gold_tokens.intersection(pred_tokens)
        tok_fp = pred_tokens - gold_tokens
        tok_fn = gold_tokens - pred_tokens
        
        token_tp += len(tok_tp)
        token_fp += len(tok_fp)
        token_fn += len(tok_fn)

    # --- FINAL MATH ---
    print("\n========================================")
    print("1. STRICT SPAN EVALUATION RESULTS")
    print("   (Exact character boundaries only)")
    print("========================================")
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"True Positives  : {true_positives}")
    print(f"False Positives : {false_positives}")
    print(f"False Negatives : {false_negatives}")
    print(f"Strict Precision: {precision:.4f}")
    print(f"Strict Recall   : {recall:.4f}")
    print(f"Strict F1-Score : {f1:.4f}")

    print("\n========================================")
    print("2. TOKEN-LEVEL / PARTIAL EVALUATION")
    print("   (Grades the model word-by-word)")
    print("========================================")
    t_precision = token_tp / (token_tp + token_fp) if (token_tp + token_fp) > 0 else 0
    t_recall = token_tp / (token_tp + token_fn) if (token_tp + token_fn) > 0 else 0
    t_f1 = 2 * (t_precision * t_recall) / (t_precision + t_recall) if (t_precision + t_recall) > 0 else 0

    print(f"True Positive Words  : {token_tp}")
    print(f"False Positive Words : {token_fp}")
    print(f"False Negative Words : {token_fn}")
    print(f"Token Precision      : {t_precision:.4f}")
    print(f"Token Recall         : {t_recall:.4f}")
    print(f"Token F1-Score       : {t_f1:.4f}")
    print("========================================")

if __name__ == "__main__":
    evaluate("data/dev.iob", max_samples=100)
