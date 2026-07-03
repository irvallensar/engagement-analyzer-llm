import os
import json
import spacy
from spacy.tokens import Doc
from pathlib import Path
from collections import defaultdict

# Import your new Outlines LLM client
from scripts.local_llm_client_4 import call_local_llm

nlp = spacy.load("en_core_web_sm")
PROMPT_PATH = Path("prompts/candidate_labeling.txt")
DRIVE_DIR = Path("logs") 
DRIVE_DIR.mkdir(parents=True, exist_ok=True)

# FIX: Rename the cache file so we don't load the broken "instant" empty arrays from the previous run
CACHE_FILE = DRIVE_DIR / "predictions_cache_mistral_3_24b_mlx.json"
EVAL_LOG_FILE = DRIVE_DIR / "comprehensive_eval_log_mistral_3_24b_mlx.json"

def load_prompt():
    return PROMPT_PATH.read_text(encoding='utf-8')

def load_cache():
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_cache(cache_data):
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, indent=2)

def save_eval_log(log_data):
    with open(EVAL_LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=4)

def run_sentence_option2(text, doc):
    prompt = load_prompt().replace("{sentence}", text)
    
    try:
        # Call the LLM. It now returns a clean Pydantic object!
        llm_result = call_local_llm(prompt)
        llm_items = llm_result.spans
    except Exception as e:
        # If the LLM still gets cut off or hallucinates unparseable text, 
        # we log it and return empty rather than crashing the whole 5000-sentence run.
        print(f"\n  [WARNING] LLM failed on this sentence (Token limit or Validation): {e}")
        return []
      
    pred_spans = []
    
    for item in llm_items:
        # Access attributes directly from the Pydantic object
        label = item.label
        if label == "O" or not label.strip():
            continue
            
        span_text = item.text
        context_before = item.context_before.strip()
        
        if not span_text:
            continue

        start_char = -1
        
        # 1. Anchoring first
        if context_before:
            search_string = f"{context_before} {span_text}"
            combo_start = text.find(search_string)
            if combo_start != -1:
                start_char = combo_start + len(context_before) + 1
        
        # 2. Fallback
        if start_char == -1:
            start_char = text.find(span_text)
            
        # 3. Converts to spaCy tokens
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
        
    print(f"Found {len(dataset)} sentences to evaluate.\n")

    cache = load_cache()
    if cache:
        print(f"*** CHECKPOINT FILE: Resuming with {len(cache)} previously saved sentences. ***\n")

    master_eval_log = []

    true_positives, false_positives, false_negatives = 0, 0, 0
    token_tp, token_fp, token_fn = 0, 0, 0
    cat_tp, cat_fp, cat_fn = defaultdict(int), defaultdict(int), defaultdict(int)

    token_cat_tp = defaultdict(int)
    token_cat_fp = defaultdict(int)
    token_cat_fn = defaultdict(int)

    for i, data in enumerate(dataset):
        cache_key = str(i)
        evaluated_live = False 
        
        if cache_key in cache:
            pred_list = [tuple(x) for x in cache[cache_key]] 
            pred_spans = set(pred_list)
        else:
            print(f"Evaluating Sentence {i+1}/{len(dataset)}...")
            pred_list = run_sentence_option2(data["text"], data["doc"])
            pred_spans = set(pred_list)
            cache[cache_key] = pred_list
            save_cache(cache)
            evaluated_live = True
        
        gold_spans = set(data["gold_spans"])

        tp_set = gold_spans.intersection(pred_spans)
        fp_set = pred_spans - gold_spans
        fn_set = gold_spans - pred_spans

        true_positives += len(tp_set)
        false_positives += len(fp_set)
        false_negatives += len(fn_set)
        
        for span in tp_set: cat_tp[span[0]] += 1
        for span in fp_set: cat_fp[span[0]] += 1
        for span in fn_set: cat_fn[span[0]] += 1

        if evaluated_live and (len(fp_set) > 0 or len(fn_set) > 0):
            print(f"  Sentence: {data['text']}")
            print(f"  Gold Spans: {gold_spans}")
            print(f"  Pred Spans: {pred_spans}")
            print(f"  -> Errors: {len(fp_set)} False Positives, {len(fn_set)} False Negatives\n")

        gold_tokens = set()
        gold_indices_with_label = set()
        for label, start, end in gold_spans:
            for idx in range(start, end):
                gold_tokens.add((idx, label))
                gold_indices_with_label.add(idx)
                
        pred_tokens = set()
        pred_indices_with_label = set()
        for label, start, end in pred_spans:
            for idx in range(start, end):
                pred_tokens.add((idx, label))
                pred_indices_with_label.add(idx)
                
        doc_length = len(data["doc"])
        
        for idx in range(doc_length):
            if idx not in gold_indices_with_label:
                gold_tokens.add((idx, "O"))
            if idx not in pred_indices_with_label:
                pred_tokens.add((idx, "O"))
                
        tok_tp = gold_tokens.intersection(pred_tokens)
        tok_fp = pred_tokens - gold_tokens
        tok_fn = gold_tokens - pred_tokens
        
        token_tp += len(tok_tp)
        token_fp += len(tok_fp)
        token_fn += len(tok_fn)

        for idx, label in tok_tp:
            token_cat_tp[label] += 1

        for idx, label in tok_fp:
            token_cat_fp[label] += 1

        for idx, label in tok_fn:
            token_cat_fn[label] += 1

        log_entry = {
            "sentence_id": i + 1,
            "text": data["text"],
            "gold_spans": [list(span) for span in gold_spans],
            "pred_spans": [list(span) for span in pred_spans],
            "strict_false_positives": [list(span) for span in fp_set],
            "strict_false_negatives": [list(span) for span in fn_set]
        }
        master_eval_log.append(log_entry)

    save_eval_log(master_eval_log)
    print(f"\n[SUCCESS] Master evaluation log saved to {EVAL_LOG_FILE}")

    print("\n")
    print("CATEGORY BREAKDOWN (STRICT)")
    all_labels = set(list(cat_tp.keys()) + list(cat_fp.keys()) + list(cat_fn.keys()))
    
    macro_strict_f1_sum = 0
    weighted_strict_f1_sum = 0
    total_strict_support = 0

    for label in sorted(all_labels):
        c_tp = cat_tp[label]
        c_fp = cat_fp[label]
        c_fn = cat_fn[label]
        
        support = c_tp + c_fn
        total_strict_support += support

        c_p = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0
        c_r = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0
        c_f1 = 2 * (c_p * c_r) / (c_p + c_r) if (c_p + c_r) > 0 else 0
        
        macro_strict_f1_sum += c_f1
        weighted_strict_f1_sum += (c_f1 * support)

        print(f"--- {label} ---")
        print(f"  TP: {c_tp} | FP: {c_fp} | FN: {c_fn}")
        print(f"  P: {c_p:.4f} | R: {c_r:.4f} | F1: {c_f1:.4f}")

    macro_strict_f1 = macro_strict_f1_sum / len(all_labels) if len(all_labels) > 0 else 0
    weighted_strict_f1 = weighted_strict_f1_sum / total_strict_support if total_strict_support > 0 else 0

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n")
    print("1. FINAL STRICT SPAN EVALUATION RESULTS")
    print(f"True Positives (Exact Matches) : {true_positives}")
    print(f"False Positives (Hallucinations) : {false_positives}")
    print(f"False Negatives (Missed Markers) : {false_negatives}")
    print("----------------------------------------")
    print(f"Precision (Micro) : {precision:.4f}")
    print(f"Recall (Micro)    : {recall:.4f}")
    print(f"F1-Score (Micro)  : {f1:.4f}")
    print(f"F1-Score (Macro)  : {macro_strict_f1:.4f}")
    print(f"F1-Score (Weighted): {weighted_strict_f1:.4f}")
  
    t_precision = token_tp / (token_tp + token_fp) if (token_tp + token_fp) > 0 else 0
    t_recall = token_tp / (token_tp + token_fn) if (token_tp + token_fn) > 0 else 0
    t_micro_f1 = 2 * (t_precision * t_recall) / (t_precision + t_recall) if (t_precision + t_recall) > 0 else 0

    token_labels = set(list(token_cat_tp.keys()) + list(token_cat_fp.keys()) + list(token_cat_fn.keys()))
    
    macro_f1_sum = 0
    weighted_f1_sum = 0
    total_true_tokens = 0

    for label in token_labels:
        l_tp = token_cat_tp[label]
        l_fp = token_cat_fp[label]
        l_fn = token_cat_fn[label]
        
        support = l_tp + l_fn
        total_true_tokens += support
        
        l_p = l_tp / (l_tp + l_fp) if (l_tp + l_fp) > 0 else 0
        l_r = l_tp / (l_tp + l_fn) if (l_tp + l_fn) > 0 else 0
        l_f1 = 2 * (l_p * l_r) / (l_p + l_r) if (l_p + l_r) > 0 else 0
        
        macro_f1_sum += l_f1
        weighted_f1_sum += l_f1 * support

    t_macro_f1 = macro_f1_sum / len(token_labels) if len(token_labels) > 0 else 0
    t_weighted_f1 = weighted_f1_sum / total_true_tokens if total_true_tokens > 0 else 0

    print("\n")
    print("2. TOKEN-LEVEL / PARTIAL EVALUATION")
    print("   (Grades the model word-by-word)")
    print(f"True Positive Words  : {token_tp}")
    print(f"False Positive Words : {token_fp}")
    print(f"False Negative Words : {token_fn}")
    print("----------------------------------------")
    print(f"Token Precision (Micro)  : {t_precision:.4f}")
    print(f"Token Recall (Micro)     : {t_recall:.4f}")
    print(f"Token F1-Score (Micro)   : {t_micro_f1:.4f}")
    print(f"Token F1-Score (Macro)   : {t_macro_f1:.4f}")
    print(f"Token F1-Score (Weighted): {t_weighted_f1:.4f}")

if __name__ == "__main__":
    evaluate("data/test.iob", max_samples=None)
