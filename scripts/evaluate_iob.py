import os
import json
import re
from spacy.tokens import Doc
import spacy    # tokenize sentences (split into words) and align character positions to token positions
from pathlib import Path
from collections import defaultdict    # dictionary that automatically starts at 0 for new keys, 
                                       # used for counting TP/FP/FN per category

# Import your existing tools
from scripts.local_llm_client import call_local_llm    # function that sends prompts to Ollama
from scripts.llm_utils import parse_llm_json    # function that converts the LLM's raw text response into a python list   

nlp = spacy.load("en_core_web_sm")    #load spaCy model

DRIVE_DIR = Path("/content/drive/MyDrive/engagement-analyzer-llm") 
DRIVE_DIR.mkdir(parents=True, exist_ok=True)

# Point the cache and the final log directly into Google Drive
CACHE_FILE = DRIVE_DIR / "predictions_cache_md.json" # the cache of the run (containing the logs such as predicted spans 
                                                  # from the LLM, data saved)
EVAL_LOG_FILE = DRIVE_DIR / "comprehensive_eval_log_md.json" # The master (final) record of the whole run

# The two parts of your new prompt architecture
GUIDELINES_PATH = Path("prompts/master_guidelines.txt")
ENGINE_PATH = Path("prompts/candidate_labeling.txt")

def build_final_prompt(sentence):
    """Reads the theoretical guidelines and the task instructions, and fuses them together."""
    guidelines = GUIDELINES_PATH.read_text(encoding='utf-8')
    engine = ENGINE_PATH.read_text(encoding='utf-8')
    
    # Replace the {sentence} placeholder in the engine text
    engine_with_sentence = engine.replace("{sentence}", sentence)
    
    # Glue them together: Guidelines first, then the Task Instructions
    full_prompt = guidelines + "\n\n" + engine_with_sentence
    return full_prompt

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
    # Saves the comprehensive evaluation record safely
    with open(EVAL_LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=4)

def run_sentence_option2(text, doc):    # takes a sentence as plain text and its spaCy doc object
    # --- CHANGED: Now uses build_final_prompt ---
    prompt = build_final_prompt(text)    # fuses the guidelines and inserts the sentence
    llm_raw = call_local_llm(prompt) # sends it to the llm (ollama) and gets the raw response
    
    try:
        # Use Regex to aggressively search for the JSON array and ignore the chatty text
        match = re.search(r'\[.*\]', llm_raw, re.DOTALL)
        if match:
            clean_json = match.group(0)
            llm_items = json.loads(clean_json)
        else:
            # Fallback to your original parser just in case
            llm_items = parse_llm_json(llm_raw)
            
    except Exception as e:
        print(f"  [!] JSON Parse Error (LLM Hallucinated bad syntax): {e}")    
        return []
      
    pred_spans = []
    
    for item in llm_items:    # Loops through each span the LLM predicted
        label = item.get("label", "O")
        # Ignore "O" and empty string labels (""), as we only want the engagement markers (labels)
        if label == "O" or not label.strip():
            continue
            
        span_text = item.get("text", "")    # gets the actual text of the predicted span
        context_before = item.get("context_before", "").strip()    # gets the words before the span (target), used 
                                                                   # to find the exact location if the same words appear multiple times
        
        if not span_text: # if there is none, then continue
            continue

        start_char = -1
        # Anchoring first (solves duplicate words)
        if context_before:                                           # Instead of just searching for "suggests" (which might appear 3 times), 
                                                                     # it searches for "arguably suggests" to find the exact occurrence
            search_string = f"{context_before} {span_text}"
            combo_start = text.find(search_string)
            if combo_start != -1:
                start_char = combo_start + len(context_before) + 1 # +1 skips the space between context_before and span_text
        
        # 2. Fallback if context anchoring fails or wasn't provided
        if start_char == -1:
            start_char = text.find(span_text)
            
        # 3. Converts from character positions back to spaCy tokens (e.g. token 3-6)
        if start_char != -1:
            end_char = start_char + len(span_text)
            span = doc.char_span(start_char, end_char, alignment_mode="expand")
            
            if span:
                pred_spans.append((label, span.start, span.end))    # Appends the final prediction as a tuple (e.g. "ENTERTAIN", 3, 6)
            else:
                pass # Suppressed warning for cleaner output
        else:
            pass # Suppressed warning for cleaner output

    return pred_spans


# ----- IOB PARSER ------

def parse_iob_file(filepath):
    """Reads the IOB file and extracts sentences and Gold Spans."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    sentences = []                 # Prepares empty containers to build sentences from the IOB file    
    current_tokens = []
    current_tags_matrix = []

    for line in lines:                       # IOB files use blank lines to separate sentences
        line = line.strip()                  # When a blank line is hit, saves the current sentence and resets for the next one.
        if not line:
            if current_tokens:
                sentences.append({"tokens": current_tokens, "tags_matrix": current_tags_matrix})
                current_tokens = []
                current_tags_matrix = []
            continue
        
        # Skip metadata/header lines (not part of annotation word)
        if "-DOCSTART-" in line or line == "-X-" or line == "O":
            continue

        # Each IOB line looks like: word TAG1 TAG2 TAG3
        # parts[0] = the word itself
        # parts[1:] = all the tag columns (supports multiple overlapping annotation layers)
        parts = line.split()
        if len(parts) >= 2:
            current_tokens.append(parts[0])
            current_tags_matrix.append(parts[1:]) 

    dataset = []
    for entry in sentences:
        # forces spaCy to use EXACTLY the tokens from your IOB file instead of re-tokenizing
        # Without this, spaCy might split "don't" differently than the annotation file did, causing misalignment
        doc = Doc(nlp.vocab, words=entry["tokens"])        
        text = doc.text
        
        gold_spans = set()

        # IOB Decoder
        # B-ENTERTAIN = Beginning of an ENTERTAIN span → start tracking a new span
        # I-ENTERTAIN = Inside the same span → continue, don't save yet
        # O or different label = span has ended → save it as ("ENTERTAIN", start_token, end_token)  
        # It rocesses each column independently to support overlapping spans
        # — e.g. a word can be both ENTERTAIN and ATTRIBUTE simultaneously.
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


# --- EVALUATION  ---

# Loads the full dataset
def evaluate(filepath, max_samples=None):
    print(f"Loading dataset from {filepath}...")
    dataset = parse_iob_file(filepath)
    
    if max_samples is not None:
        dataset = dataset[:max_samples]
        
    print(f"Found {len(dataset)} sentences to evaluate.\n")

    cache = load_cache()
    if cache:
        print(f"*** CHECKPOINT FILE: Resuming with {len(cache)} previously saved sentences. ***\n")

    # NEW: Initialize the master log list
    master_eval_log = []

    true_positives, false_positives, false_negatives = 0, 0, 0
    token_tp, token_fp, token_fn = 0, 0, 0
    cat_tp, cat_fp, cat_fn = defaultdict(int), defaultdict(int), defaultdict(int)

    for i, data in enumerate(dataset):
        cache_key = str(i)
        evaluated_live = False 
        
        # 1. Fetch LLM Prediction (from cache or live)
        if cache_key in cache:
            pred_list = [tuple(x) for x in cache[cache_key]] 
            pred_spans = set(pred_list)
        else:
            print(f"Evaluating Sentence {i+1}/{len(dataset)}...")
            pred_list = run_sentence_option2(data["text"], data["doc"])
            pred_spans = set(pred_list)
            cache[cache_key] = pred_list
            save_cache(cache)
            evaluated_live = True  # <--- NEW: We flip the flag to True!
        
        gold_spans = set(data["gold_spans"])

        # 2. Strict Matches
        tp_set = gold_spans.intersection(pred_spans)
        fp_set = pred_spans - gold_spans
        fn_set = gold_spans - pred_spans

        true_positives += len(tp_set)
        false_positives += len(fp_set)
        false_negatives += len(fn_set)
        
        for span in tp_set: cat_tp[span[0]] += 1
        for span in fp_set: cat_fp[span[0]] += 1
        for span in fn_set: cat_fn[span[0]] += 1

        # --- THE CORRECTED PRINT BLOCK ---
        if evaluated_live and (len(fp_set) > 0 or len(fn_set) > 0):
            print(f"  Sentence: {data['text']}")
            print(f"  Gold Spans: {gold_spans}")
            print(f"  Pred Spans: {pred_spans}")
            print(f"  -> Errors: {len(fp_set)} False Positives, {len(fn_set)} False Negatives\n")
        # -------------------------------

        # 3. Token-Level Matches
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

        # 4. NEW: Append everything to the Master Log
        # We convert sets to lists so they can be saved as JSON
        log_entry = {
            "sentence_id": i + 1,
            "text": data["text"],
            "gold_spans": [list(span) for span in gold_spans],
            "pred_spans": [list(span) for span in pred_spans],
            "strict_false_positives": [list(span) for span in fp_set],
            "strict_false_negatives": [list(span) for span in fn_set]
        }
        master_eval_log.append(log_entry)

    # Save the complete log to disk after the loop finishes
    save_eval_log(master_eval_log)
    print(f"\n[SUCCESS] Master evaluation log saved to {EVAL_LOG_FILE}")

  # ----- CALCULATION OF PERFORMANCE METRIC SCORES ------
    
    print("\n")
    print("CATEGORY BREAKDOWN (STRICT)")
    all_labels = set(list(cat_tp.keys()) + list(cat_fp.keys()) + list(cat_fn.keys()))
    for label in sorted(all_labels):
        c_tp = cat_tp[label]
        c_fp = cat_fp[label]
        c_fn = cat_fn[label]
        c_p = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0
        c_r = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0
        c_f1 = 2 * (c_p * c_r) / (c_p + c_r) if (c_p + c_r) > 0 else 0
        print(f"--- {label} ---")
        print(f"  TP: {c_tp} | FP: {c_fp} | FN: {c_fn}")
        print(f"  P: {c_p:.4f} | R: {c_r:.4f} | F1: {c_f1:.4f}")

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n")
    print("1. FINAL STRICT SPAN EVALUATION RESULTS")
    print(f"True Positives (Exact Matches) : {true_positives}")
    print(f"False Positives (Hallucinations) : {false_positives}")
    print(f"False Negatives (Missed Markers) : {false_negatives}")
    print("----------------------------------------")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")

    # Token math printout and its formula
    t_precision = token_tp / (token_tp + token_fp) if (token_tp + token_fp) > 0 else 0
    t_recall = token_tp / (token_tp + token_fn) if (token_tp + token_fn) > 0 else 0
    t_f1 = 2 * (t_precision * t_recall) / (t_precision + t_recall) if (t_precision + t_recall) > 0 else 0

    print("\n")
    print("2. TOKEN-LEVEL / PARTIAL EVALUATION")
    print("   (Grades the model word-by-word)")
    print(f"True Positive Words  : {token_tp}")
    print(f"False Positive Words : {token_fp}")
    print(f"False Negative Words : {token_fn}")
    print("----------------------------------------")
    print(f"Token Precision      : {t_precision:.4f}")
    print(f"Token Recall         : {t_recall:.4f}")
    print(f"Token F1-Score       : {t_f1:.4f}")

if __name__ == "__main__":
    evaluate("data/dev.iob", max_samples=20)
