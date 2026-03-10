# SCRIPT BEFORE TWO-PASS IMPLEMENTATION

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
        label = item.get("label", "O")
        # FIX: Ignore "O" and empty string labels ("")
        if label == "O" or not label.strip():
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
                pred_spans.append((label, span.start, span.end))
            else:
                print(f"  [!] Warning: Could not align tokens for text: '{span_text}'")
        else:
            print(f"  [!] Warning: LLM hallucinated text not in sentence: '{span_text}'")

    return pred_spans


# ------ IOB PARSER ------

def parse_iob_file(filepath):
    """Reads the IOB file and extracts sentences and Gold Spans."""
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
        
        # Skip metadata lines
        if "-DOCSTART-" in line or line == "-X-" or line == "O":
            continue

        parts = line.split()
        if len(parts) >= 2:
            current_tokens.append(parts[0])
            # Grab all tag columns to support overlapping spans
            current_tags_matrix.append(parts[1:]) 

    # Convert IOB tags to strict span tuples: (Label, Start, End)
    dataset = []
    for entry in sentences:
        # Force spaCy to use the exact tokens from the IOB file
        doc = Doc(nlp.vocab, words=entry["tokens"])
        text = doc.text
        
        gold_spans = set()
        
        if entry["tags_matrix"]:
            num_cols = len(entry["tags_matrix"][0])
            # Iterate through each tag column independently
            for col_idx in range(num_cols):
                current_label = None
                start_idx = -1
                
                for i, row in enumerate(entry["tags_matrix"]):
                    # Safely get the tag for this column
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
                            
                # Catch any span that runs to the end of the sentence
                if current_label:
                    gold_spans.add((current_label, start_idx, len(entry["tags_matrix"])))
            
        dataset.append({
            "text": text,
            "doc": doc,
            "gold_spans": list(gold_spans)
        })
        
    return dataset


# --- EVALUATION ENGINE ---
def evaluate(filepath, max_samples=None):
    print(f"Loading dataset from {filepath}...")
    dataset = parse_iob_file(filepath)
    
    if max_samples is not None:
        dataset = dataset[:max_samples]
        print(f"*** QUICK TEST MODE: Limiting to first {max_samples} sentences ***")
        
    print(f"Found {len(dataset)} sentences to evaluate.\n")

    # Overall counters
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Category-specific counters
    cat_tp = defaultdict(int)
    cat_fp = defaultdict(int)
    cat_fn = defaultdict(int)

    for i, data in enumerate(dataset):
        print(f"Evaluating Sentence {i+1}/{len(dataset)}...")
        
        gold_spans = set(data["gold_spans"])
        
        # Get predictions from LLM
        pred_list = run_sentence_option2(data["text"], data["doc"])
        pred_spans = set(pred_list)

        # Calculate Strict Matches
        tp_set = gold_spans.intersection(pred_spans)
        fp_set = pred_spans - gold_spans
        fn_set = gold_spans - pred_spans

        # Update overall counters
        true_positives += len(tp_set)
        false_positives += len(fp_set)
        false_negatives += len(fn_set)
        
        # Update category-specific counters
        for span in tp_set: cat_tp[span[0]] += 1
        for span in fp_set: cat_fp[span[0]] += 1
        for span in fn_set: cat_fn[span[0]] += 1

        if len(fp_set) > 0 or len(fn_set) > 0:
            print(f"  Sentence: {data['text']}")
            print(f"  Gold Spans: {gold_spans}")
            print(f"  Pred Spans: {pred_spans}")
            print(f"  -> Errors: {len(fp_set)} False Positives, {len(fn_set)} False Negatives\n")

    # --- FINAL MATH ---
    print("\n========================================")
    print("CATEGORY BREAKDOWN")
    print("========================================")
    
    # Get all unique labels encountered in the test
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

    print("\n========================================")
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
    evaluate("data/dev.iob", max_samples=100)        base_label = label[2:] if label != 'O' else 'O'
        tag_prefix = label[0] if label != 'O' else 'O'
        
        if tag_prefix == 'B':
            if current_label:
                spans.add((current_label, start_index, i))
            current_label = base_label
            start_index = i
        elif tag_prefix == 'I':
            if current_label and base_label == current_label:
                continue
            else:
                if current_label:
                    spans.add((current_label, start_index, i))
                if base_label != 'O':
                    current_label = base_label
                    start_index = i
                else:
                    current_label = None
        else:
            if current_label:
                spans.add((current_label, start_index, i))
                current_label = None
                
    if current_label:
        spans.add((current_label, start_index, len(sentence)))
    return spans

def parse_llm_json(text):
    """Extracts JSON array from the LLM's raw string output."""
    try:
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return []
    except:
        return []

# ---------------------------------------------------------
# 2. SMART MATCHING (CHARACTER ALIGNMENT)
# ---------------------------------------------------------
def align_llm_text_to_words(llm_text, words):
    """
    Finds exact start/end token indices matching the LLM's text, 
    ignoring spaces, punctuation mismatches, and tokenization rules.
    """
    if not llm_text or not isinstance(llm_text, str) or not words:
        return None
        
    dense_char_to_word_idx = []
    dense_string = ""
    
    # Create a dense map of the original sentence
    for i, w in enumerate(words):
        for char in w:
            dense_string += char.lower()
            dense_char_to_word_idx.append(i)
            
    # Clean the LLM's predicted text
    dense_llm_text = "".join(llm_text.split()).lower()
    if not dense_llm_text:
        return None
    
    # Find the substring in the dense map
    start_dense = dense_string.find(dense_llm_text)
    
    if start_dense != -1:
        end_dense = start_dense + len(dense_llm_text) - 1
        start_word_idx = dense_char_to_word_idx[start_dense]
        end_word_idx = dense_char_to_word_idx[end_dense]
        return start_word_idx, end_word_idx + 1 # +1 because Python slices are exclusive at the end
    
    return None

# ---------------------------------------------------------
# 3. MAIN EVALUATION LOOP
# ---------------------------------------------------------
def evaluate_dataset(file_path, prompt_file="prompts/candidate_labeling.txt", max_samples=None):
    print(f"Loading dataset from {file_path}...")
    sentences = load_iob_dataset(file_path)

    with open(prompt_file, 'r', encoding='utf-8') as f:
        base_prompt_template = f.read()

    if max_samples:
        print(f"*** LIMITING TO FIRST {max_samples} SENTENCES ***")
        sentences = sentences[:max_samples]

    print(f"Found {len(sentences)} sentences to evaluate using SMART MATCHING.")

    category_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    total_tp, total_fp, total_fn = 0, 0, 0

    for i, sentence in enumerate(tqdm(sentences, desc="Evaluating")):
        sentence_text = " ".join([word for word, label in sentence])
        prompt = base_prompt_template.replace("{sentence}", sentence_text)
        
        response = call_local_llm(prompt)
        pred_list = parse_llm_json(response)

        gold_spans = extract_spans_from_iob(sentence)
        pred_spans = set()
        words = [w for w, l in sentence]
        
        for p in pred_list:
            if not isinstance(p, dict):
                continue
            
            # Robust label extraction (handles if the LLM uses a different key name)
            raw_label = p.get('label') or p.get('type') or p.get('category')
            text = p.get('text')
            
            if not raw_label or not text:
                continue
                
            # Force perfectly clean, UPPERCASE matching
            label = str(raw_label).upper().strip()
            
            # Use the Smart Matcher
            match_indices = align_llm_text_to_words(text, words)
            if match_indices:
                pred_spans.add((label, match_indices[0], match_indices[1]))

        # Calculate Stats for the sentence
        tp = len(gold_spans & pred_spans)
        fp = len(pred_spans - gold_spans)
        fn = len(gold_spans - pred_spans)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Per-Category Stats
        all_labels = set([s[0] for s in gold_spans] + [s[0] for s in pred_spans])
        for label in all_labels:
            gold_subset = {s for s in gold_spans if s[0] == label}
            pred_subset = {s for s in pred_spans if s[0] == label}
            
            category_metrics[label]['tp'] += len(gold_subset & pred_subset)
            category_metrics[label]['fp'] += len(pred_subset - gold_subset)
            category_metrics[label]['fn'] += len(gold_subset - pred_subset)

    # Print Report
    print("\n" + "="*40)
    print("CATEGORY BREAKDOWN (SMART MATCHING)")
    print("="*40)
    for cat in sorted(category_metrics.keys()):
        m = category_metrics[cat]
        p = m['tp'] / (m['tp'] + m['fp']) if (m['tp'] + m['fp']) > 0 else 0
        r = m['tp'] / (m['tp'] + m['fn']) if (m['tp'] + m['fn']) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        print(f"--- {cat} ---")
        print(f"  TP: {m['tp']} | FP: {m['fp']} | FN: {m['fn']}")
        print(f"  P: {p:.4f} | R: {r:.4f} | F1: {f1:.4f}")

    print("\n" + "="*40)
    print("FINAL STRICT SPAN EVALUATION RESULTS")
    print("="*40)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"True Positives (Exact Matches) : {total_tp}")
    print(f"False Positives (Hallucinations) : {total_fp}")
    print(f"False Negatives (Missed Markers) : {total_fn}")
    print("-" * 40)
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print("="*40)

if __name__ == "__main__":
    evaluate_dataset("data/dev.iob", max_samples=100)
