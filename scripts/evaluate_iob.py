import os
import json
import re
from tqdm import tqdm
from scripts.local_llm_client import call_local_llm
from collections import defaultdict

# ---------------------------------------------------------
# 1. DATA LOADERS
# ---------------------------------------------------------
def load_iob_dataset(file_path):
    sentences = []
    current_sentence = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    current_sentence.append((parts[0], parts[-1]))
                else:
                    current_sentence.append((parts[0], 'O'))
    if current_sentence:
        sentences.append(current_sentence)
    return sentences

def extract_spans_from_iob(sentence):
    spans = set()
    current_label = None
    start_index = -1
    
    for i, (word, label) in enumerate(sentence):
        base_label = label[2:] if label != 'O' else 'O'
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
    """Robustly extracts JSON even if the LLM wraps it in markdown blocks."""
    try:
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return []
    except:
        return []

# ---------------------------------------------------------
# 2. MAIN EVALUATION LOOP (STRICT MATCHING)
# ---------------------------------------------------------
def evaluate_dataset(file_path, prompt_file="prompts/candidate_labeling.txt", max_samples=None):
    print(f"Loading dataset from {file_path}...")
    sentences = load_iob_dataset(file_path)

    with open(prompt_file, 'r', encoding='utf-8') as f:
        base_prompt_template = f.read()

    if max_samples:
        print(f"*** LIMITING TO FIRST {max_samples} SENTENCES ***")
        sentences = sentences[:max_samples]

    print(f"Found {len(sentences)} sentences to evaluate using STRICT MATCHING.")

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
            
            raw_label = p.get('label', '')
            text = p.get('text', '')
            
            if not raw_label or not text:
                continue
                
            label = str(raw_label).upper().strip()
            
            text_words = text.split()
            if not text_words:
                continue
                
            # EXACT WORD-LIST MATCHING (Original Logic)
            for start_idx in range(len(words) - len(text_words) + 1):
                if words[start_idx : start_idx + len(text_words)] == text_words:
                    pred_spans.add((label, start_idx, start_idx + len(text_words)))

        tp = len(gold_spans & pred_spans)
        fp = len(pred_spans - gold_spans)
        fn = len(gold_spans - pred_spans)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        all_labels = set([s[0] for s in gold_spans] + [s[0] for s in pred_spans])
        for label in all_labels:
            gold_subset = {s for s in gold_spans if s[0] == label}
            pred_subset = {s for s in pred_spans if s[0] == label}
            
            category_metrics[label]['tp'] += len(gold_subset & pred_subset)
            category_metrics[label]['fp'] += len(pred_subset - gold_subset)
            category_metrics[label]['fn'] += len(gold_subset - pred_subset)

    print("\n" + "="*40)
    print("CATEGORY BREAKDOWN (STRICT MATCHING)")
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
