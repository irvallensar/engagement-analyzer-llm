import os
import json
import re
from tqdm import tqdm
from scripts.local_llm_client import call_local_llm
from collections import defaultdict

# ==========================================
# 1. DATA LOADING HELPER FUNCTIONS
# ==========================================
def load_iob_dataset(file_path):
    """
    Reads an IOB formatted file and returns a list of sentences.
    Each sentence is a list of (word, label) tuples.
    """
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
    """
    Converts a list of (word, label) tuples into a set of spans.
    Span format: (Label, StartIndex, EndIndex)
    """
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
                    
        else: # tag is 'O'
            if current_label:
                spans.add((current_label, start_index, i))
                current_label = None
                
    if current_label:
        spans.add((current_label, start_index, len(sentence)))
        
    return spans

# ==========================================
# 2. PROMPTS: DIVIDE & CONQUER (FIXED FORMATTING)
# ==========================================
# Note: Double curly braces {{ }} escape them so Python doesn't crash.

PROMPT_PASS_1 = """You are an expert linguistic annotator. 
TASK: Extract specific structural engagement markers from the sentence.
OUTPUT: JSON array of objects. If none, return [].

DEFINITIONS:
1. ATTRIBUTION: Delegates responsibility. Extract the reporting verb phrase (e.g., 'strongly asserts', 'reveals', 'stipulate') or attributing phrase ('According to', 'Their justification').
2. CITATION: Explicit in-text citations. Extract the ENTIRE parenthetical block.
3. SOURCES: The specific entity making a claim. Extract the exact noun or pronoun (e.g., 'Descartes', 'researchers', 'Some studies', 'Their', 'literature'). Tag this independently even if it is right next to an ATTRIBUTION.
4. ENDOPHORIC: Structural references. Extract explicit document cross-references, including the preposition if present (e.g., 'in Figure 1', 'Table 8', '( references )').

FORMAT: [{{"text": "...", "label": "LABEL", "context_before": "..."}}]

Sentence:
{sentence}
"""

PROMPT_PASS_2 = """You are an expert linguistic annotator.
TASK: Extract specific rhetorical engagement markers from the sentence.
OUTPUT: JSON array of objects. If none, return [].

DEFINITIONS:
1. MONOGLOSS: Bare assertions. Extract the main verb/verb phrase of a definitive, undeniable factual statement.
2. DENY: Rejects an alternative directly. Extract ONLY literal syntactic negators ('no', 'not', 'never', 'none', 'fail').
3. COUNTER: Replaces an expected alternative. Extract single transition words (e.g., 'however', 'although', 'only', 'but') or full contrastive clauses.
4. PROCLAIM: Shows rhetorical commitment. Extract words showing authorial backing (e.g., 'undoubtedly', 'in fact', 'conclude').
5. ENTERTAIN: Presents possibility or conditionality. Extract modal verbs ('would', 'could', 'might', 'must'), conditional words ('if'), and hedges ('appear', 'seem', 'largely', 'often').
6. JUSTIFYING: Signals persuasion/substantiation. Extract transition words (e.g., 'Thus', 'Therefore') and causal clauses.

FORMAT: [{{"text": "...", "label": "LABEL", "context_before": "..."}}]

Sentence:
{sentence}
"""

# ==========================================
# 3. EVALUATION LOGIC
# ==========================================
def parse_llm_json(text):
    try:
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return []
    except:
        return []

def evaluate_dataset(file_path, max_samples=None):
    print(f"Loading dataset from {file_path}...")
    sentences = load_iob_dataset(file_path)

    if max_samples:
        print(f"*** LIMITING TO FIRST {max_samples} SENTENCES ***")
        sentences = sentences[:max_samples]

    print(f"Found {len(sentences)} sentences to evaluate using TWO-PASS strategy.")

    # Metrics containers
    category_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    total_tp, total_fp, total_fn = 0, 0, 0

    # Processing Loop
    for i, sentence in enumerate(tqdm(sentences, desc="Evaluating")):
        sentence_text = " ".join([word for word, label in sentence])
        
        # --- PASS 1: STRUCTURE ---
        prompt_1 = PROMPT_PASS_1.format(sentence=sentence_text)
        response_1 = call_local_llm(prompt_1)
        pred_list_1 = parse_llm_json(response_1)
        
        # --- PASS 2: STANCE ---
        prompt_2 = PROMPT_PASS_2.format(sentence=sentence_text)
        response_2 = call_local_llm(prompt_2)
        pred_list_2 = parse_llm_json(response_2)

        # --- MERGE RESULTS ---
        merged_preds = pred_list_1 + pred_list_2
        
        # Extract Gold Spans
        gold_spans = extract_spans_from_iob(sentence)
        
        # Convert LLM preds to Spans
        pred_spans = set()
        words = [w for w, l in sentence]
        
        for p in merged_preds:
            label = p.get('label')
            text = p.get('text')
            if not label or not text: continue
            
            # Simple substring matching
            text_words = text.split()
            if not text_words: continue
            
            # Find all occurrences
            for start_idx in range(len(words) - len(text_words) + 1):
                if words[start_idx : start_idx + len(text_words)] == text_words:
                    pred_spans.add((label, start_idx, start_idx + len(text_words)))

        # Calculate Metrics
        tp = len(gold_spans & pred_spans)
        fp = len(pred_spans - gold_spans)
        fn = len(gold_spans - pred_spans)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Update per-category metrics
        all_labels = set([s[0] for s in gold_spans] + [s[0] for s in pred_spans])
        for label in all_labels:
            gold_subset = {s for s in gold_spans if s[0] == label}
            pred_subset = {s for s in pred_spans if s[0] == label}
            
            cat_tp = len(gold_subset & pred_subset)
            cat_fp = len(pred_subset - gold_subset)
            cat_fn = len(gold_subset - pred_subset)
            
            category_metrics[label]['tp'] += cat_tp
            category_metrics[label]['fp'] += cat_fp
            category_metrics[label]['fn'] += cat_fn

    # --- FINAL REPORT ---
    print("\n" + "="*40)
    print("CATEGORY BREAKDOWN (TWO-PASS)")
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
