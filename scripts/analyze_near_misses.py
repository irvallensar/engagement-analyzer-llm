import json

def analyze_near_misses(log_file):
    with open(log_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    near_misses = 0
    total_errors = 0

    print("--- NEAR MISS ANALYSIS ---")

    for entry in data:
        gold_spans = entry.get("gold_spans", [])
        pred_spans = entry.get("pred_spans", [])
        
        # Only look at sentences where the model made predictions
        if not gold_spans or not pred_spans:
            continue

        for gold in gold_spans:
            g_label, g_start, g_end = gold
            
            for pred in pred_spans:
                p_label, p_start, p_end = pred
                
                # Check if it's the exact same category, but boundaries don't strictly match
                if g_label == p_label and not (g_start == p_start and g_end == p_end):
                    total_errors += 1
                    
                    # Check for overlap (Is the prediction overlapping with the gold span?)
                    overlap = max(0, min(g_end, p_end) - max(g_start, p_start))
                    if overlap > 0:
                        near_misses += 1
                        
                        # Extract the actual words to see the difference
                        words = entry["text"].split()
                        gold_text = " ".join(words[g_start:g_end])
                        pred_text = " ".join(words[p_start:p_end])
                        
                        print(f"Sentence {entry['sentence_id']}: {g_label}")
                        print(f"  [GOLD]: '{gold_text}' (Tokens {g_start}-{g_end})")
                        print(f"  [PRED]: '{pred_text}' (Tokens {p_start}-{p_end})")
                        print("-" * 50)

    print(f"\nTotal Overlapping Near Misses: {near_misses} (Out of {total_errors} boundary mismatched errors)")

if __name__ == "__main__":
    analyze_near_misses('logs/comprehensive_eval_log_32b_finetune_run13.json')
