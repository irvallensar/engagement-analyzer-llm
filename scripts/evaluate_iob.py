# --- EVALUATION ENGINE ---
def evaluate(filepath, max_samples=None):
    print(f"Loading dataset from {filepath}...")
    dataset = parse_iob_file(filepath)
    
    # --- NEW: Slicing the dataset for quick tests ---
    if max_samples is not None:
        dataset = dataset[:max_samples]
        print(f"*** QUICK TEST MODE: Limiting to first {max_samples} sentences ***")
        
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
    # Change max_samples to whatever number you want to test (e.g., 5). 
    # When you are ready for the full run, change it to max_samples=None
    evaluate("data/dev.iob", max_samples=5)
