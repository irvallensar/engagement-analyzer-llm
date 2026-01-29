import sys
import os
from collections import defaultdict
import spacy
from tqdm import tqdm

# Import your existing pipeline
# Ensure this script is run from the root directory so python can find 'scripts'
from scripts.run_evaluation import run_sentence

def read_iob_file(file_path):
    """
    Reads an IOB file and groups lines into sentences.
    Returns a list of sentences, where each sentence is a list of (word, tag) tuples.
    """
    sentences = []
    current_sentence = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue
            
            parts = line.split()
            # Standard IOB format: Word Tag (sometimes Word POS Tag)
            # We assume the last column is the IOB tag, first is the word
            word = parts[0]
            tag = parts[-1] 
            current_sentence.append((word, tag))
            
    if current_sentence:
        sentences.append(current_sentence)
        
    return sentences

def extract_spans_from_iob(sentence_data):
    """
    Converts IOB tags (B-ATTR, I-ATTR) into strict spans (Label, Start_Token, End_Token).
    Logic:
      - B-TAG starts a span.
      - I-TAG continues a span (only if strictly following B or I of same tag).
      - O or B-NEWTAG ends the previous span.
    """
    spans = []
    current_label = None
    start_idx = None
    
    for i, (word, tag) in enumerate(sentence_data):
        # Handle "O" tag
        if tag == "O":
            if current_label:
                # Close previous span
                spans.append((current_label, start_idx, i))
                current_label = None
                start_idx = None
            continue
            
        # Handle B-TAG or I-TAG
        # Typical format: B-ATTRIBUTION or I-ATTRIBUTION
        if "-" in tag:
            prefix, label = tag.split("-", 1)
        else:
            # Fallback if just "ATTRIBUTION" (rare in strict IOB)
            prefix, label = "B", tag
            
        if prefix == "B":
            if current_label:
                # Close previous span before starting new one
                spans.append((current_label, start_idx, i))
            current_label = label
            start_idx = i
            
        elif prefix == "I":
            if current_label != label:
                # Mismatch (I-TAG without B-TAG, or different label). 
                # Treat as new start or ignore. Strict IOB usually treats as new start if B missing.
                if current_label:
                     spans.append((current_label, start_idx, i))
                current_label = label
                start_idx = i
                
    # End of sentence cleanup
    if current_label:
        spans.append((current_label, start_idx, len(sentence_data)))
        
    return set(spans)

def evaluate_dataset(iob_path):
    print(f"Loading data from: {iob_path}")
    sentences = read_iob_file(iob_path)
    print(f"Found {len(sentences)} sentences.")
    
    tp = 0 # True Positives
    fp = 0 # False Positives
    fn = 0 # False Negatives
    
    # We use tqdm for a progress bar because LLMs are slow
    for sent_data in tqdm(sentences, desc="Evaluating"):
        # 1. Reconstruct Text
        # Note: We simply join by space. This assumes the tokenizer matches reasonably well.
        # If strict token alignment fails, numbers will be slightly off, but usually close enough for analysis.
        words = [w for w, t in sent_data]
        text = " ".join(words)
        
        # 2. Get Gold Spans
        gold_spans = extract_spans_from_iob(sent_data)
        
        # 3. Get Model Predictions
        try:
            # predicted_spans format: List of (Label, Start, End)
            predicted_list = run_sentence(text)
            
            # Convert tuple list to set for comparison
            # Note: run_sentence outputs spans based on spaCy tokenization.
            # We assume spaCy tokenization aligns with the IOB file words.
            predicted_spans = set(predicted_list)
            
        except Exception as e:
            print(f"\nError processing sentence: {text[:30]}... {e}")
            predicted_spans = set()

        # 4. Compare (Strict Match)
        # Intersection = Correct spans (Label + Start + End must match exactly)
        correct = gold_spans.intersection(predicted_spans)
        
        tp += len(correct)
        fp += len(predicted_spans - gold_spans) # Predicted but not in Gold
        fn += len(gold_spans - predicted_spans) # In Gold but not Predicted

    # 5. Calculate Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print("\n" + "="*40)
    print(f"STRICT SPAN EVALUATION REPORT")
    print("="*40)
    print(f"Sentences Processed: {len(sentences)}")
    print(f"True Positives:  {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print("-" * 40)
    print(f"PRECISION: {precision:.4f}")
    print(f"RECALL:    {recall:.4f}")
    print(f"F1 SCORE:  {f1:.4f}")
    print("="*40)

if __name__ == "__main__":
    # Change this path to point to your actual test.iob file location
    # Example: "data/test.iob"
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        # Default fallback
        test_file = "data/test.iob"
        
    if not os.path.exists(test_file):
        print(f"Error: File not found at {test_file}")
        print("Usage: uv run python -m scripts.evaluate_iob path/to/test.iob")
    else:
        evaluate_dataset(test_file)
