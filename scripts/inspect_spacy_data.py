import spacy
from spacy.tokens import DocBin
from collections import Counter
import random
import os

def inspect_spacy_file(file_path, sample_size=20):
    # Stop early if the expected .spacy file is missing, so the rest of the inspection can continue.
    if not os.path.exists(file_path):
        print(f"[WARNING] Could not find {file_path}. Skipping...\n")
        return

    print(f"=== Analyzing: {file_path} ===")
    # A blank English pipeline is enough here because DocBin only needs the vocabulary to reconstruct docs.
    nlp = spacy.blank("en")
    doc_bin = DocBin().from_disk(file_path)
    
    # Unpack the binary into a list of readable spaCy documents
    docs = list(doc_bin.get_docs(nlp.vocab))
    
    # 1. Count Total Sentences
    total_sentences = len(docs)
    print(f"  -> Total Sentences: {total_sentences}")
    
    # 2. Check for Overlapping Spans, Nested Spans, and Count Labels
    overlapping_count = 0
    nested_count = 0
    label_counts = Counter()
    
    for doc in docs:
        # The custom span group "sc" is where this project stores span categorizer annotations.
        spans = doc.spans.get("sc", [])
        has_overlap = False
        has_nested = False
        
        for i in range(len(spans)):
            # Tally the label
            label_counts[spans[i].label_] += 1
            
            # Check for overlaps and nesting against other spans in the sentence
            for j in range(i + 1, len(spans)):
                span_a = spans[i]
                span_b = spans[j]
                
                # Overlap logic: A starts before B ends AND B starts before A ends
                if span_a.start < span_b.end and span_b.start < span_a.end:
                    has_overlap = True
                    # Nested logic: A is entirely inside B OR B is entirely inside A
                    if (span_a.start >= span_b.start and span_a.end <= span_b.end) or \
                       (span_b.start >= span_a.start and span_b.end <= span_a.end):
                        has_nested = True
        # Count each sentence only once, even if it contains multiple overlapping or nested span pairs.
        if has_overlap: overlapping_count += 1
        if has_nested: nested_count += 1
        
    print(f"  -> Sentences with Overlapping Spans: {overlapping_count}")
    print(f"  -> Sentences with Nested Spans: {nested_count}")
    
    # Print the Label Distribution
    print("  -> Label Distribution:")
    for label, count in label_counts.most_common():
        print(f"       - {label}: {count}")
    
    # 3. Export Sample for Manual Quality Check
    # The sample review file is saved next to the inspected .spacy file for easier comparison.
    base_name = os.path.basename(file_path).replace('.spacy', '')
    directory = os.path.dirname(file_path)
    output_sample_file = os.path.join(directory, f"{base_name}_manual_review.txt")
    
    print(f"  -> Extracting {sample_size} random samples for manual review...")
    # Safely sample, in case the file has fewer sentences than the requested sample size
    sample_docs = random.sample(docs, min(sample_size, len(docs)))
    
    with open(output_sample_file, "w", encoding="utf-8") as f:
        f.write(f"--- MANUAL REVIEW SAMPLE FOR: {file_path} ---\n")
        f.write(f"Total Dataset Size: {total_sentences} sentences\n\n")
        
        for i, doc in enumerate(sample_docs):
            # Each sampled sentence is printed with its annotated spans and token boundaries.
            f.write(f"Sentence {i+1}: {doc.text}\n")
            spans = doc.spans.get("sc", [])
            if not spans:
                f.write("  [WARNING] No spans found in this sentence!\n")
            else:
                for span in spans:
                    f.write(f"  [{span.label_}] -> '{span.text}' (Tokens {span.start} to {span.end})\n")
            f.write("-" * 60 + "\n")
    
    print(f"  -> [SUCCESS] Sample saved to {output_sample_file}\n")

if __name__ == "__main__":
    print("\nStarting Comprehensive Pre-Training Data Inspection...\n")
    
    # 1. Inspect the original static split files
    inspect_spacy_file("data/synthetic_balanced.spacy", sample_size=30)
    inspect_spacy_file("data/synthetic_pseudo_labeled.spacy", sample_size=30)
    inspect_spacy_file("data/synthetic_train.spacy", sample_size=30)
    
    # 2. Inspect the 5-Fold Cross Validation files
    for fold in range(1, 6):
        # The fold banner makes it easier to scan console output across all cross-validation splits.
        print(f"\n" + "="*40)
        print(f"         EVALUATING FOLD {fold}")
        print("="*40 + "\n")
        
        inspect_spacy_file(f"data/5_fold_exp/train{fold}.spacy", sample_size=30)
        inspect_spacy_file(f"data/5_fold_exp/da_train{fold}.spacy", sample_size=30)
