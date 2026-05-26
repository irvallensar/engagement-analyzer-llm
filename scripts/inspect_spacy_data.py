import spacy
from spacy.tokens import DocBin
import random
import os

def inspect_spacy_file(file_path, sample_size=20):
    if not os.path.exists(file_path):
        print(f"[ERROR] Could not find {file_path}. Skipping...\n")
        return

    print(f"=== Analyzing: {file_path} ===")
    nlp = spacy.blank("en")
    doc_bin = DocBin().from_disk(file_path)
    
    # Unpack the binary into a list of readable spaCy documents
    docs = list(doc_bin.get_docs(nlp.vocab))
    
    # 1. Count Total Sentences
    total_sentences = len(docs)
    print(f"  -> Total Sentences: {total_sentences}")
    
    # 2. Check for Overlapping & Nested Spans
    overlapping_count = 0
    nested_count = 0
    
    for doc in docs:
        spans = doc.spans.get("sc", [])
        has_overlap = False
        has_nested = False
        
        for i in range(len(spans)):
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
        
        if has_overlap: overlapping_count += 1
        if has_nested: nested_count += 1
        
    print(f"  -> Sentences with Overlapping Spans: {overlapping_count}")
    print(f"  -> Sentences with Nested Spans: {nested_count}")
    
    # 3. Export Sample for Manual Quality Check
    base_name = os.path.basename(file_path).replace('.spacy', '')
    output_sample_file = f"data/{base_name}_manual_review.txt"
    
    print(f"  -> Extracting {sample_size} random samples for manual review...")
    # Safely sample, in case the file has fewer sentences than the requested sample size
    sample_docs = random.sample(docs, min(sample_size, len(docs)))
    
    with open(output_sample_file, "w", encoding="utf-8") as f:
        f.write(f"--- MANUAL REVIEW SAMPLE FOR: {file_path} ---\n")
        f.write(f"Total Dataset Size: {total_sentences} sentences\n\n")
        
        for i, doc in enumerate(sample_docs):
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
    print("\nStarting Pre-Training Data Inspection...\n")
    
    # Inspect the purely synthetic dataset
    inspect_spacy_file("data/synthetic_balanced.spacy", sample_size=30)
    
    # Inspect the finalized dataset
    inspect_spacy_file("data/synthetic_train.spacy", sample_size=30)
    
    # Inspect the finalized dataser (for 5-fold CV)
    inspect_spacy_file("data/da_train1.spacy", sample_size=30)
    inspect_spacy_file("data/da_train2.spacy", sample_size=30)
    inspect_spacy_file("data/da_train3.spacy", sample_size=30)
    inspect_spacy_file("data/da_train4.spacy", sample_size=30)
    inspect_spacy_file("data/da_train5.spacy", sample_size=30)
    
