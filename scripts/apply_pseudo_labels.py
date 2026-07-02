import spacy
from spacy.tokens import DocBin, Span
import os

# Minimum confidence score required for a pseudo-label to be accepted.
CONFIDENCE_THRESHOLD = 0.7

def merge_pseudo_labels(baseline_model_path, synthetic_input_path, output_path):

    print(f"Loading teacher model from: {baseline_model_path}")
    # Load the custom RoBERTa+LSTM model
    nlp = spacy.load(baseline_model_path)
    
    print(f"Reading pristine synthetic documents using model vocabulary...")
    doc_bin_in = DocBin().from_disk(synthetic_input_path)

    raw_docs = list(doc_bin_in.get_docs(nlp.vocab))
    processed_doc_bin = DocBin()

    added_labels_count = 0
    original_labels_count = 0
    rejected_labels_count = 0 
    
    print("Beginning knowledge distillation scan...")

    for doc in raw_docs:
        # STEP 1: Preserve the original synthetic (gold) annotations from the LLM
        gold_spans = list(doc.spans.get("sc", []))
        original_labels_count += len(gold_spans)
        
        # Build occupancy map to prevent overlap
        gold_boundaries = set()
        for span in gold_spans:
            for token_idx in range(span.start, span.end):
                gold_boundaries.add(token_idx)
        
        # STEP 2: Run the baseline model to predict secondary spans
        predicted_doc = nlp(doc.text)
        predicted_spans = predicted_doc.spans.get("sc", [])
        
        # Safely attempt to extract confidence scores (Handling 2022 architecture quirks)
        try:
            span_scores = predicted_doc.spans["sc"].attrs.get("scores", [])
        except AttributeError:
            span_scores = []
            
        has_scores = len(predicted_spans) == len(span_scores) and len(predicted_spans) > 0
        
        # STEP 3: Merge only non-conflicting predictions
        merged_spans = gold_spans.copy()

        for i, p_span in enumerate(predicted_spans):
            # If the custom model outputs scores, use them. Otherwise, default to 1.0.
            score = span_scores[i] if has_scores else 1.0

            if score >= CONFIDENCE_THRESHOLD:
                span_tokens = set(range(p_span.start, p_span.end))

                # Only add predictions that do not overlap with the LLM's primary gold annotations
                if not span_tokens.intersection(gold_boundaries):
                    new_span = Span(
                        doc,
                        p_span.start,
                        p_span.end,
                        label=p_span.label_
                    )

                    merged_spans.append(new_span)
                    added_labels_count += 1

                    for idx in span_tokens:
                        gold_boundaries.add(idx)
            else:
                rejected_labels_count += 1 
                    
        # Store the final merged span collection
        doc.spans["sc"] = merged_spans
        processed_doc_bin.add(doc)
        
    print(f"\nDistillation Complete!")
    print(f"-> Preserved Gold Labels from LLM: {original_labels_count}")
    print(f"-> Recovered Teacher Pseudo-Labels: {added_labels_count}")
    print(f"-> Rejected Low-Confidence Pseudo-Labels: {rejected_labels_count}")
    
    processed_doc_bin.to_disk(output_path)
    print(f"[SUCCESS] Dense synthetic training data saved to: {output_path}")

if __name__ == "__main__":
    # Pointed to the custom package installed in eguchi_env
    BEST_BASELINE = "en_engagement_LSTM"

    # Input synthetic dataset containing only gold synthetic labels
    SYNTHETIC_DATA = "data/synthetic_few_shot_v3.spacy"

    # Output dataset containing gold labels + pseudo-labels
    OUTPUT_DATA = "data/synthetic_pseudo_labeled_few_shot_v3.spacy"
    
    merge_pseudo_labels(BEST_BASELINE, SYNTHETIC_DATA, OUTPUT_DATA)
