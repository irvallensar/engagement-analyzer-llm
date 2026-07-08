import spacy
from spacy.tokens import DocBin, Span
import os

# Minimum confidence score required for a pseudo-label to be accepted.
CONFIDENCE_THRESHOLD = 0.7

def merge_and_qa_pseudo_labels(baseline_model_path, synthetic_input_path, output_path):

    print(f"Loading teacher model from: {baseline_model_path}")
    nlp = spacy.load(baseline_model_path)
    
    print(f"Reading pristine synthetic documents using model vocabulary...")
    doc_bin_in = DocBin().from_disk(synthetic_input_path)

    raw_docs = list(doc_bin_in.get_docs(nlp.vocab))
    processed_doc_bin = DocBin()

    # Statistics for reporting
    total_sentences = len(raw_docs)
    accepted_sentences_count = 0
    rejected_sentences_count = 0
    recovered_secondary_labels = 0 
    
    print("Beginning QA and Knowledge Distillation scan...")

    for doc in raw_docs:
        # STEP 1: Extract the original synthetic (gold) annotations from the LLM
        llm_primary_spans = list(doc.spans.get("sc", []))
        
        # If the LLM failed to generate any tags at all, discard the sentence
        if not llm_primary_spans:
            rejected_sentences_count += 1
            continue
            
        # STEP 2: Run the teacher model to predict all spans
        predicted_doc = nlp(doc.text)
        predicted_spans = predicted_doc.spans.get("sc", [])
        
        # Safely extract confidence scores (Handling 2022 architecture quirks)
        try:
            span_scores = predicted_doc.spans["sc"].attrs.get("scores", [])
        except AttributeError:
            span_scores = []
            
        has_scores = len(predicted_spans) == len(span_scores) and len(predicted_spans) > 0
        
        # STEP 3: The QA Gate (Verify Primary LLM Labels)
        qa_passed = True
        verified_primary_spans = []
        occupancy_map = set()
        
        for llm_span in llm_primary_spans:
            llm_tokens = set(range(llm_span.start, llm_span.end))
            match_found = False
            
            for i, p_span in enumerate(predicted_spans):
                # We only check predictions that match the LLM's intended category
                if p_span.label_ == llm_span.label_:
                    teacher_tokens = set(range(p_span.start, p_span.end))
                    
                    # Check for boundary overlap
                    if llm_tokens.intersection(teacher_tokens):
                        score = span_scores[i] if has_scores else 1.0
                        
                        if score >= CONFIDENCE_THRESHOLD:
                            match_found = True
                            # ACCEPT: Override the LLM boundary with the teacher's boundary
                            verified_span = Span(doc, p_span.start, p_span.end, label=p_span.label_)
                            verified_primary_spans.append(verified_span)
                            
                            # Update occupancy map to block secondary labels from overwriting this
                            occupancy_map.update(teacher_tokens)
                            break 
            
            # If even one primary LLM span in this sentence fails verification, the sentence fails QA
            if not match_found:
                qa_passed = False
                break
                
        # If QA failed, throw the entire document in the trash and move to the next one
        if not qa_passed:
            rejected_sentences_count += 1
            continue
            
        accepted_sentences_count += 1

        # STEP 4: Secondary Label Distillation
        final_merged_spans = verified_primary_spans.copy()
        
        for i, p_span in enumerate(predicted_spans):
            # Skip the ones we already accepted as primary spans
            if p_span.text in [s.text for s in verified_primary_spans] and p_span.label_ in [s.label_ for s in verified_primary_spans]:
                continue
                
            score = span_scores[i] if has_scores else 1.0

            if score >= CONFIDENCE_THRESHOLD:
                teacher_tokens = set(range(p_span.start, p_span.end))

                # Only add secondary predictions that do not overlap with our primary labels
                if not teacher_tokens.intersection(occupancy_map):
                    new_secondary_span = Span(doc, p_span.start, p_span.end, label=p_span.label_)
                    final_merged_spans.append(new_secondary_span)
                    recovered_secondary_labels += 1
                    occupancy_map.update(teacher_tokens)

        # Store the final, QA-verified, densified span collection
        doc.spans["sc"] = final_merged_spans
        processed_doc_bin.add(doc)
        
    print(f"\n=== QA & Distillation Complete ===")
    print(f"Total LLM Sentences Scanned: {total_sentences}")
    print(f"-> Sentences PASSED & Preserved: {accepted_sentences_count}")
    print(f"-> Sentences DISCARDED (Failed QA): {rejected_sentences_count}")
    print(f"-> Missing Secondary Labels Recovered: {recovered_secondary_labels}")
    
    processed_doc_bin.to_disk(output_path)
    print(f"\n[SUCCESS] QA-verified synthetic training data saved to: {output_path}")

if __name__ == "__main__":
    # Pointed to the custom package installed in eguchi_env
    BEST_BASELINE_FOLD = "models/roberta_large_teacher/model-best"

    # Input synthetic dataset containing only gold synthetic labels
    SYNTHETIC_DATA = "data/synthetic_few_shot_v3.spacy"

    # Output dataset containing gold labels + pseudo-labels
    OUTPUT_DATA = "data/synthetic_pseudo_labeled_few_shot_v3.spacy"
    
    merge_and_qa_pseudo_labels(BEST_BASELINE_FOLD, SYNTHETIC_DATA, OUTPUT_DATA)
