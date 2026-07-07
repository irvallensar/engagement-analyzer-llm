import spacy
from spacy.tokens import DocBin, Span
import os

# We remain strict for capturing unprompted, accidental background labels.
BACKGROUND_CONFIDENCE_THRESHOLD = 0.7

def merge_and_qa_pseudo_labels(baseline_model_path, synthetic_input_path, output_path):

    print(f"Loading teacher model from: {baseline_model_path}")
    nlp = spacy.load(baseline_model_path)
    
    print(f"Reading pristine synthetic documents using model vocabulary...")
    doc_bin_in = DocBin().from_disk(synthetic_input_path)

    raw_docs = list(doc_bin_in.get_docs(nlp.vocab))
    processed_doc_bin = DocBin()

    total_sentences = len(raw_docs)
    accepted_sentences_count = 0
    rejected_sentences_count = 0
    recovered_secondary_labels = 0 
    
    print("Beginning character-aligned QA and Knowledge Distillation scan...")

    for doc in raw_docs:
        llm_primary_spans = list(doc.spans.get("sc", []))
        
        if not llm_primary_spans:
            rejected_sentences_count += 1
            continue
            
        predicted_doc = nlp(doc.text)
        predicted_spans = predicted_doc.spans.get("sc", [])
        
        try:
            span_scores = predicted_doc.spans["sc"].attrs.get("scores", [])
        except AttributeError:
            span_scores = []
            
        has_scores = len(predicted_spans) == len(span_scores) and len(predicted_spans) > 0
        
        # STEP 1: Character-Level QA Gate
        qa_passed = True
        verified_primary_spans = []
        occupancy_map = set() # Tracked via token indices inside the target 'doc'
        
        for llm_span in llm_primary_spans:
            match_found = False
            
            for i, p_span in enumerate(predicted_spans):
                if p_span.label_ == llm_span.label_:
                    
                    # ABSOLUTE CHARACTER OVERLAP CHECK:
                    # Two character segments overlap if max(start1, start2) < min(end1, end2)
                    if max(llm_span.start_char, p_span.start_char) < min(llm_span.end_char, p_span.end_char):
                        match_found = True
                        
                        # Safely map teacher character bounds back onto target doc tokens smoothly
                        verified_span = doc.char_span(
                            p_span.start_char, 
                            p_span.end_char, 
                            label=p_span.label_,
                            alignment_mode="expand"
                        )
                        
                        if verified_span is not None:
                            verified_primary_spans.append(verified_span)
                            occupancy_map.update(range(verified_span.start, verified_span.end))
                        else:
                            # Robust fallback to original tokenization bounds if character alignment fails
                            verified_primary_spans.append(llm_span)
                            occupancy_map.update(range(llm_span.start, llm_span.end))
                        break 
            
            if not match_found:
                qa_passed = False
                break
                
        if not qa_passed:
            rejected_sentences_count += 1
            continue
            
        accepted_sentences_count += 1

        # STEP 2: Background Secondary Label Distillation
        final_merged_spans = verified_primary_spans.copy()
        
        for i, p_span in enumerate(predicted_spans):
            # Skip if this is the primary label we verified in Step 1
            is_primary = False
            for v_span in verified_primary_spans:
                if p_span.start_char == v_span.start_char and p_span.end_char == v_span.end_char and p_span.label_ == v_span.label_:
                    is_primary = True
                    break
            if is_primary:
                continue
                
            score = span_scores[i] if has_scores else 1.0

            # Enforce strict 0.7 rule for unexpected background findings
            if score >= BACKGROUND_CONFIDENCE_THRESHOLD:
                s_span = doc.char_span(p_span.start_char, p_span.end_char, label=p_span.label_, alignment_mode="expand")
                
                if s_span is not None:
                    teacher_tokens = set(range(s_span.start, s_span.end))
                    # Prevent background labels from clipping over verified primary bounds
                    if not teacher_tokens.intersection(occupancy_map):
                        final_merged_spans.append(s_span)
                        recovered_secondary_labels += 1
                        occupancy_map.update(teacher_tokens)

        doc.spans["sc"] = final_merged_spans
        processed_doc_bin.add(doc)
        
    print(f"\n=== QA & Distillation Complete ===")
    print(f"Total LLM Sentences Scanned: {total_sentences}")
    print(f"-> Sentences PASSED & Preserved: {accepted_sentences_count}")
    print(f"-> Sentences DISCARDED (Failed QA): {rejected_sentences_count}")
    print(f"-> Missing Secondary Labels Recovered: {recovered_secondary_labels}")
    
    processed_doc_bin.to_disk(output_path)
    print(f"\n[SUCCESS] Character-aligned QA-verified data saved to: {output_path}")

if __name__ == "__main__":
    BEST_BASELINE_FOLD = "en_engagement_LSTM"
    SYNTHETIC_DATA = "data/synthetic_few_shot_v3.spacy"
    OUTPUT_DATA = "data/synthetic_pseudo_labeled_few_shot_v3.spacy"
    
    merge_and_qa_pseudo_labels(BEST_BASELINE_FOLD, SYNTHETIC_DATA, OUTPUT_DATA)
