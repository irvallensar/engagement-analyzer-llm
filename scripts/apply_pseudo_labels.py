import spacy
from spacy.tokens import DocBin, Span
import os

# DUAL-THRESHOLD STRATEGY:
# Forgiving threshold for the primary label Qwen was explicitly asked to generate.
PRIMARY_CONFIDENCE_THRESHOLD = 0.5
# Strict threshold for recovering unprompted background/secondary labels.
SECONDARY_CONFIDENCE_THRESHOLD = 0.7

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
    
    print("Beginning QA and Knowledge Distillation scan...")

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
        
        # STEP 3: The QA Gate (Using PRIMARY Threshold)
        qa_passed = True
        verified_primary_spans = []
        occupancy_map = set()
        
        for llm_span in llm_primary_spans:
            llm_tokens = set(range(llm_span.start, llm_span.end))
            match_found = False
            
            for i, p_span in enumerate(predicted_spans):
                if p_span.label_ == llm_span.label_:
                    teacher_tokens = set(range(p_span.start, p_span.end))
                    
                    if llm_tokens.intersection(teacher_tokens):
                        score = span_scores[i] if has_scores else 1.0
                        
                        # USE THE LOWER PRIMARY THRESHOLD HERE
                        if score >= PRIMARY_CONFIDENCE_THRESHOLD:
                            match_found = True
                            verified_span = Span(doc, p_span.start, p_span.end, label=p_span.label_)
                            verified_primary_spans.append(verified_span)
                            occupancy_map.update(teacher_tokens)
                            break 
            
            if not match_found:
                qa_passed = False
                break
                
        if not qa_passed:
            rejected_sentences_count += 1
            continue
            
        accepted_sentences_count += 1

        # STEP 4: Secondary Label Distillation (Using SECONDARY Threshold)
        final_merged_spans = verified_primary_spans.copy()
        
        for i, p_span in enumerate(predicted_spans):
            if p_span.text in [s.text for s in verified_primary_spans] and p_span.label_ in [s.label_ for s in verified_primary_spans]:
                continue
                
            score = span_scores[i] if has_scores else 1.0

            # USE THE STRICT SECONDARY THRESHOLD HERE
            if score >= SECONDARY_CONFIDENCE_THRESHOLD:
                teacher_tokens = set(range(p_span.start, p_span.end))

                if not teacher_tokens.intersection(occupancy_map):
                    new_secondary_span = Span(doc, p_span.start, p_span.end, label=p_span.label_)
                    final_merged_spans.append(new_secondary_span)
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
    print(f"\n[SUCCESS] QA-verified synthetic training data saved to: {output_path}")

if __name__ == "__main__":
    BEST_BASELINE_FOLD = "en_engagement_LSTM"
    SYNTHETIC_DATA = "data/synthetic_few_shot_v3.spacy"
    OUTPUT_DATA = "data/synthetic_pseudo_labeled_few_shot_v3.spacy"
    
    merge_and_qa_pseudo_labels(BEST_BASELINE_FOLD, SYNTHETIC_DATA, OUTPUT_DATA)
