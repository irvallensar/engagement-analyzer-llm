import json
import spacy
from spacy.tokens import DocBin

def jsonl_to_spancat(input_jsonl, output_spacy):
    nlp = spacy.blank("en")
    db = DocBin()
    
    success = 0
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            text = data["text"]
            label = data["label"]
            span_text = data["span"]
            
            doc = nlp(text)
            start_char = text.find(span_text)
            
            if start_char == -1: 
                continue # Safely drops misaligned hallucinations
                
            end_char = start_char + len(span_text)
            
            # Use expand to flexibly catch LLM spacing weirdness
            span_obj = doc.char_span(start_char, end_char, alignment_mode="expand")
            
            if span_obj is not None:
                span_obj.label_ = label
                doc.spans["sc"] = [span_obj]
                db.add(doc)
                success += 1
                
    db.to_disk(output_spacy)
    print(f"Successfully compiled {success} synthetic sentences directly to {output_spacy}.")

jsonl_to_spancat('data/synthetic_few_shot_v3.jsonl', 'data/synthetic_few_shot_v3.spacy')
