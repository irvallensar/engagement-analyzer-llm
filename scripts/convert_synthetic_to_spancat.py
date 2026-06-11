import json
import spacy
from spacy.tokens import DocBin

# Convert a JSONL dataset of synthetic examples into a spaCy spancat training file
def jsonl_to_spancat(input_jsonl, output_spacy):
    # Create a blank English pipeline (tokenization only)
    nlp = spacy.blank("en")

    # Container used to store training documents efficiently
    db = DocBin()
    
    # Track how many examples are successfully converted
    success = 0

    # Read the JSONL file line-by-line to avoid loading everything into memory
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            # Each line should be a JSON object
            data = json.loads(line.strip())

            # Expected schema:
            # {
            #   "text": "...",
            #   "label": "...",
            #   "span": "..."
            # }
            text = data["text"]
            label = data["label"]
            span_text = data["span"]
            
            # Create a spaCy Doc from the raw text
            doc = nlp(text)

            # Find the character offset of the target span
            start_char = text.find(span_text)
            
            # Skip examples where the span text cannot be located
            # (often caused by synthetic data hallucinations or mismatches)
            if start_char == -1: 
                continue # Safely drops misaligned hallucinations
                
            # Compute the end character position
            end_char = start_char + len(span_text)
            
            # Convert character offsets into a spaCy Span.
            # "expand" helps recover spans when token boundaries do not
            # perfectly align with the character offsets.
            span_obj = doc.char_span(start_char, end_char, alignment_mode="expand")
            
            # Only keep spans that spaCy was able to construct successfully
            if span_obj is not None:
                # Assign the classification label to the span
                span_obj.label_ = label

                # Store spans under the "sc" key expected by spancat
                doc.spans["sc"] = [span_obj]

                # Add the document to the DocBin dataset
                db.add(doc)

                # Increment success counter
                success += 1
                
    # Write the compiled training data to disk
    db.to_disk(output_spacy)

    # Report how many examples were successfully converted
    print(f"Successfully compiled {success} synthetic sentences directly to {output_spacy}.")

# Convert the synthetic dataset into spaCy's binary training format
jsonl_to_spancat('data/synthetic_zero_shot.jsonl', 'data/synthetic_zero_shot.spacy')
