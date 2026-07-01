import spacy
from spacy.tokens import DocBin
import json

# Load a blank modern English pipeline to handle the tokenization
nlp = spacy.blank("en")
doc_bin = DocBin()

input_file = "pseudo_labeled_corpus.jsonl"
output_file = "data/pseudo_labeled_training_data.spacy"

print(f"Ingesting {input_file} into modern spaCy environment...")
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        doc = nlp(data["text"])
        
        # Initialize the span dictionary
        doc.spans["sc"] = []
        
        # Reconstruct the spans using the imported character offsets
        for span_data in data["spans"]:
            # char_span snaps the character coordinates to the modern token boundaries
            span = doc.char_span(
                span_data["start_char"], 
                span_data["end_char"], 
                label=span_data["label"],
                alignment_mode="strict" # Ensures boundaries remain precise
            )
            
            if span is not None:
                doc.spans["sc"].append(span)
            else:
                print(f"Warning: Could not align span in text: '{data['text']}'")
                
        doc_bin.add(doc)

# Save the final training data binary for DA-RoBERTa
doc_bin.to_disk(output_file)
print(f"Successfully generated {output_file} for DA-RoBERTa training.")
