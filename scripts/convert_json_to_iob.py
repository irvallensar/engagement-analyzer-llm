import json
import spacy

# Load a blank English tokenizer (to match RoBERTa's preprocessing behavior)
nlp = spacy.blank("en")

def convert_to_iob(text, label, span_text):
    doc = nlp(text)
    
    # Initialize all tokens to 'O'
    iob_tags = ["O"] * len(doc)
    
    # Find the character start and end of the span
    start_char = text.find(span_text)
    if start_char == -1:
        return None # Span not found, fail safely
    end_char = start_char + len(span_text)
    
    # Use spaCy's alignment to find which tokens fall in these character offsets
    span_obj = doc.char_span(start_char, end_char, alignment_mode="strict")
    
    if span_obj is None:
        return None # The LLM generated a span that breaks across token boundaries
        
    for i, token in enumerate(doc):
        if token.i == span_obj.start:
            iob_tags[i] = f"B-{label}"
        elif span_obj.start < token.i < span_obj.end:
            iob_tags[i] = f"I-{label}"
            
    # Format as CoNLL text
    output_lines = []
    for token, tag in zip(doc, iob_tags):
        output_lines.append(f"{token.text}\t{tag}")
    output_lines.append("") # Empty line to separate sentences
    
    return "\n".join(output_lines)

def main():
    input_file = "data/synthetic_raw.jsonl"
    output_file = "data/synthetic_train.iob"
    
    success_count = 0
    with open(input_file, "r", encoding="utf-8") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:
         
        for line in infile:
            data = json.loads(line.strip())
            iob_string = convert_to_iob(data["text"], data["label"], data["span"])
            
            if iob_string:
                outfile.write(iob_string + "\n")
                success_count += 1
                
    print(f"Successfully converted {success_count} synthetic sentences to strict IOB format.")

if __name__ == "__main__":
    main()
