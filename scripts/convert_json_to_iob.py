import json
import spacy

# Load a blank English tokenizer
nlp = spacy.blank("en")

def convert_to_iob(text, label, span_text):
    doc = nlp(text)
    iob_tags = ["O"] * len(doc)
    
    start_char = text.find(span_text)
    if start_char == -1:
        return None 
        
    end_char = start_char + len(span_text)
    
    # Try strict first, fallback to contract, then expand to save the data
    span_obj = doc.char_span(start_char, end_char, alignment_mode="strict")
    if span_obj is None:
        span_obj = doc.char_span(start_char, end_char, alignment_mode="contract")
    if span_obj is None:
        span_obj = doc.char_span(start_char, end_char, alignment_mode="expand")
        
    if span_obj is None:
        return None # Only fail if it's completely unalignable
        
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
    # Point to the 8,000 data augmented set
    input_file = "data/synthetic_balanced.jsonl"
    output_file = "data/synthetic_train.iob"
    
    success_count = 0
    fail_count = 0
    
    with open(input_file, "r", encoding="utf-8") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:
         
        for line in infile:
            data = json.loads(line.strip())
            iob_string = convert_to_iob(data["text"], data["label"], data["span"])
            
            if iob_string:
                outfile.write(iob_string + "\n")
                success_count += 1
            else:
                fail_count += 1
                
    print(f"Successfully converted {success_count} sentences to IOB format.")
    if fail_count > 0:
        print(f"Skipped {fail_count} sentences due to complex tokenization boundaries.")

if __name__ == "__main__":
    main()
