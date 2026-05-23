import spacy
from spacy.tokens import DocBin, Span

def parse_multicolumn_to_spancat(input_path, output_path):
    nlp = spacy.blank("en")
    db = DocBin()
    
    with open(input_path, 'r', encoding='utf-8') as f:
        # Split by empty lines to isolate sentences
        blocks = f.read().strip().split('\n\n')
        
    for block in blocks:
        lines = block.split('\n')
        if not lines: continue
        
        # Extract the raw words to build the document
        words = [line.split()[0] for line in lines if line.strip()]
        doc = spacy.tokens.Doc(nlp.vocab, words=words)
        spans_list = []
        
        # Find how many columns exist in this specific sentence block
        max_cols = max(len(line.split()) for line in lines)
        
        # Loop through every column (skipping column 0, which is the text)
        for col_idx in range(1, max_cols):
            start = None
            label = None
            for i, line in enumerate(lines):
                parts = line.split()
                tag = parts[col_idx] if col_idx < len(parts) else "0"
                
                if tag.startswith("B-"):
                    if start is not None:
                        spans_list.append(Span(doc, start, i, label=label))
                    start = i
                    label = tag[2:]
                elif tag.startswith("I-") and start is not None and tag[2:] == label:
                    pass # The span continues
                else:
                    if start is not None:
                        spans_list.append(Span(doc, start, i, label=label))
                        start = None
                        label = None
            # Catch spans that end on the very last word of the sentence
            if start is not None:
                spans_list.append(Span(doc, start, len(lines), label=label))
                
        # Save the spans to the spancat dictionary (NOT doc.ents)
        doc.spans["engagement"] = spans_list
        db.add(doc)
        
    db.to_disk(output_path)
    print(f"Successfully compiled {output_path} with overlapping spans preserved.")

# Run for all your original data splits
parse_multicolumn_to_spancat('data/train.iob', 'data/train.spacy')
parse_multicolumn_to_spancat('data/dev.iob', 'data/dev.spacy')
parse_multicolumn_to_spancat('data/test.iob', 'data/test.spacy')
