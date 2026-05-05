import spacy
from spacy.tokens import DocBin, Span
import os

def robust_iob_parser(iob_path, spacy_path):
    if not os.path.exists(iob_path):
        return
        
    print(f"Reading {iob_path}...")
    nlp = spacy.blank("en")
    doc_bin = DocBin()
    
    with open(iob_path, "r", encoding="utf-8") as f:
        sentences = f.read().strip().split("\n\n")
        
    success = 0
    for sent in sentences:
        lines = [line.strip() for line in sent.split("\n") if line.strip()]
        if not lines:
            continue
            
        words = []
        tags = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                words.append(parts[0])
                
                # SMART SCANNER: Look through all columns for a tag
                found_tag = "O"
                for p in parts[1:]:
                    if p.startswith("B-") or p.startswith("I-"):
                        found_tag = p
                        break
                tags.append(found_tag)
        
        if not words:
            continue
            
        doc = spacy.tokens.Doc(nlp.vocab, words=words)
        
        spans = []
        start_idx = None
        current_label = None
        
        for i, tag in enumerate(tags):
            if tag.startswith("B-"):
                if start_idx is not None: 
                    spans.append(Span(doc, start_idx, i, label=current_label))
                start_idx = i
                current_label = tag[2:]
            elif tag.startswith("I-"):
                if start_idx is None: 
                    start_idx = i
                    current_label = tag[2:]
                elif tag[2:] != current_label: 
                    spans.append(Span(doc, start_idx, i, label=current_label))
                    start_idx = i
                    current_label = tag[2:]
            elif tag == "O":
                if start_idx is not None:
                    spans.append(Span(doc, start_idx, i, label=current_label))
                    start_idx = None
                    current_label = None
                    
        if start_idx is not None:
            spans.append(Span(doc, start_idx, len(tags), label=current_label))
            
        try:
            doc.ents = spans
            doc_bin.add(doc)
            success += 1
        except ValueError:
            pass 
            
    doc_bin.to_disk(spacy_path)
    print(f"  -> Saved {success} fully annotated sentences to {spacy_path}")

print("Rebuilding binary files with Multi-Column extraction...")
robust_iob_parser("data/train.iob", "data/train.spacy")
robust_iob_parser("data/dev.iob", "data/dev.spacy")

if os.path.exists("data/test.iob"):
    robust_iob_parser("data/test.iob", "data/test.spacy")
    
print("Done. Ready for real training.")
