import spacy
import json
import os
from collections import Counter

# 1. Configuration
# Replace this with the absolute path to the directory containing your 64 JSON files
BATCH_DIRECTORY = "/Users/irvallen/engagement-analyzer-llm/Engagement-Discourse-Treebank-Construction/output/annotation_batches" 
OUTPUT_FILE = "pseudo_labeled_corpus.jsonl"

# The target number of sentences you want to extract for your minority classes
TARGET_COUNT = 2000  
TARGET_CLASSES = {"JUSTIFYING", "ENDOPHORIC", "SOURCES", "CITATION"}

# 2. Load the Gold Standard Texts to prevent duplicates
gold_texts = set()
try:
    with open("gold_texts.txt", "r", encoding="utf-8") as f:
        for line in f:
            gold_texts.add(line.strip())
    print(f"Loaded {len(gold_texts)} Gold Standard sentences for deduplication.")
except FileNotFoundError:
    print("WARNING: gold_texts.txt not found. Running without deduplication!")

print("Loading RoBERTa+LSTM Teacher Model...")
nlp = spacy.load("en_engagement_LSTM")

class_counter = Counter()
extracted_data = []

# 3. Process the Tier 2 JSON files in strict numerical order (1 to 64)
for i in range(1, 65):
    filename = f"{i}_annotation_data.json"
    file_path = os.path.join(BATCH_DIRECTORY, filename)
    
    # Check if the file exists before trying to open it
    if not os.path.exists(file_path):
        print(f"\nSkipping {filename} (File not found at {file_path})")
        continue
        
    print(f"\nProcessing {filename}...")
    
    with open(file_path, "r", encoding="utf-8") as f:
        batch_data = json.load(f)
        
    for sentence_id, text in batch_data:
        clean_text = text.strip()
        
        # DATA LEAKAGE PREVENTION: Skip if the exact text is already in the Gold Standard
        if clean_text in gold_texts:
            continue
            
        # Run the RoBERTa+LSTM inference
        doc = nlp(text)
        
        spans = []
        found_target_class = False
        
        if "sc" in doc.spans:
            for span in doc.spans["sc"]:
                label = span.label_
                spans.append({
                    "start_char": span.start_char,
                    "end_char": span.end_char,
                    "label": label
                })
                
                # Check if this sentence contains one of our target minority classes
                if label in TARGET_CLASSES and class_counter[label] < TARGET_COUNT:
                    class_counter[label] += 1
                    found_target_class = True
        
        # Only save the sentence if it actually helped us find a target class
        if found_target_class:
            extracted_data.append({
                "sentence_id": sentence_id,
                "text": text,
                "spans": spans
            })
            
            # Print progress dynamically
            print(f"Counts: JUSTIFYING: {class_counter['JUSTIFYING']} | ENDOPHORIC: {class_counter['ENDOPHORIC']} | SOURCES: {class_counter['SOURCES']} | CITATION: {class_counter['CITATION']}", end="\r")

        # Stop entirely if we hit 2,000 for all our target classes
        if all(class_counter[c] >= TARGET_COUNT for c in TARGET_CLASSES):
            print("\n\nTARGET REACHED FOR ALL CLASSES! Stopping extraction.")
            break
            
    # Break out of the outer file loop if we hit the target
    if all(class_counter[c] >= TARGET_COUNT for c in TARGET_CLASSES):
        break

# 4. Export the results
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for entry in extracted_data:
        f.write(json.dumps(entry) + "\n")
        
print(f"\nSuccessfully exported {len(extracted_data)} pseudo-labeled sentences to {OUTPUT_FILE}.")
