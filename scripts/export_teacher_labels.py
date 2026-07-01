import spacy
import json
import glob
from collections import Counter

# 1. Configuration
# Put the path to your folder containing the 64 JSON files here:
BATCH_FILES_PATTERN = "output/annotation_batches/*_annotation_data.json" 
OUTPUT_FILE = "pseudo_labeled_corpus.jsonl"

# The target number of sentences you want to extract for your minority classes
TARGET_COUNT = 2000  
TARGET_CLASSES = {"JUSTIFYING", "ENDOPHORIC", "SOURCES", "CITATION"}

# 2. Load the Gold Standard IDs to prevent duplicates
# You will need to extract the IDs from your current Gold .spacy/IOB files and put them in a text file.
# For now, let's pretend we load them into a set:
gold_sentence_ids = set() 
# Example: gold_sentence_ids.add("0413d.xml_s1.5;p4.47")

print("Loading RoBERTa+LSTM Teacher Model...")
nlp = spacy.load("en_engagement_LSTM")

class_counter = Counter()
extracted_data = []

# 3. Process the Tier 2 JSON files
file_list = glob.glob(BATCH_FILES_PATTERN)

for file_path in file_list:
    print(f"\nProcessing {file_path}...")
    
    with open(file_path, "r", encoding="utf-8") as f:
        batch_data = json.load(f)
        
    for sentence_id, text in batch_data:
        # DATA LEAKAGE PREVENTION: Skip if it's already in the Gold Standard
        if sentence_id in gold_sentence_ids:
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
            
            # Print progress
            print(f"Counts: JUSTIFYING: {class_counter['JUSTIFYING']} | ENDOPHORIC: {class_counter['ENDOPHORIC']} | SOURCES: {class_counter['SOURCES']} | CITATION: {class_counter['CITATION']}", end="\r")

        # Stop entirely if we hit 2,000 for all our target classes
        if all(class_counter[c] >= TARGET_COUNT for c in TARGET_CLASSES):
            print("\n\nTARGET REACHED FOR ALL CLASSES! Stopping extraction.")
            break
            
    if all(class_counter[c] >= TARGET_COUNT for c in TARGET_CLASSES):
        break

# 4. Export the results
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for entry in extracted_data:
        f.write(json.dumps(entry) + "\n")
        
print(f"\nSuccessfully exported {len(extracted_data)} pseudo-labeled sentences to {OUTPUT_FILE}.")
