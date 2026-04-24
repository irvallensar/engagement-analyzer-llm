import json
import os

SYSTEM_PROMPT = (
    "You are an expert linguistic annotator. "
    "Rewrite the provided sentence and wrap all Engagement markers in XML tags corresponding to their category. "
    "The 10 valid tags are: <ATTRIBUTION>, <CITATION>, <COUNTER>, <DENY>, <ENDOPHORIC>, <ENTERTAIN>, <JUSTIFYING>, <MONOGLOSS>, <PROCLAIM>, <SOURCES>. "
    "Example Input: I do not believe this approach works. "
    "Example Output: I do <DENY>not</DENY> <ENTERTAIN>believe</ENTERTAIN> this approach works. "
    "If there are no markers, simply output the original sentence exactly as written."
)

def iob_to_xml(iob_file_path, output_file_path):
    dataset = []
    
    with open(iob_file_path, 'r', encoding='utf-8') as f:
        tokens, labels = [], []
        
        for line in f:
            line = line.strip()
            if not line: # End of sentence
                if tokens:
                    dataset.append(process_sentence(tokens, labels))
                tokens, labels = [], []
                continue
                
            parts = line.split()
            if len(parts) >= 2:
                tokens.append(parts[0])
                labels.append(parts[-1]) # Assuming label is the last column
                
        # Catch the last sentence if no trailing newline
        if tokens:
            dataset.append(process_sentence(tokens, labels))

    # Read the synthetic data we just generated
    synthetic_file = 'data/synthetic_xml.jsonl'
    if os.path.exists(synthetic_file):
        with open(synthetic_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                dataset.append(format_chatml(data["raw_text"], data["tagged_text"]))

    # Save everything to the final training file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry) + '\n')
            
    print(f"[SUCCESS] Converted {len(dataset)} sentences to XML ChatML format!")

def process_sentence(tokens, tags):
    raw_sentence = " ".join(tokens)
    xml_result = []
    current_tag = None
    
    for i, (word, tag) in enumerate(zip(tokens, tags)):
        # Close an active tag if needed
        if current_tag:
            if tag == "O" or tag.startswith("B-") or (tag.startswith("I-") and tag[2:] != current_tag):
                xml_result[-1] += f"</{current_tag}>"
                current_tag = None
                
        # Open a new tag or continue
        if tag.startswith("B-"):
            current_tag = tag[2:]
            xml_result.append(f"<{current_tag}>{word}")
        else:
            xml_result.append(word)
            
    # Close any trailing tag at the end of the sentence
    if current_tag:
        xml_result[-1] += f"</{current_tag}>"
        
    tagged_sentence = " ".join(xml_result)
    return format_chatml(raw_sentence, tagged_sentence)

def format_chatml(raw_text, tagged_text):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyze this sentence:\n\n{raw_text}"},
            {"role": "assistant", "content": tagged_text}
        ]
    }

if __name__ == "__main__":
    # Ensure you point this to your actual raw IOB file
    iob_to_xml('data/train.iob', 'data/train.jsonl')
