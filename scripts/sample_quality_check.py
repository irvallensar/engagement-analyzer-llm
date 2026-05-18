import random

def extract_samples(iob_file_path, output_file_path, samples_per_label=50):
    with open(iob_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split the file into individual sentence blocks (separated by blank lines)
    blocks = [b.strip() for b in content.split('\n\n') if b.strip()]
    
    # Dictionary to hold sentences grouped by the label they contain
    categorized_sentences = {
        "CITATION": [], "SOURCES": [], "JUSTIFYING": [], "ENDOPHORIC": []
    }
    
    for block in blocks:
        lines = block.split('\n')
        words = []
        labels_in_sentence = set()
        
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                word = parts[0]
                tag = parts[-1] # Grabs the last column
                words.append(word)
                if tag.startswith('B-'):
                    labels_in_sentence.add(tag[2:]) # e.g., 'CITATION'
        
        sentence_text = " ".join(words)
        
        # Categorize the sentence
        for label in labels_in_sentence:
            if label in categorized_sentences:
                categorized_sentences[label].append(sentence_text)
                
    # Sample and write to output
    with open(output_file_path, 'w', encoding='utf-8') as out:
        for label, sentences in categorized_sentences.items():
            out.write(f"=== {label} SAMPLES ===\n")
            # Take a random sample, or all if there are fewer than requested
            sample_size = min(samples_per_label, len(sentences))
            sampled = random.sample(sentences, sample_size)
            
            for i, sent in enumerate(sampled, 1):
                out.write(f"{i}. {sent}\n")
            out.write("\n")
            
    print(f"Sampling complete! Check {output_file_path}")

# Run the function
extract_samples('data/synthetic_train.iob', 'quality_review_samples.txt')
