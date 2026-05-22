import random

def extract_samples(iob_file_path, output_file_path, samples_per_label=50):
    # Read the full IOB file content
    with open(iob_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # Split text into sentence blocks using blank lines
    blocks = [b.strip() for b in content.split('\n\n') if b.strip()]
    # Store sentences grouped by citation-related labels
    categorized_sentences = {
        "CITATION": [],
        "SOURCES": [],
        "JUSTIFYING": [],
        "ENDOPHORIC": []
    }
    # Process each sentence block separately
    for block in blocks:
        lines = block.split('\n')
        words = []
        labels_in_sentence = set()  # Using set avoids duplicate labels

        for line in lines:
            parts = line.split()
            # Ensure the line contains both token and tag
            if len(parts) >= 2:
                word = parts[0]
                tag = parts[1]
                words.append(word)
                # Only collect beginning entity tags (B-*)
                if tag.startswith('B-'):
                    labels_in_sentence.add(tag[2:])  # Removes "B-" prefix
        # Reconstruct sentence from token list
        sentence_text = " ".join(words)
        # Save sentence under each detected label category
        for label in labels_in_sentence:
            if label in categorized_sentences:
                categorized_sentences[label].append(sentence_text)
                
    # Write sampled sentences into output file
    with open(output_file_path, 'w', encoding='utf-8') as out:
        for label, sentences in categorized_sentences.items():
            out.write(f"=== {label} SAMPLES ===\n")
            # Prevent sampling more items than available
            sample_size = min(samples_per_label, len(sentences))
            # Randomly select sentences for quality checking
            sampled = random.sample(sentences, sample_size)
            for i, sent in enumerate(sampled, 1):
                out.write(f"{i}. {sent}\n")
            out.write("\n")
    print(f"Sampling complete! Check {output_file_path}")

# Main execution:
# Extracts sample sentences from the training data
extract_samples(
    'data/synthetic_train.iob',
    'quality_review_samples.txt'
)
