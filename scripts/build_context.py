import sys

def add_sliding_context(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    sentences = []
    current_sentence = []
    
    # Parse the IOB file into a list of sentences
    for line in lines:
        line = line.strip()
        if not line:
            if current_sentence:
                sentences.append(current_sentence)
                current_sentence = []
        else:
            current_sentence.append(line)
    if current_sentence:
        sentences.append(current_sentence)

    # Write the new Context-Aware IOB file
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(len(sentences)):
            # 1. Write the Context (Previous Sentence)
            if i == 0:
                f.write("[PAD]\tO\n") # Start of file has no preceding sentence
            else:
                for token_line in sentences[i-1]:
                    parts = token_line.split('\t')
                    word = parts[0]
                    # Force all context tags to 'O' so it is purely used as background info
                    f.write(f"{word}\tO\n")
            
            # Add a separator token to help RoBERTa distinguish context from target
            f.write("[SEP]\tO\n")

            # 2. Write the Target (Current Sentence with actual IOB tags)
            for token_line in sentences[i]:
                f.write(f"{token_line}\n")
            
            # Blank line to indicate end of spaCy document
            f.write("\n")

if __name__ == "__main__":
    add_sliding_context("data/combined_train.iob", "data/context_combined_train.iob")
    add_sliding_context("data/dev.iob", "data/context_dev.iob")
    add_sliding_context("data/test.iob", "data/context_test.iob")
    print("Context-aware IOB files generated successfully.")
