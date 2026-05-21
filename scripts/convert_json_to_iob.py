import json
import spacy
# Load a blank English tokenizer
#
# We only need tokenization here,
# not POS tagging or parsing
#
# Example:
# "because they work"
# ->
# ["because", "they", "work"]
nlp = spacy.blank("en")
def convert_to_iob(text, label, span_text):
    # Convert raw text into spaCy Doc object
    # spaCy tokenizes the sentence automatically
    # Example:
    # "because they work"
    # ->
    # Doc(tokens)
    doc = nlp(text)
    # Initialize all tokens with "O"
    # O = Outside entity
    # Example:
    # ["O", "O", "O"]
    iob_tags = ["O"] * len(doc)
    # Find character-level start position of the target span inside sentence
    # Example:
    # text = "because they work"
    # span = "because"
    # start_char = 0
    start_char = text.find(span_text)
    # If span not found,
    # conversion impossible
    if start_char == -1:
        return None
    # Compute character end position
    # Example:
    # start=0
    # len("because")=7
    # end=7
    end_char = start_char + len(span_text)
    # Convert character offsets
    # into token-aligned spaCy span
    # alignment_mode="strict"
    # STRICT:
    # exact token boundaries only
    # Example:
    # valid:
    # token perfectly aligned
    # invalid:
    # partial token overlap
    span_obj = doc.char_span(
        start_char,
        end_char,
        alignment_mode="strict"
    )
    # If strict alignment fails,
    # try "contract"
    # CONTRACT:
    # shrink boundaries inward
    # to nearest valid token boundaries
    if span_obj is None:
        span_obj = doc.char_span(
            start_char,
            end_char,
            alignment_mode="contract"
        )
    # If still failing,
    # try "expand"
    # EXPAND:
    # enlarge boundaries outward
    # to nearest token boundaries
    # This is aggressive recovery mode
    if span_obj is None:

        span_obj = doc.char_span(
            start_char,
            end_char,
            alignment_mode="expand"
        )
    # If all alignment methods fail,
    # sentence cannot be converted safely
    if span_obj is None:
        # Return None so caller skips sample
        return None
    # Assign BIO tags to matching tokens
    #
    # enumerate(doc):
    # gives:
    # i     -> token index
    # token -> actual token object
    for i, token in enumerate(doc):
        # First token of entity
        #
        # B- = Beginning
        if token.i == span_obj.start:

            iob_tags[i] = f"B-{label}"
        # Tokens inside entity
        #
        # I- = Inside
        elif span_obj.start < token.i < span_obj.end:

            iob_tags[i] = f"I-{label}"
    # Convert into CoNLL/IOB text format
    # Example:
    # because   B-JUSTIFYING
    # they      O
    # work      O
    output_lines = []
    # Combine tokens + BIO labels
    for token, tag in zip(doc, iob_tags):

        output_lines.append(
            f"{token.text}\t{tag}"
        )
    # Empty line separates sentences
    #
    # Standard CoNLL convention
    output_lines.append("")
    # Join all lines into final string
    return "\n".join(output_lines)


def main():
    # Input synthetic JSONL dataset
    # Example line:
    # {
    #   "text": "...",
    #   "label": "...",
    #   "span": "..."
    # }
    input_file = "data/synthetic_balanced.jsonl"
    output_file = "data/synthetic_train.iob"
    # Successful conversion counter
    success_count = 0
    # Failed conversion counter
    fail_count = 0
    # Open input + output files simultaneously
    # infile:
    # reads JSONL
    # outfile:
    # writes IOB format
    with open(input_file, "r", encoding="utf-8") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:
        # Read JSONL line-by-line
        # JSONL:
        # one JSON object per line
        for line in infile:
            # Convert JSON string into Python dict
            data = json.loads(line.strip())
            # Convert annotation into IOB format
            iob_string = convert_to_iob(
                data["text"],
                data["label"],
                data["span"]
            )
            # If conversion succeeded
            if iob_string:
                # Save converted sentence
                outfile.write(iob_string + "\n")
                success_count += 1
            else:
                # Count failed alignments
                fail_count += 1
    # Final success report
    print(
        f"Successfully converted "
        f"{success_count} sentences to IOB format."
    )
    # Only print failure report if needed
    if fail_count > 0:
        print(
            f"Skipped {fail_count} sentences "
            f"due to complex tokenization boundaries."
        )
        
if __name__ == "__main__":
    main()
