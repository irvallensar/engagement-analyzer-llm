import spacy
# Imports the spaCy NLP library

from spacy.tokens import DocBin, Span
# DocBin:
# Efficient binary storage format for spaCy Docs
import os
import sys
# Used for command-line arguments

def robust_iob_parser(iob_path, spacy_path):
    # Check whether the input IOB file exists
    # If not, immediately stop the function
    if not os.path.exists(iob_path):
        return

    print(f"Reading {iob_path}...")
    # Create a blank English NLP pipeline
    # We only need vocabulary/token structure here
    # No pretrained model needed
    nlp = spacy.blank("en")
    # DocBin stores multiple spaCy Doc objects efficiently
    # Later saved as .spacy binary file
    doc_bin = DocBin()
    # Open the IOB annotation file
    with open(iob_path, "r", encoding="utf-8") as f:
        # Read entire file as text
        # strip() removes trailing whitespace
        #
        # split("\n\n") separates sentences
        #
        # In IOB format:
        # blank line = sentence boundary
        sentences = f.read().strip().split("\n\n")
    # Counter for successfully processed sentences
    success = 0
    # Loop through every sentence block
    for sent in sentences:
        # Split sentence into lines
        #
        # Example:
        # because B-JUSTIFYING
        # they    O
        # work    O
        #
        # line.strip() removes spaces
        # if line.strip() prevents empty lines
        lines = [line.strip() for line in sent.split("\n") if line.strip()]
        # Skip completely empty sentence blocks
        if not lines:
            continue
        # Store tokens/words
        words = []
        # Store BIO tags
        tags = []
        # Process each token line
        for line in lines:
            # Split columns by whitespace
            #
            # Example:
            # ["because", "B-JUSTIFYING"]
            parts = line.split()
            # Ensure line contains at least:
            # token + one tag
            if len(parts) >= 2:
                # First column is always token
                words.append(parts[0])

                # SMART SCANNER:
                # Some datasets contain multiple columns
                #
                # Example:
                # token POS B-LABEL
                #
                # This loop searches all remaining columns
                # until it finds BIO tag
                found_tag = "O"

                for p in parts[1:]:
                    # BIO tags start with:
                    # B- or I-
                    if p.startswith("B-") or p.startswith("I-"):
                        found_tag = p
                        break

                # Save detected tag
                tags.append(found_tag)

        # Skip if sentence somehow contains no tokens
        if not words:
            continue
        # Create spaCy Doc object manually
        # words=words preserves exact tokenization
        doc = spacy.tokens.Doc(nlp.vocab, words=words)
        # List that will store entity spans
        spans = []
        # Starting token index of current entity
        start_idx = None
        # Current entity label
        current_label = None
        # Enumerate gives:
        # i = token index
        # tag = BIO label
        #
        # Example:
        # i=0, tag="B-JUSTIFYING"
        for i, tag in enumerate(tags):
            # Beginning of new entity
            if tag.startswith("B-"):
                # If previous entity exists,
                # close/save it first
                if start_idx is not None:
                    spans.append(
                        Span(
                            doc,
                            start_idx,
                            i,
                            label=current_label
                        )
                    )
                # Start new entity
                start_idx = i
                # Remove "B-" prefix
                current_label = tag[2:]
            # Inside existing entity
            elif tag.startswith("I-"):
                # Invalid I- without previous B-
                #
                # Repair strategy:
                # treat it as new entity
                if start_idx is None:
                    start_idx = i
                    current_label = tag[2:]
                # Label changed unexpectedly
                #
                # Example:
                # I-CITATION after I-JUSTIFYING
                #
                # Close previous entity and start new one
                elif tag[2:] != current_label:

                    spans.append(
                        Span(
                            doc,
                            start_idx,
                            i,
                            label=current_label
                        )
                    )
                    start_idx = i
                    current_label = tag[2:]
            # Outside any entity
            elif tag == "O":
                # If entity currently open,
                # close it
                if start_idx is not None:
                    spans.append(
                        Span(
                            doc,
                            start_idx,
                            i,
                            label=current_label
                        )
                    )
                    # Reset tracking
                    start_idx = None
                    current_label = None

        # Sentence ended while entity still open
        #
        # Example:
        # because B-JUSTIFYING
        # this    I-JUSTIFYING
        #
        # No O after it
        #
        # So we must manually close entity
        if start_idx is not None:
            spans.append(
                Span(
                    doc,
                    start_idx,
                    len(tags),
                    label=current_label
                )
            )

        try:
            # Assign entities to spaCy Doc
            doc.ents = spans
            # Add processed doc into DocBin
            doc_bin.add(doc)
            # Count successful sentence
            success += 1
        except ValueError:

            # spaCy throws ValueError if:
            # overlapping entities exist
            #
            # Instead of crashing,
            # silently skip bad sentence
            pass

    # Save DocBin to disk
    #
    # Creates binary .spacy training file
    doc_bin.to_disk(spacy_path)
    print(f"  -> Saved {success} fully annotated sentences to {spacy_path}")


if __name__ == "__main__":
    # sys.argv contains terminal arguments
    #
    # Example:
    # python rebuild_spacy.py input.iob output.spacy
    #
    # argv[0] = script name
    # argv[1] = input
    # argv[2] = output
    if len(sys.argv) == 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]

        print(f"Rebuilding: {input_file} -> {output_file}")

        robust_iob_parser(input_file, output_file)
    else:
        # Default fallback mode
        #
        # If no terminal arguments supplied,
        # rebuild standard dataset files
        print("Rebuilding default binary files...")
        robust_iob_parser(
            "data/synthetic_train.iob",
            "data/syntethic_train.spacy"
        )
        robust_iob_parser(
            "data/dev.iob",
            "data/dev.spacy"
        )
        # Only rebuild test file if it exists
        if os.path.exists("data/test.iob"):
            robust_iob_parser(
                "data/test.iob",
                "data/test.spacy"
            )
        print("Done. Ready for real training.")
