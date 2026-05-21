def check_original_multicolumn_iob(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        # Split file into sentences by double line breaks
        blocks = f.read().strip().split('\n\n')

    total_sentences = len(blocks)
    multiple_spans = 0
    nested_spans = 0
    overlapping_spans = 0

    for block in blocks:
        lines = block.split('\n')
        
        # A list to store extracted spans as (start_idx, end_idx, label)
        extracted_spans = []
        
        # Dictionary to track active spans currently open in each column
        # Format: { column_index: {"label": "TAG", "start": int} }
        active_spans = {}

        for token_idx, line in enumerate(lines):
            parts = line.split()
            if len(parts) < 2: 
                continue
                
            tags = parts[1:] # Grab all tag columns (ignoring the word itself)

            for col_idx, tag in enumerate(tags):
                if tag.startswith('B-'):
                    # If there's already an active span in this column, close it and save it
                    if col_idx in active_spans:
                        extracted_spans.append((active_spans[col_idx]['start'], token_idx, active_spans[col_idx]['label']))
                    # Open the new span
                    active_spans[col_idx] = {'label': tag[2:], 'start': token_idx}
                    
                elif tag.startswith('I-'):
                    continue # Span remains open
                    
                else: # Tag is 'O' or '0'
                    # Close the active span in this column if one exists
                    if col_idx in active_spans:
                        extracted_spans.append((active_spans[col_idx]['start'], token_idx, active_spans[col_idx]['label']))
                        del active_spans[col_idx]

        # End of sentence: Close any spans that are still active
        for col_idx, span_data in active_spans.items():
            extracted_spans.append((span_data['start'], len(lines), span_data['label']))

        # --- Evaluate the extracted spans for this sentence ---
        if len(extracted_spans) > 1:
            multiple_spans += 1

        has_nested = False
        has_overlap = False

        for i in range(len(extracted_spans)):
            for j in range(i + 1, len(extracted_spans)):
                a_start, a_end, _ = extracted_spans[i]
                b_start, b_end, _ = extracted_spans[j]

                # Check for Nested (One span is entirely inside the other, or they are exact duplicates)
                if (a_start <= b_start and b_end <= a_end) or (b_start <= a_start and a_end <= b_end):
                    has_nested = True
                # Check for Partial Overlap (They intersect, but neither fully contains the other)
                elif (a_start < b_end and b_start < a_end):
                    has_overlap = True

        if has_nested:
            nested_spans += 1
        if has_overlap and not has_nested:
            overlapping_spans += 1

    print("=== Original Multi-Column IOB Analysis ===")
    print(f"Total sentences:                  {total_sentences}")
    print(f"Sentences with >1 total spans:    {multiple_spans}")
    print(f"Sentences with nested spans:      {nested_spans}")
    print(f"Sentences with overlapping spans: {overlapping_spans}")

# Run the function on the original Eguchi dataset
check_original_multicolumn_iob('data/train.iob') # Update path if needed
