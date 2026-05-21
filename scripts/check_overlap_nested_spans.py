def check_original_iob_overlaps(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        # Split the file into sentences by blank lines
        blocks = f.read().strip().split('\n\n')

    total_sentences = len(blocks)
    sentences_with_overlaps = 0
    sentences_with_nested = 0
    total_multiple_spans = 0

    for block in blocks:
        lines = block.split('\n')
        if not lines: continue

        # Determine the maximum number of columns in this sentence block
        max_cols = max(len(line.split()) for line in lines)
        if max_cols < 2:
            continue

        all_spans = []

        # Iterate horizontally through each tag column (skipping column 0, which is the word)
        for col_idx in range(1, max_cols):
            current_span = None
            for i, line in enumerate(lines):
                parts = line.split()
                # If a row has fewer columns, treat the missing column as a "0" (no tag)
                tag = parts[col_idx] if col_idx < len(parts) else "0"

                if tag.startswith("B-"):
                    if current_span:
                        all_spans.append(current_span)
                    current_span = {"label": tag[2:], "start": i, "end": i}
                elif tag.startswith("I-") and current_span and tag[2:] == current_span["label"]:
                    current_span["end"] = i
                else:
                    if current_span:
                        all_spans.append(current_span)
                        current_span = None
            if current_span:
                all_spans.append(current_span)

        if len(all_spans) > 1:
            total_multiple_spans += 1

        # Check for overlaps and nested spans mathematically
        has_overlap = False
        has_nested = False

        for i in range(len(all_spans)):
            for j in range(i + 1, len(all_spans)):
                span_a = all_spans[i]
                span_b = all_spans[j]

                # Nested: Span A is entirely inside Span B, or vice versa
                if (span_a["start"] >= span_b["start"] and span_a["end"] <= span_b["end"]) or \
                   (span_b["start"] >= span_a["start"] and span_b["end"] <= span_a["end"]):
                    has_nested = True
                
                # Overlapping: Not nested, but their boundaries intersect
                elif span_a["start"] <= span_b["end"] and span_b["start"] <= span_a["end"]:
                    has_overlap = True

        if has_overlap: sentences_with_overlaps += 1
        if has_nested: sentences_with_nested += 1

    print(f"Total Sentences Analyzed: {total_sentences}")
    print(f"Sentences with >1 Span:   {total_multiple_spans}")
    print(f"Sentences with Nested:    {sentences_with_nested}")
    print(f"Sentences with Overlaps:  {sentences_with_overlaps}")

# Point this directly to the original, multi-column EDT train.iob file
check_original_iob_overlaps('data/train.iob')
