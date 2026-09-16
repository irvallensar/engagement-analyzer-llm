import json

def extract_spans_with_context(tokens, labels):
    """
    Extracts B-/I- spans from IOB labels and returns Context Anchoring format.
    Each span includes the label, the span text, and the preceding context.
    """
    spans = []
    i = 0
    while i < len(tokens):
        if labels[i].startswith("B-"):
            label = labels[i][2:]  # strip "B-"
            span_tokens = [tokens[i]]
            j = i + 1
            # Collect continuation tokens
            while j < len(tokens) and labels[j] == f"I-{label}":
                span_tokens.append(tokens[j])
                j += 1
            span_text = " ".join(span_tokens)
            # Context: all tokens before this span, joined
            context_before = " ".join(tokens[:i]) if i > 0 else ""
            spans.append({
                "label": label,
                "span": span_text,
                "context_before": context_before
            })
            i = j
        else:
            i += 1
    return spans


def parse_iob_to_mlx_jsonl(iob_file_path, output_file_path):
    with open(iob_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    dataset = []
    current_tokens = []
    current_labels = []

    system_prompt = (
        "You are an expert linguistic annotator. "
        "Extract Engagement markers and output them as a JSON array. "
        "Each item must follow this format: "
        "[{\"label\": \"CATEGORY\", \"span\": \"target text\", \"context_before\": \"preceding text\"}]. "
        "If there are no Engagement markers, output []."
    )

    def flush(tokens, labels):
        if not tokens:
            return
        sentence = " ".join(tokens)
        spans = extract_spans_with_context(tokens, labels)
        assistant_response = json.dumps(spans, ensure_ascii=False)
        dataset.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this sentence:\n\n{sentence}"},
                {"role": "assistant", "content": assistant_response}
            ]
        })

    for line in lines:
        line = line.strip()
        if not line or line.startswith("-DOCSTART-"):
            flush(current_tokens, current_labels)
            current_tokens, current_labels = [], []
            continue

        parts = line.split('\t')
        if len(parts) >= 2:
            current_tokens.append(parts[0])
            current_labels.append(parts[1])

    # Flush final sentence if file doesn't end with blank line
    flush(current_tokens, current_labels)

    with open(output_file_path, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    # Print stats
    total = len(dataset)
    empty = sum(1 for e in dataset if e["messages"][2]["content"] == "[]")
    print(f"Total examples : {total}")
    print(f"Empty []       : {empty} ({100*empty/total:.1f}%)")
    print(f"With spans     : {total - empty} ({100*(total-empty)/total:.1f}%)")


if __name__ == "__main__":
    import sys
    splits = [
        ("data/train.iob", "data/train.jsonl"),
        ("data/dev.iob", "data/valid.jsonl"),  
        ("data/test.iob",  "data/test.jsonl"),
    ]
    # Default: regenerate all three splits
    for iob_path, out_path in splits:
        import os
        if os.path.exists(iob_path) and iob_path.endswith(".iob"):
            print(f"\nConverting {iob_path} → {out_path}")
            parse_iob_to_mlx_jsonl(iob_path, out_path)
