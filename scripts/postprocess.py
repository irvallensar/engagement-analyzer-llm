def to_spacy_tuples(candidates, llm_labels):
    cand_map = {c["id"]: c for c in candidates}
    spans = set()

    for item in llm_labels:
        if item["label"] == "O":
            continue

        c = cand_map[item["id"]]
        spans.add((
            item["label"],
            c["start"],
            c["end"]
        ))

    return spans
