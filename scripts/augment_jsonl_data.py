import mlx_lm
import json
import random

TARGET_CLASSES = {
    "CITATION":    2400,
    "SOURCES":     2200,
    "JUSTIFYING":  2082,
    "ENDOPHORIC":  1982,
}

# ── Domains to force semantic diversity ─────────────────────────────────────
DOMAINS = [
    "Computer Science", "Sociology", "Molecular Biology", "History",
    "Macroeconomics", "Linguistics", "Quantum Physics", "Cognitive Psychology",
    "Philosophy of Mind", "Clinical Medicine", "Environmental Science"
]

# ── Seed triggers to force surface-form variety within each category ─────────
SEED_TRIGGERS = {
    "ENDOPHORIC": [
        "in Table", "in Figure", "above", "below", "the following",
        "as shown in", "see section", "the aforementioned",
        "in the previous section", "as illustrated in",
        "as depicted in", "the latter", "the former"
    ],
    "JUSTIFYING": [
        "thus", "therefore", "because", "hence", "consequently",
        "so", "given that", "since", "as a result", "for this reason",
        "owing to", "due to", "this is why", "it follows that"
    ],
    "SOURCES": [
        "researchers", "scholars", "studies", "the literature",
        "previous work", "the author", "critics", "proponents",
        "opponents", "analysts", "experts", "investigators", "theorists"
    ],
    "PROCLAIM": [
        "clearly", "obviously", "undoubtedly", "certainly",
        "it is evident that", "demonstrably", "unquestionably",
        "it is clear that", "as is well known", "importantly",
        "crucially", "it must be noted that"
    ],
    "CITATION": [
        "as noted in", "as argued by", "as shown by",
        "as demonstrated in", "as reported by",
        "as discussed in", "as outlined in", "as suggested by",
        "as observed by", "as concluded by"
    ]
}

# ── System prompt used in train.jsonl (must match exactly) ───────────────────
SYSTEM_PROMPT = (
    "You are an expert linguistic annotator. Analyze the sentence and extract all Engagement markers. "
    "Output a JSON array of dictionaries with 'label' and 'span' keys. "
    "The 10 valid tags are: ATTRIBUTION, CITATION, COUNTER, DENY, ENDOPHORIC, ENTERTAIN, JUSTIFYING, MONOGLOSS, PROCLAIM, SOURCES. "
    "Example Input: I do not believe this approach works. "
    "Example Output: [{\"label\": \"DENY\", \"span\": \"not\"}, {\"label\": \"ENTERTAIN\", \"span\": \"believe\"}] "
    "If there are no markers, output []."
)

# ─────────────────────────────────────────────────────────────────────────────

print("Loading MLX model for synthetic data generation...")
model_id = "mlx-community/Qwen2.5-32B-Instruct-4bit"
model, tokenizer = mlx_lm.load(model_id)
print("Model loaded.\n")


def build_prompt(category: str, domain: str, trigger: str) -> str:
    return (
        f"You are an expert academic writer in the field of {domain}.\n"
        f"Write exactly 10 distinct academic sentences that contain a {category} "
        f"engagement marker. Each sentence MUST naturally use the trigger phrase "
        f"\"{trigger}\" or a close variant as the {category} marker.\n\n"
        f"DIVERSITY RULES:\n"
        f"- Vary sentence length (mix short and long sentences).\n"
        f"- Place the marker at different positions: beginning, middle, and end.\n"
        f"- Do NOT repeat the same grammatical structure across sentences.\n"
        f"- Each sentence must be on its own line in this exact format:\n"
        f"  <sentence> | <marker span>\n\n"
        f"Example for ENDOPHORIC with trigger 'in Table':\n"
        f"  The correlation coefficients shown in Table 2 confirm the hypothesis. | in Table 2\n\n"
        f"Output ONLY the 10 formatted lines. No numbering, no extra text."
    )


def parse_response(response: str, category: str, seen: set) -> list:
    """Parse 'sentence | marker' lines into synthetic dataset entries."""
    results = []
    for line in response.split('\n'):
        line = line.strip()
        if '|' not in line:
            continue

        parts = line.split('|', 1)
        if len(parts) < 2:
            continue

        sentence = parts[0].strip()
        marker = parts[1].strip()

        # Strip rogue numbering e.g. "1. " or "1) "
        if sentence and sentence[0].isdigit() and len(sentence) > 2 and sentence[1:3] in ['. ', ') ']:
            sentence = sentence[3:].strip()

        # Filter artifacts
        if not sentence or not marker:
            continue
        if '-DOCSTART-' in sentence or '-DOCSTART-' in marker:
            continue
        if marker not in sentence:
            continue  # marker must actually appear in sentence
        if len(sentence.split()) < 5:
            continue  # skip suspiciously short sentences

        # Deduplication
        key = sentence.lower().strip()
        if key in seen:
            continue
        seen.add(key)

        results.append({
            "sentence": sentence,
            "label": category,
            "span": marker
        })

    return results


def to_mlx_format(entry: dict) -> dict:
    """Convert a synthetic entry to the MLX chat format used in train.jsonl."""
    # REPLACED: Enforcing pure [{"label": "X", "span": "Y"}] schema
    span_json = json.dumps([{
        "label": entry["label"],
        "span": entry["span"]
    }], ensure_ascii=False)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": entry["sentence"]},
            {"role": "assistant", "content": span_json}
        ]
    }


def generate_synthetic_data():
    synthetic_entries = []
    seen_sentences = set()

    for category, target_count in TARGET_CLASSES.items():
        print(f"Generating {target_count} sentences for {category}...")
        collected = []
        triggers = SEED_TRIGGERS[category]
        attempts = 0
        max_attempts = (target_count // 10) * 8  
        
        while len(collected) < target_count and attempts < max_attempts:
            domain = random.choice(DOMAINS)
            trigger = random.choice(triggers)
            prompt = build_prompt(category, domain, trigger)

            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            response = mlx_lm.generate(
                model, tokenizer,
                prompt=formatted,
                max_tokens=1024,
                verbose=False
            )

            parsed = parse_response(response, category, seen_sentences)
            collected.extend(parsed)
            attempts += 1

            print(f"  [{category}] {len(collected)}/{target_count} collected "
                  f"(attempt {attempts}, domain={domain}, trigger='{trigger}')")

        # Trim to exact target in case we overshot
        collected = collected[:target_count]
        synthetic_entries.extend(collected)
        print(f"  -> Final count for {category}: {len(collected)}\n")

    # ── Save raw synthetic data for inspection ───────────────────────────────
    raw_path = 'data/synthetic_json_raw.jsonl'
    with open(raw_path, 'w', encoding='utf-8') as f:
        for entry in synthetic_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"[INFO] Raw synthetic data saved to {raw_path}")

    # ── Convert to MLX chat format and save ──────────────────────────────────
    mlx_path = 'data/synthetic_json.jsonl'
    with open(mlx_path, 'w', encoding='utf-8') as f:
        for entry in synthetic_entries:
            mlx_entry = to_mlx_format(entry)
            f.write(json.dumps(mlx_entry, ensure_ascii=False) + '\n')
    print(f"[INFO] MLX-format synthetic data saved to {mlx_path}")

    total = len(synthetic_entries)
    print(f"\n[SUCCESS] Generated {total} diverse synthetic sentences!")
    print("Breakdown:")
    from collections import Counter
    counts = Counter(e["label"] for e in synthetic_entries)
    for label, count in sorted(counts.items()):
        print(f"  {label:15s}: {count}")

    return synthetic_entries


if __name__ == "__main__":
    generate_synthetic_data()
