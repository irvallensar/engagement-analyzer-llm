import mlx_lm
import json
import random
import re

# Set how many new sentences you want to generate to balance the 8000+ majority classes
TARGET_CLASSES = {
    "ENDOPHORIC": 2000,
    "JUSTIFYING": 2000,
    "SOURCES":    2000,
    "CITATION":   2000
}

DOMAINS = [
    "Computer Science", "Sociology", "Molecular Biology", "History",
    "Macroeconomics", "Linguistics", "Quantum Physics", "Cognitive Psychology",
    "Philosophy of Mind", "Clinical Medicine", "Environmental Science"
]

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
    "CITATION": [
        "as noted in", "as argued by", "as shown by",
        "as demonstrated in", "as reported by",
        "as discussed in", "as outlined in", "as suggested by",
        "as observed by", "as concluded by"
    ]
}

print("Loading MLX model for synthetic data generation...")
model_id = "mlx-community/Qwen2.5-32B-Instruct-4bit"
model, tokenizer = mlx_lm.load(model_id)
print("Model loaded.\n")

def build_prompt(category: str, domain: str, trigger: str) -> str:
    # Strict negative constraints to prevent label noise for RoBERTa
    negative_constraints = (
        "CRITICAL RULE: Do NOT include parenthetical citations (e.g., Smith, 2021), "
        "modal verbs of probability (might, may, could), negations (not, never), or "
        "explicit authorial pronouns (I believe, we contend) UNLESS they are specifically "
        "part of the requested marker.\n"
    )

    return (
        f"You are an expert academic writer in the field of {domain}.\n"
        f"Write exactly 5 distinct academic sentences that contain a {category} "
        f"engagement marker. Each sentence MUST naturally use the trigger phrase "
        f"\"{trigger}\" or a close variant as the {category} marker.\n\n"
        f"{negative_constraints}\n"
        f"DIVERSITY RULES:\n"
        f"- Vary sentence length.\n"
        f"- Place the marker at different positions (beginning, middle, end).\n"
        f"- Each sentence must be on its own line in this exact format:\n"
        f"  <sentence> | <marker span>\n\n"
        f"Example for ENDOPHORIC with trigger 'in Table':\n"
        f"  The correlation coefficients shown in Table 2 confirm the hypothesis. | in Table 2\n\n"
        f"Output ONLY the 5 formatted lines. No numbering, no extra text."
    )

def parse_response_and_validate(response: str, category: str, seen: set) -> list:
    """Parse output, run regex sanity check, and format to pure JSON."""
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

        # Strip rogue numbering (e.g., "1. " or "- ")
        sentence = re.sub(r'^[\d\.\-\)\s]+', '', sentence).strip()

        if not sentence or not marker:
            continue
        
        # Deduplication
        key = sentence.lower().strip()
        if key in seen:
            continue
        
        # SANITY CHECK: Case-insensitive match and frequency check
        # This prevents boundary errors in the IOB converter
        match = re.search(re.escape(marker), sentence, re.IGNORECASE)
        matches_count = len(re.findall(re.escape(marker), sentence, re.IGNORECASE))
        
        if match and matches_count == 1:
            seen.add(key)
            # Use match.group(0) to grab the EXACT capitalization used in the sentence
            exact_span = match.group(0)
            
            results.append({
                "text": sentence,
                "label": category,
                "span": exact_span
            })
        else:
            # Silently discard noisy data (marker not found or found multiple times)
            continue

    return results

def generate_synthetic_data():
    synthetic_entries = []
    seen_sentences = set()

    for category, target_count in TARGET_CLASSES.items():
        print(f"Generating sentences for {category} (Target: {target_count})...")
        collected = []
        triggers = SEED_TRIGGERS[category]
        attempts = 0
        max_attempts = (target_count // 5) * 3 # Prevent infinite loops
        
        while len(collected) < target_count and attempts < max_attempts:
            domain = random.choice(DOMAINS)
            trigger = random.choice(triggers)
            prompt = build_prompt(category, domain, trigger)

            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            try:
                response = mlx_lm.generate(
                    model, tokenizer,
                    prompt=formatted,
                    max_tokens=1024,
                    verbose=False
                )

                parsed = parse_response_and_validate(response, category, seen_sentences)
                collected.extend(parsed)
                attempts += 1
                
                # Print progress every 5 successful attempts
                if attempts % 5 == 0:
                    print(f"  [{category}] {len(collected)}/{target_count} collected...")

            except Exception as e:
                print(f"Generation error: {e}")

        # Trim to exact target if we overshot
        collected = collected[:target_count]
        synthetic_entries.extend(collected)
        print(f"  -> Final count for {category}: {len(collected)}\n")

    # Save to pure JSONL format (ready for the IOB converter)
    output_path = 'data/synthetic_data_clean.jsonl'
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in synthetic_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print(f"\n[SUCCESS] Clean synthetic data saved to {output_path}")
    
    from collections import Counter
    counts = Counter(e["label"] for e in synthetic_entries)
    print("Final Breakdown:")
    for label, count in sorted(counts.items()):
        print(f"  {label:15s}: {count}")

if __name__ == "__main__":
    generate_synthetic_data()
