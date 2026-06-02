import argparse
import mlx_lm
import json
import random
import re
import os
from collections import Counter

TARGET_CLASSES = {
    "ENDOPHORIC": 2000,
    "JUSTIFYING": 2000,
    "SOURCES":    2000,
    "CITATION":   2000
}

DOMAINS = [
    "Computer Science", "Sociology", "Molecular Biology", "History",
    "Macroeconomics", "Linguistics", "Quantum Physics", "Cognitive Psychology",
    "Philosophy of Mind", "Clinical Medicine", "Environmental Science",
    "Political Science", "Anthropology", "Education Research", "Public Health",
    "Literary Studies", "Urban Planning", "Neuroscience", "International Relations"
]

SEED_TRIGGERS = {
    "ENDOPHORIC": [
        "in Table", "as shown in", "above", "the following", "as described",
        "in Figure", "as noted earlier", "on page", "in Section", "as outlined",
        "as discussed", "as illustrated in", "as presented in", "see Table",
        "as mentioned above", "in the previous section"
    ],
    "SOURCES": [
        "researchers", "scholars", "previous studies", "recent studies",
        "experts", "the literature", "existing research", "empirical evidence",
        "studies have shown", "the findings", "evidence suggests",
        "analysts", "practitioners", "survey data", "field studies",
        "academic consensus", "prior work"
    ],
    "JUSTIFYING": [
        "because", "therefore", "thus", "since", "given that",
        "due to", "as a result", "consequently", "for this reason",
        "in order to", "so that", "owing to", "hence", "on account of",
        "this is why", "which explains why"
    ],
    "CITATION": [
        "as noted in", "as argued by", "as shown by", "as demonstrated in",
        "according to", "as stated by", "as reported by", "as found by",
        "as suggested by", "as observed by", "as discussed by",
        "as proposed by", "as outlined by", "as established by"
    ]
}

GOLD_EXAMPLES = {
    "ENDOPHORIC": [
        "As Table 3 shows, vocabulary size and depth were significantly correlated. | Table 3",
        "Nursing and agriculture journals, as noted earlier in this paper, often have required sections. | earlier in this paper",
        "Response to Joan Kelly Hall's article by Paul Seedhouse follows on page 527 of this paper. | on page 527 of this paper"
    ],
    "SOURCES": [
        "Previous studies showed this effect consistently. | Previous studies",
        "The annual report by Google mentioned the possibility of further expansion. | The annual report by Google",
        "Amazon mentioned the possibility of investing in different sectors. | Amazon"
    ],
    "JUSTIFYING": [
        "These methods were selected because they offer greater precision. | because",
        "Given that about a quarter of all employed people move on and off the payrolls of individual firms during the year, a need to move between firms to climb the career ladder would not seem to be a difficult barrier to surmount. | Given that about a quarter of all employed people move on and off the payrolls of individual firms during the year",
        "The lack of connection between the SLA theories, which tend to be English or European languages based, and the Japanese teaching that is found in JLT is probably due to a lack of attention to pragmatic and sociocultural aspects of Japanese language and communication. | due to a lack of attention to pragmatic and sociocultural aspects of Japanese language and communication"
    ],
    "CITATION": [
        "Some predictions made by Robertson et al (1999) make this even clearer. | Robertson et al (1999)",
        "Ideas of nationalism often associated with both the ideology and actions of racism (Fenton, 1999). | (Fenton, 1999)",
        "Relativity's theoretical foundations can be traced to earlier work by Faraday and Maxwell (Einstein 782). | (Einstein 782)"
    ]
}

def build_prompt(category: str, domain: str, trigger: str) -> str:
    negative_constraints = (
        "CRITICAL RULE: Do NOT include parenthetical citations (e.g., Smith, 2021), "
        "modal verbs of probability (might, may, could), negations (not, never), or "
        "explicit authorial pronouns (I believe, we contend) UNLESS they are specifically "
        "part of the requested marker.\n"
    )
    variation_seed = random.randint(10000, 99999)
    few_shot_context = "\n".join([f"{i+1}. {ex}" for i, ex in enumerate(GOLD_EXAMPLES[category])])

    return (
        f"You are an expert academic writer in the field of {domain}.\n"
        f"Write exactly 5 distinct academic sentences where the phrase \"{trigger}\" "
        f"functions specifically as a {category} engagement marker.\n\n"
        f"IMPORTANT: The phrase \"{trigger}\" must function as {category} in context.\n\n"
        f"Variation Seed: {variation_seed}\n\n"
        f"{negative_constraints}\n"
        f"STRICT FORMAT RULE: Every line MUST follow this exact format with the marker "
        f"copied EXACTLY as it appears in the sentence:\n"
        f"  <full sentence> | <exact marker text copied from sentence>\n\n"
        f"FEW-SHOT EXAMPLES:\n"
        f"{few_shot_context}\n\n"
        f"NOW GENERATE 5 NEW SENTENCES:\n"
    )

def get_robust_span_boundaries(text: str, span_text: str):
    """
    FIX: Accept first match even for multi-match spans.
    Also try progressively relaxed matching if exact match fails.
    """
    # Attempt 1: flexible regex match
    escaped_span = re.escape(span_text)
    flexible_pattern = escaped_span.replace(r'\ ', r'\s+')
    flexible_pattern = flexible_pattern.replace(r'\(', r'\(\s*').replace(r'\)', r'\s*\)')
    matches = list(re.finditer(flexible_pattern, text, re.IGNORECASE))
    if matches:
        return matches[0].group(0)

    # Attempt 2: try matching just the first few words of the span
    # Handles cases where model writes slightly different span than sentence
    span_words = span_text.split()
    if len(span_words) >= 2:
        short_pattern = r'\s+'.join(re.escape(w) for w in span_words[:3])
        short_matches = list(re.finditer(short_pattern, text, re.IGNORECASE))
        if short_matches:
            # Return the full span starting from this match position
            start = short_matches[0].start()
            end = min(start + len(span_text) + 10, len(text))
            return text[start:end].strip()

    return None

def parse_response_and_validate(response: str, category: str, seen: set) -> tuple:
    """Returns (results, rejection_counts) for diagnostics."""
    results = []
    rejections = Counter()

    for line in response.split('\n'):
        line = line.strip()

        # FIX: Also handle numbered lines like "1. sentence | span"
        line = re.sub(r'^[\d]+[\.\)]\s*', '', line).strip()

        if '|' not in line:
            rejections['no_pipe'] += 1
            continue
        parts = line.split('|', 1)
        if len(parts) < 2:
            rejections['no_pipe'] += 1
            continue

        sentence = parts[0].strip()
        marker = parts[1].strip()

        if not sentence or not marker:
            rejections['empty'] += 1
            continue
        if len(sentence.split()) < 5:
            rejections['too_short'] += 1
            continue

        key = sentence.lower().strip()
        if key in seen:
            rejections['duplicate'] += 1
            continue

        exact_span = get_robust_span_boundaries(sentence, marker)
        if not exact_span:
            rejections['span_not_found'] += 1
            continue

        seen.add(key)
        results.append({"text": sentence, "label": category, "span": exact_span})

    return results, rejections

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/synthetic_few_shot.jsonl")
    args = parser.parse_args()
    output_path = args.output

    print("Loading MLX model...")
    model_id = "mlx-community/Qwen2.5-32B-Instruct-4bit"
    model, tokenizer = mlx_lm.load(model_id)
    print("Model loaded.\n")

    seen_sentences = set()
    all_valid_entries = []

    if os.path.exists(output_path):
        print(f"Resuming from {output_path}...")
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line.strip())
                seen_sentences.add(data["text"].lower().strip())
                all_valid_entries.append(data)
        print(f"Loaded {len(seen_sentences)} existing sentences.\n")

    for category, target_count in TARGET_CLASSES.items():
        existing_count = sum(1 for e in all_valid_entries if e["label"] == category)
        collected_count = existing_count

        if collected_count >= target_count:
            print(f"[{category}] Already completed ({collected_count}/{target_count}). Skipping...")
            continue

        print(f"\nGenerating {category} (Target: {target_count}, Currently: {collected_count})...")
        attempts = 0
        consecutive_zeros = 0
        total_rejections = Counter()

        with open(output_path, 'a', encoding='utf-8') as f:
            while collected_count < target_count and attempts < 99999:
                domain = random.choice(DOMAINS)
                trigger = random.choice(SEED_TRIGGERS[category])
                prompt = build_prompt(category, domain, trigger)
                messages = [{"role": "user", "content": prompt}]
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                try:
                    response = mlx_lm.generate(
                        model, tokenizer, prompt=formatted,
                        max_tokens=1024, verbose=False
                    )
                    parsed, rejections = parse_response_and_validate(
                        response, category, seen_sentences
                    )
                    total_rejections.update(rejections)

                    if len(parsed) == 0:
                        consecutive_zeros += 1
                    else:
                        consecutive_zeros = 0

                    for entry in parsed:
                        if collected_count < target_count:
                            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                            f.flush()
                            collected_count += 1
                            all_valid_entries.append(entry)

                    attempts += 1

                    # FIX: Raise consecutive zero threshold significantly
                    if consecutive_zeros >= 150:
                        print(f"  [WARNING] 150 consecutive empty batches. Breaking.")
                        break

                    # Print progress + rejection breakdown every 20 attempts
                    if attempts % 20 == 0:
                        print(f"  [{category}] {collected_count}/{target_count} "
                              f"(Attempt {attempts}) | Rejections: {dict(total_rejections)}")
                        total_rejections.clear()  # reset per window

                except Exception as e:
                    print(f"Generation error: {e}")

        print(f"  -> Final count for {category}: {collected_count}")

    print(f"\n[SUCCESS] Saved to {output_path}")
    counts = Counter(e["label"] for e in all_valid_entries)
    print("Final Breakdown:")
    for label, count in sorted(counts.items()):
        print(f"  {label:15s}: {count}")

if __name__ == "__main__":
    main()
