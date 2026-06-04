import argparse
import mlx_lm
import json
import random
import re
import os

# Target generation counts
TARGET_CLASSES = {
    "ENDOPHORIC": 2000,
    "JUSTIFYING": 2000,
    "SOURCES":    2000,
    "CITATION":   2000
}

# Academic domains for topical diversity
DOMAINS = [
    "Computer Science", "Sociology", "Molecular Biology", "History",
    "Macroeconomics", "Linguistics", "Quantum Physics", "Cognitive Psychology",
    "Philosophy of Mind", "Clinical Medicine", "Environmental Science",
    "Political Science", "Anthropology", "Education Research", "Public Health",
    "Literary Studies", "Urban Planning", "Neuroscience", "International Relations"
]

# Expanded SEED_TRIGGERS — more variety reduces deduplication rejections
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
        "the report by", "according to the", "studies have shown",
        "the findings of", "evidence suggests", "the data from",
        "the survey by", "analysts", "practitioners in the field"
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

# V3 FIX: Hand-Curated, Leak-Proof Complex Human Examples
GOLD_EXAMPLES = {
    "ENDOPHORIC": [
        "The final method, however, as it is used commonly as well as in the present study, was rendered by Beatty and Gibbons in 1937 (Farber, 1965). | in the present study",
        "As shown in Figure 1, it can be reasonably concluded that sacrifices are made at frequency levels where the human ear is not as sensitive. | in Figure 1",
        "Finally, given that both approaches have essentially been developed within the wider Generative Grammar framework, it is likely that the minor differences of perspective and emphasis noted in this essay will, in time, be reconciled along the lines suggested by Grimshaw (1994). | in this essay"
    ],
    "SOURCES": [
        "Considerable research indicates that people expect positive behavior from others when they are not given reasons to expect negative behavior. | Considerable research",
        "While historically imported nursery stock has been the most common source of nonindigenous forest insects, most researchers agree that infested solid wood packing materials was the source for this invasion (Poland and Mcullough, et al., 2006). | most researchers",
        "Many scholars, the most important of which may be Judith Butler (1990) and Joan Scott (1988) have argued that the differences among those traditionally identified as women are discursively produced and unfixed. | Many scholars"
    ],
    "JUSTIFYING": [
        "However, intuitively it is insufficient to ensure the safety of safety-critical systems, since safety-critical software must have a very low probability of failure — typically 10⁻⁸ to 10⁻⁹. | since safety-critical software must have a very low probability of failure — typically 10⁻⁸ to 10⁻⁹",
        "Therefore it is imperative that students, wherever possible and physically able-bodied, be able to contribute to the household income by working part time. | Therefore",
        "Poor management practices that cause stress also contribute to high rates of infection as they lower a calf's ability to fight off the infection. | as they"
    ],
    "CITATION": [
        "However Reichheld and Schefter (2000) found that less than 20% of the online companies take advantage of this opportunity thus neglecting chances of up selling. | Reichheld and Schefter (2000)",
        "The existing AWV is planned to be demolished in 2012 despite the lack of a consensus by federal, state, and local government officials of how to reroute the traffic (McGann 2008). | (McGann 2008)",
        "For example, Bandura (1965) had a group of 3-5 year old children watch a video of an adult acting aggressively with a Bobo doll and another group who did not see this demonstration. | Bandura (1965)"
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
        f"IMPORTANT: The phrase \"{trigger}\" must function as {category} in context, "
        f"NOT as any other engagement type. For example:\n"
        f"- If category is JUSTIFYING, '{trigger}' must establish a causal justification, "
        f"not a counter-expectation.\n"
        f"- If category is ENDOPHORIC, '{trigger}' must refer to content within the same "
        f"text, not to an external source.\n"
        f"- If category is SOURCES, '{trigger}' must reference a vague group of researchers "
        f"or scholars, not a specific named author.\n"
        f"- If category is CITATION, '{trigger}' must reference a specific named external "
        f"source, not a vague group.\n\n"
        f"Variation Seed: {variation_seed}\n\n"
        f"{negative_constraints}\n"
        f"DIVERSITY RULES:\n"
        f"- Vary sentence length.\n"
        f"- Place the marker at different positions (beginning, middle, end).\n"
        f"- Each sentence must be on its own line in this exact format:\n"
        f"  <sentence> | <marker span>\n\n"
        f"FEW-SHOT EXAMPLES (MIMIC THIS EXACT TONE, COMPLEXITY, AND FORMAT):\n"
        f"{few_shot_context}\n\n"
        f"NOW GENERATE 5 NEW SENTENCES:\n"
    )

def get_robust_span_boundaries(text: str, span_text: str):
    escaped_span = re.escape(span_text)
    flexible_pattern = escaped_span.replace(r'\ ', r'\s*')
    flexible_pattern = flexible_pattern.replace(r'\(', r'\(\s*').replace(r'\)', r'\s*\)')
    matches = list(re.finditer(flexible_pattern, text, re.IGNORECASE))
    
    if len(matches) >= 1:
        return matches[0].group(0)
        
    return None

def parse_response_and_validate(response: str, category: str, seen: set) -> list:
    results = []
    for line in response.split('\n'):
        line = line.strip()
        if '|' not in line: continue
        parts = line.split('|', 1)
        if len(parts) < 2: continue
        
        sentence = parts[0].strip()
        marker = parts[1].strip()
        sentence = re.sub(r'^[\d\.\-\)\s]+', '', sentence).strip()
        
        if not sentence or not marker: continue
        if len(sentence.split()) < 5: continue  # skip suspiciously short sentences
        
        key = sentence.lower().strip()
        if key in seen: continue

        exact_span = get_robust_span_boundaries(sentence, marker)
        
        if exact_span:
            seen.add(key)
            results.append({"text": sentence, "label": category, "span": exact_span})
            
    return results

def main():
    parser = argparse.ArgumentParser()
    # Outputting directly to v3
    parser.add_argument("--output", type=str, default="data/synthetic_few_shot_v3.jsonl", help="Output JSONL file")
    args = parser.parse_args()
    output_path = args.output

    print("Loading MLX model for Few-Shot V3 synthetic data generation...")
    model_id = "mlx-community/Qwen2.5-32B-Instruct-4bit"
    model, tokenizer = mlx_lm.load(model_id)
    print("Model loaded.\n")

    print(f"Starting generation. Saving directly to {output_path}...\n")
    seen_sentences = set()
    all_valid_entries = []

    if os.path.exists(output_path):
        print(f"Found existing file at {output_path}. Loading previous progress...")
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line.strip())
                seen_sentences.add(data["text"].lower().strip())
                all_valid_entries.append(data)
        print(f"Loaded {len(seen_sentences)} sentences into deduplication memory.\n")

    for category, target_count in TARGET_CLASSES.items():
        existing_count = sum(1 for e in all_valid_entries if e["label"] == category)
        collected_count = existing_count

        if collected_count >= target_count:
            print(f"[{category}] Already completed ({collected_count}/{target_count}). Skipping...")
            continue

        print(f"Generating Few-Shot V3 sentences for {category} (Target: {target_count}, Currently at: {collected_count})...")
        attempts = 0
        consecutive_zeros = 0

        # Run until target is hit or truly stuck
        max_attempts = 99999
        max_consecutive_zeros = 150  # Elevated threshold for complex examples

        with open(output_path, 'a', encoding='utf-8') as f:
            while collected_count < target_count and attempts < max_attempts:
                domain = random.choice(DOMAINS)
                trigger = random.choice(SEED_TRIGGERS[category])
                prompt = build_prompt(category, domain, trigger)
                messages = [{"role": "user", "content": prompt}]
                formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                try:
                    response = mlx_lm.generate(model, tokenizer, prompt=formatted, max_tokens=1024, verbose=False)
                    parsed = parse_response_and_validate(response, category, seen_sentences)

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
                    if consecutive_zeros >= max_consecutive_zeros:
                        print(f"  [WARNING] Stuck! {max_consecutive_zeros} consecutive failed attempts. Breaking early.")
                        break
                    if attempts % 10 == 0:
                        print(f"  [{category}] {collected_count}/{target_count} collected (Attempt {attempts})...")
                except Exception as e:
                    print(f"Generation error: {e}")

        print(f"  -> Final count for {category}: {collected_count}\n")

    print(f"\n[SUCCESS] Clean Few-Shot V3 synthetic data saved to {output_path}")
    from collections import Counter
    counts = Counter(e["label"] for e in all_valid_entries)

    print("Final Breakdown:")
    for label, count in sorted(counts.items()):
        print(f"  {label:15s}: {count}")

if __name__ == "__main__":
    main()
