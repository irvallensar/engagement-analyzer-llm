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

# V3 FIX: Complex, argumentative examples to prevent background class starvation.
# These force the model to mimic sentences containing COUNTER, DENY, and PROCLAIM vocabulary.
GOLD_EXAMPLES = {
    "ENDOPHORIC": [
        "While the information above , all discussed carbon in the soil , the Great Britain paper provided information for how much carbon was in the vegetation . | the information above",
        "As illustrated by the above studies , the exact effects of particular prosecutorial strategies and case dispositions upon repeat violence or arrest remains unclear at this time — although we might tentatively assert that prosecution appears to bring about some positive outcomes , particularly within the context of a coordinated community response . | above",
        "Finally , given that both approaches have essentially been developed within the wider Generative Grammar framework , it is likely that the minor differences of perspective and emphasis noted in this essay will , in time , be reconciled along the lines suggested by Grimshaw ( 1994 ) . | in this essay"
    ],
    "JUSTIFYING": [
        "Although a wealth of research has identified more violence exposure , particularly victimization , among boys , perhaps this is a fallacious conclusion due to the existence of unmeasured community violence in these studies that girls are more likely to experience than males . | due to the existence of unmeasured community violence in these studies that girls are more likely to experience than males",
        "Admittedly , this is not a conclusive finding as the sample size used was quite small and the ranking system was not completely objective or scientific . | as the sample size used",
        "However , intuitively it is insufficient to ensure the safety of safety - critical system , since safety - critical software must have a very low probability of failure - typically 10 -8 to 10 -9 . | since safety - critical software must have a very low probability of failure - typically 10 -8 to 10 -9"
    ],
    "SOURCES": [
        "Although the FDA denied that it \" [ bowed ] to political pressure in making this decision , \" conservative lobbyists and congress members applied political pressure before the decision was made , and post - decision , liberal congress members called for the resignation of Mr. Galson ( 2328 ) . | the FDA",
        "Although a wealth of research has identified more violence exposure , particularly victimization , among boys , perhaps this is a fallacious conclusion due to the existence of unmeasured community violence in these studies that girls are more likely to experience than males . | a wealth of research",
        "However , although this interpretation of Hegel 's claim is generally considered flawed , modern interpretations have had more success . | Hegel"
    ],
    "CITATION": [
        "Although the FDA denied that it \" [ bowed ] to political pressure in making this decision , \" conservative lobbyists and congress members applied political pressure before the decision was made , and post - decision , liberal congress members called for the resignation of Mr. Galson ( 2328 ) . | ( 2328 )",
        "Mass assault and arrests of women were not an uncommon sight any longer , which the particularly violent events of ' Black Friday ' in November 1910 highlighted ( Vicinus 1985 ) . | ( Vicinus 1985 )",
        "His theory of commoditization of IT , with electricity and railroad analogy , however , do have their limitations and constraints . ( Brown et al 2003 ) | ( Brown et al 2003 )"
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
        if len(sentence.split()) < 5: continue 
        
        key = sentence.lower().strip()
        if key in seen: continue

        exact_span = get_robust_span_boundaries(sentence, marker)
        
        if exact_span:
            seen.add(key)
            results.append({"text": sentence, "label": category, "span": exact_span})
            
    return results

def main():
    parser = argparse.ArgumentParser()
    # Outputting to v3 to keep a clean record
    parser.add_argument("--output", type=str, default="data/synthetic_few_shot_v3.jsonl", help="Output JSONL file")
    args = parser.parse_args()
    output_path = args.output

    print("Loading MLX model for Few-Shot V3 (Complex Examples) generation...")
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

        print(f"Generating V3 Few-Shot sentences for {category} (Target: {target_count}, Currently at: {collected_count})...")
        attempts = 0
        consecutive_zeros = 0
        max_attempts = 99999
        max_consecutive_zeros = 150  

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

    print(f"\n[SUCCESS] Clean V3 Few-Shot synthetic data saved to {output_path}")
    from collections import Counter
    counts = Counter(e["label"] for e in all_valid_entries)

    print("Final Breakdown:")
    for label, count in sorted(counts.items()):
        print(f"  {label:15s}: {count}")

if __name__ == "__main__":
    main()
