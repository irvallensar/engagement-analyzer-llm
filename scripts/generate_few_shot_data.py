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
    "Philosophy of Mind", "Clinical Medicine", "Environmental Science"
]

# Unchanged seed triggers for strict A/B comparison
SEED_TRIGGERS = {
    "ENDOPHORIC": ["in Table", "as shown in", "above", "the following", "as described"],
    "SOURCES": ["researchers", "scholars", "previous studies"],
    "JUSTIFYING": ["because", "therefore", "thus", "since"],  
    "CITATION": ["as noted in", "as argued by", "as shown by", "as demonstrated in"]
}

# The new multi-example Few-Shot dictionary
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
    
    # Format the few-shot examples into a clean numbered list
    few_shot_context = "\n".join([f"{i+1}. {ex}" for i, ex in enumerate(GOLD_EXAMPLES[category])])
    
    # Few-Shot Prompt Architecture
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
    
    if len(matches) == 1:
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
        
        key = sentence.lower().strip()
        if key in seen: continue

        exact_span = get_robust_span_boundaries(sentence, marker)
        
        if exact_span: 
            seen.add(key)
            results.append({"text": sentence, "label": category, "span": exact_span})
            
    return results

def main():
    parser = argparse.ArgumentParser() 
    # CHANGED DEFAULT OUTPUT TO FEW-SHOT FILE
    parser.add_argument("--output", type=str, default="data/synthetic_few_shot.jsonl", help="Output JSONL file")
    args = parser.parse_args() 
    output_path = args.output 

    print("Loading MLX model for Few-Shot synthetic data generation...")
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

        print(f"Generating Few-Shot sentences for {category} (Target: {target_count}, Currently at: {collected_count})...")
        attempts = 0 
        consecutive_zeros = 0 
        max_attempts = target_count * 2 
        
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
                    if consecutive_zeros >= 50: 
                        print(f"  [WARNING] Stuck! 50 consecutive failed attempts. Breaking early.")
                        break
                    if attempts % 5 == 0: 
                        print(f"  [{category}] {collected_count}/{target_count} collected (Attempt {attempts})...")
                except Exception as e:
                    print(f"Generation error: {e}") 
        print(f"  -> Final count for {category}: {collected_count}\n")

    print(f"\n[SUCCESS] Clean Few-Shot synthetic data saved to {output_path}") 
    from collections import Counter
    counts = Counter(e["label"] for e in all_valid_entries) 
    
    print("Final Breakdown:")
    for label, count in sorted(counts.items()):
        print(f"  {label:15s}: {count}")

if __name__ == "__main__":
    main()
