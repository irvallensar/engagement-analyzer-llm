import argparse
import mlx_lm
import json
import random
import re
import os

# How many sentences to generate for data augmentation
TARGET_CLASSES = {
    "ENDOPHORIC": 2000,
    "JUSTIFYING": 2000,
    "SOURCES":    2000,
    "CITATION":   2000
}

# A list of 11 academic domains. generator randomly picks one per prompt to esnure domain diversity in the syntethic data. 
DOMAINS = [
    "Computer Science", "Sociology", "Molecular Biology", "History",
    "Macroeconomics", "Linguistics", "Quantum Physics", "Cognitive Psychology",
    "Philosophy of Mind", "Clinical Medicine", "Environmental Science"
]

# Category-specific trigger phrases, associated with each engagement marker class. each prompt instructs the model to use one
# of these triggers, making the marker explicit and locatable in the generated sentence.
SEED_TRIGGERS = {
    "ENDOPHORIC": ["in Table", "as shown in", "above", "the following", "as described"],
    "SOURCES": ["researchers", "scholars", "previous studies"],
    "JUSTIFYING": ["because", "therefore", "thus", "since"],  
    "CITATION": ["as noted in", "as argued by", "as shown by", "as demonstrated in"]
}

#taken directly from engagement-annotation-project github (EDT)
GOLD_EXAMPLES = {
    "ENDOPHORIC": "As Table 3 shows, vocabulary size and depth were significantly correlated. | Table 3",
    "SOURCES": "Previous studies showed this effect consistently. | Previous studies",
    "JUSTIFYING": "These methods were selected because they offer greater precision. | because",
    "CITATION": " Some predictions made by Robertson et al (1999) make this even clearer. | Robertson et al (1999)"
}

# Takes a category, domain, trigger phrase, returns a fully formatted prompt string.
# Negative constraints that tells the model what to avoid.
def build_prompt(category: str, domain: str, trigger: str) -> str:
    negative_constraints = (
        "CRITICAL RULE: Do NOT include parenthetical citations (e.g., Smith, 2021), "
        "modal verbs of probability (might, may, could), negations (not, never), or "
        "explicit authorial pronouns (I believe, we contend) UNLESS they are specifically "
        "part of the requested marker.\n"
    )
    # A random integer injected into the prompt text. Adds noise to the prompt text itseld, 
    # nudging the model to produce.
    # different output across calls (to fight against repetitive generation).
    variation_seed = random.randint(10000, 99999)
    # Zero-shot Prompt: To adopt an academic persona for a specific domain; 
    # generate exactly 5 sentences, use the trigger phrase,
    # vary marker position and sentence length; and output in a strict sentence | span format. 
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
        f"EXAMPLE OF GOLD STANDARD FORMATTING:\n"
        f"{GOLD_EXAMPLES[category]}\n\n"
        f"NOW GENERATE 5 NEW SENTENCES:\n"
)

def get_robust_span_boundaries(text: str, span_text: str):
    """
    Finds the exact matched string of a span within a text,
    ignoring typographical whitespace errors commonly made by LLMs.
    Returns the exact string as it appears in the sentence, or None if invalid.
    """
    # 1. Escape any special regex characters in the target span (like parentheses)
    escaped_span = re.escape(span_text)
    
    # 2. Replace literal spaces with a flexible whitespace matcher (\s*)
    flexible_pattern = escaped_span.replace(r'\ ', r'\s*')
    
    # 3. Add flexibility around parentheses, as LLMs love adding spaces there
    flexible_pattern = flexible_pattern.replace(r'\(', r'\(\s*').replace(r'\)', r'\s*\)')
    
    # 4. Find all matches to ensure it only appears exactly once in the sentence
    matches = list(re.finditer(flexible_pattern, text, re.IGNORECASE))
    
    if len(matches) == 1:
        return matches[0].group(0) # Return the exact string as it appears in the text
        
    return None

# 1. Takes the raw text generated by the LLM
# 2. Parses each generated sentence
# 3. Validates whether the annotation is usable
# 4. Removes duplicates
# 5. Extracts the exact engagement marker span using robust matching
# 6. Returns only clean training examples
def parse_response_and_validate(response: str, category: str, seen: set) -> list:
    results = [] # creates list to store all valid parsed examples
    for line in response.split('\n'): # split the model response into individual lines. 
        line = line.strip() # Removes whitespace from the beginning and end
        if '|' not in line: continue # requires the required format, if not skip it
        parts = line.split('|', 1) # split the line into two parts, 1. sentences and 2. marker span
        if len(parts) < 2: continue # checks the format, if not the same, skip it.
        sentence = parts[0].strip() # extracts the the sentence and annotated marker. 
        marker = parts[1].strip()
        sentence = re.sub(r'^[\d\.\-\)\s]+', '', sentence).strip() # cleans numbering 
        # artifacts from LLM outputs
        if not sentence or not marker: continue # rejects empty data
        
        key = sentence.lower().strip() # normalize duplicate detection
        if key in seen: continue # skips duplicates to prevent repetitive dataset

        # Use the robust regex matcher to handle LLM spacing errors around punctuation
        exact_span = get_robust_span_boundaries(sentence, marker)
        
        # Only accept sentences if the marker exists and it appears exactly once
        if exact_span: 
            seen.add(key) # stores sentence in the deduplication set
            results.append({"text": sentence, "label": category, "span": exact_span})
            
    return results

def main():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--output", type=str, default="data/synthetic_data_clean.jsonl", help="Output JSONL file")
    args = parser.parse_args() 
    output_path = args.output 

    print("Loading MLX model for synthetic data generation...")
    model_id = "mlx-community/Qwen2.5-32B-Instruct-4bit"
    model, tokenizer = mlx_lm.load(model_id)
    print("Model loaded.\n")

    print(f"Starting generation. Saving directly to {output_path}...\n")
    seen_sentences = set() 
    all_valid_entries = [] 

    # --- RESUME LOGIC (PREVENTS DUPLICATES AND OVER-GENERATION) ---
    if os.path.exists(output_path):
        print(f"Found existing file at {output_path}. Loading previous progress...")
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line.strip())
                seen_sentences.add(data["text"].lower().strip())
                all_valid_entries.append(data)
        print(f"Loaded {len(seen_sentences)} sentences into deduplication memory.\n")
    # --------------------------------------------------------------

    for category, target_count in TARGET_CLASSES.items(): 
        # Calculate how many sentences we already generated for this specific category
        existing_count = sum(1 for e in all_valid_entries if e["label"] == category)
        collected_count = existing_count

        if collected_count >= target_count:
            print(f"[{category}] Already completed ({collected_count}/{target_count}). Skipping...")
            continue

        print(f"Generating sentences for {category} (Target: {target_count}, Currently at: {collected_count})...")
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

    print(f"\n[SUCCESS] Clean synthetic data saved to {output_path}") 
    from collections import Counter
    counts = Counter(e["label"] for e in all_valid_entries) 
    
    print("Final Breakdown:")
    for label, count in sorted(counts.items()):
        print(f"  {label:15s}: {count}")

if __name__ == "__main__":
    main()
