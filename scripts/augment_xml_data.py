import mlx_lm
import json
import random

# The classes that need life support
TARGET_CLASSES = ["ENDOPHORIC", "JUSTIFYING", "SOURCES", "PROCLAIM", "CITATION"]
SENTENCES_PER_CLASS = 100 # Adjust this if you want more/less

print("Loading MLX model for synthetic data generation...")
model_id = "mlx-community/Qwen2.5-32B-Instruct-4bit"
model, tokenizer = mlx_lm.load(model_id)

def generate_synthetic_data():
    synthetic_dataset = []

    for category in TARGET_CLASSES:
        print(f"\nGenerating {SENTENCES_PER_CLASS} sentences for {category}...")
        
        prompt = (
            f"You are an expert academic writer. Write 10 distinct, highly academic sentences "
            f"that contain a {category} engagement marker. "
            f"You MUST wrap the exact marker phrase in <{category}></{category}> XML tags. "
            f"Do not write any introductory or concluding text. Just output the 10 sentences, one per line.\n"
            f"Example for ENDOPHORIC: The results shown <ENDOPHORIC>in the table</ENDOPHORIC> suggest a correlation."
        )

        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # We run this in batches of 10 to hit our target
        for _ in range(SENTENCES_PER_CLASS // 10):
            response = mlx_lm.generate(model, tokenizer, prompt=formatted_prompt, max_tokens=1024, verbose=False)
            
            # Split the response into individual sentences
            sentences = [s.strip() for s in response.split('\n') if s.strip() and f"<{category}>" in s]
            
            for sentence in sentences:
                # Strip any numbering the LLM might have added (e.g., "1. The results...")
                if sentence[0].isdigit() and sentence[1:3] in ['. ', ') ']:
                    sentence = sentence[3:]
                
                # Create the raw version by stripping out the XML tags
                raw_text = sentence.replace(f"<{category}>", "").replace(f"</{category}>", "")
                
                synthetic_dataset.append({
                    "raw_text": raw_text,
                    "tagged_text": sentence
                })
                
    # Save the synthetic sentences
    with open('data/synthetic_xml.jsonl', 'w', encoding='utf-8') as f:
        for entry in synthetic_dataset:
            f.write(json.dumps(entry) + '\n')
            
    print(f"\n[SUCCESS] Generated {len(synthetic_dataset)} synthetic XML sentences!")

if __name__ == "__main__":
    generate_synthetic_data()
