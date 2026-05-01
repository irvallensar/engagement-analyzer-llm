import json
import time
# Import your existing local client (adjust the import based on your actual file)
from local_llm_client import generate_response 

# The target minority classes and their definitions for the prompt
TARGET_CLASSES = {
    "ENDOPHORIC": "A marker that refers to information in other parts of its own text (e.g., as mentioned above, in the next section, shown in Table 1).",
    "JUSTIFYING": "A marker that engages in persuasion through justification or substantiation (e.g., because, therefore, due to, the reason being).",
    "CITATION": "A segment referencing external sources explicitly (e.g., Smith (2000), (Jones, 2019)).",
    "SOURCES": "Nominalized expressions referencing sources without formal citation (e.g., previous studies, researchers, the literature)."
}

def create_prompt(label, definition):
    return [
        {
            "role": "system",
            "content": (
                "You are an expert computational linguist. Generate synthetic, highly realistic "
                "sentences for an academic discourse dataset. \n"
                "CRITICAL CONSTRAINTS:\n"
                "1. Lexical Diversity: Vary the academic disciplines (biology, sociology, etc.).\n"
                "2. Category Isolation: Include ONLY the requested marker. Do NOT include modals (might, may), "
                "negations (not), or authorial pronouns (I, we) unless they are part of the target.\n"
                "3. Output strictly as a JSON array containing a dictionary with 'text', 'label', and 'span'."
            )
        },
        {
            "role": "user",
            "content": (
                f"Target Category: {label}\n"
                f"Definition: {definition}\n\n"
                f"Generate 5 entirely new, diverse academic sentences for the {label} category. "
                "Output ONLY valid JSON. Example format:\n"
                "[\n  {\"text\": \"The results are significant because the p-value is low.\", \"label\": \"JUSTIFYING\", \"span\": \"because\"}\n]"
            )
        }
    ]

def main():
    output_file = "data/synthetic_raw.jsonl"
    
    with open(output_file, "a", encoding="utf-8") as f:
        for label, definition in TARGET_CLASSES.items():
            print(f"Generating synthetic data for {label}...")
            
            # Change this loop range to generate more (e.g., 100 loops * 5 sentences = 500 sentences)
            for i in range(10): 
                prompt = create_prompt(label, definition)
                
                try:
                    # Replace with however your local_llm_client handles calls
                    response_text = generate_response(prompt) 
                    
                    # Clean markdown if Qwen wraps it in ```json ... ```
                    if "```json" in response_text:
                        response_text = response_text.split("```json")[1].split("```")[0].strip()
                        
                    generated_items = json.loads(response_text)
                    
                    for item in generated_items:
                        text = item.get("text", "")
                        span = item.get("span", "")
                        
                        # SANITY CHECK: Quality Control to prevent label noise
                        if span in text and text.count(span) == 1:
                            f.write(json.dumps(item) + "\n")
                        else:
                            print(f"Discarded noisy data: {span} not found perfectly in text.")
                            
                except Exception as e:
                    print(f"Error during generation/parsing: {e}")
                
                time.sleep(1) # Be nice to your local GPU

if __name__ == "__main__":
    main()
