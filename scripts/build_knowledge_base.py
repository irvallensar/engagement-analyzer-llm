import os
import re
from pathlib import Path

# Paths (Adjust these if your folder structure is slightly different)
DOCS_DIR = Path("/Users/irvallensaragih/engagement-annotation-project/docs")
OUTPUT_FILE = Path("prompts/master_guidelines.txt")

def clean_markdown(text):
    # 1. Remove Markdown links but keep the visible text (e.g., [click here](url) -> click here)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # 2. Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # 3. Clean up excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def compile_guidelines():
    if not DOCS_DIR.exists():
        print(f"Error: Could not find {DOCS_DIR}. Make sure you cloned the repo.")
        return

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    compiled_text = []
    
    # Add a system primer at the very top
    compiled_text.append("### LINGUISTIC ANNOTATION GUIDELINES ###\n")
    compiled_text.append("The following text contains the official manual for annotating Engagement resources in academic discourse.\n")

    # Get all subfolders and sort them so 1_Clause comes before 2_...
    subfolders = sorted([f for f in DOCS_DIR.iterdir() if f.is_dir()])

    for folder in subfolders:
        # Get all markdown files in the folder and sort them
        md_files = sorted(folder.glob("*.md"))
        for md_file in md_files:
            print(f"Processing: {folder.name}/{md_file.name}")
            raw_text = md_file.read_text(encoding='utf-8')
            cleaned = clean_markdown(raw_text)
            
            # Append the cleaned text with a clear section break
            compiled_text.append(f"\n\n--- SECTION: {md_file.stem.replace('_', ' ').upper()} ---\n")
            compiled_text.append(cleaned)

    # Save the master file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("".join(compiled_text))
    
    print(f"\n[SUCCESS] Master guidelines compiled and saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    compile_guidelines()
