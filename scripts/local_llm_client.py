import subprocess

def call_local_llm(prompt: str) -> str:
    result = subprocess.run(
        [
            "ollama", "run", "mistral",
            "--temperature", "0",
            "--top-p", "1"
        ],
        input=prompt,
        text=True,
        capture_output=True
    )
    return result.stdout.strip()
