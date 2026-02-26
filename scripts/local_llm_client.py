import subprocess

def call_local_llm(prompt: str) -> str:
    process = subprocess.Popen(
        ["ollama", "run", "phi3"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    stdout, stderr = process.communicate(prompt)

    if stderr:
        print("OLLAMA STDERR:")
        print(stderr)

    return stdout.strip()
