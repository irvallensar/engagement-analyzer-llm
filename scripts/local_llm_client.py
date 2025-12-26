import subprocess

def call_local_llm(prompt):
    result = subprocess.run(
        ["ollama", "run", "phi3"],  # 👈 CHANGE HERE
        input=prompt,
        text=True,
        capture_output=True
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    return result.stdout.strip()
