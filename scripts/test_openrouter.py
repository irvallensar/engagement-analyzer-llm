from scripts.openrouter_client import call_openrouter

def main():
    prompt = "Reply with exactly one word: OK"
    response = call_openrouter(
        prompt,
        model="meta-llama/llama-3.2-3b-instruct:free"
    )
    print(response["choices"][0]["message"]["content"])

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
