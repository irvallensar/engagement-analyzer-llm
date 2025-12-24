from script.openrouter_client import call_openrouter

def main():
    prompt = "Say hello in one sentence."
    response = call_openrouter(prompt)
    print("LLM response:")
    print(response)

if __name__ == "__main__":
    main()
