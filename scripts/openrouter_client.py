# scripts/openrouter_client.py

import os
import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def call_openrouter(prompt: str, model: str = "openai/gpt-3.5-turbo") -> str:
    """
    Sends a prompt to OpenRouter and returns the model's text response.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not found in environment")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(OPENROUTER_URL, headers=headers, json=payload)
    response.raise_for_status()

    data = response.json()
    return data["choices"][0]["message"]["content"]
