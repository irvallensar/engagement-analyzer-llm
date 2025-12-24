# scripts/openrouter_client.py

import os
import requests

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY not set")

API_URL = "https://openrouter.ai/api/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    # REQUIRED by OpenRouter
    "HTTP-Referer": "https://github.com/irvallensar/engagement-analyzer-llm",
    "X-Title": "engagement-analyzer-llm",
}

def call_openrouter(prompt: str, model="openai/gpt-3.5-turbo"):
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload)
    response.raise_for_status()
    return response.json()
