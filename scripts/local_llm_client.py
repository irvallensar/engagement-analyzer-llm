import requests
import json
import sys

# Configuration for Ollama HTTP API
OLLAMA_API_URL = "http://localhost:11434/api/generate"

def call_local_llm(prompt):
    """
    Sends the prompt to the local Ollama server and returns the text response.
    """
    payload = {
        "model": "qwen2.5:14b", 
        "prompt": prompt,
        "temperature": 0.0,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        
        result = response.json()
        return result.get('response', '[]')
        
    except requests.exceptions.RequestException as e:
        print(f"\n[!] API ERROR: {e}")
        return "[]"
