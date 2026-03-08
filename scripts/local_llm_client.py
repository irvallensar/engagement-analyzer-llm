import requests
import json
import sys

# Configuration for Ollama HTTP API
OLLAMA_API_URL = "http://localhost:11434/api/generate"

def get_completion(prompt):
    """
    Sends the prompt to the local Ollama server and returns the text response.
    """
    payload = {
        # CRITICAL: This must match the model we pull in Colab
        "model": "qwen2.5:14b-instruct-q6_K", 
        "prompt": prompt,
        "temperature": 0.0,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        return result['response']
        
    except requests.exceptions.RequestException as e:
        print(f"\n[!] API ERROR: {e}")
        return "[]"
