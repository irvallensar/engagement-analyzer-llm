import urllib.request
import json

def call_local_llm(prompt: str) -> str:
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": "qwen3.5:9b",
        "prompt": prompt,
        "temperature": 0.0,
        "stream": False
    }
    
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result.get("response", "").strip()
    except Exception as e:
        print(f"API ERROR: {e}")
        return ""
