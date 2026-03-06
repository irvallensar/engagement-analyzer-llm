import urllib.request
import json

def call_local_llm(prompt: str) -> str:
    # --- UPDATE THIS URL TO THE ONE COLAB GIVES YOU ---
    url = "https://YOUR-RANDOM-WORDS.trycloudflare.com/api/generate"
    
    payload = {
        "model": "qwen2.5:14b",
        "prompt": prompt,
        "temperature": 0.0,
        "stream": False
    }
    
    data = json.dumps(payload).encode('utf-8')
    
    # Cloudflare sometimes blocks standard python headers, so we spoof a browser
    headers = {
        'Content-Type': 'application/json',
        'User-Agent': 'Mozilla/5.0'
    }
    
    req = urllib.request.Request(url, data=data, headers=headers)
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result.get("response", "").strip()
    except Exception as e:
        print(f"API ERROR: {e}")
        return ""
