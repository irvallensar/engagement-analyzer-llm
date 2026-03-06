import urllib.request
import json

def call_local_llm(prompt: str) -> str:
    # --- UPDATE THIS URL TO THE NEW LOCALTUNNEL ONE ---
    url = "https://great-donuts-argue.loca.lt/api/generate"
    
    payload = {
        "model": "qwen2.5:14b",
        "prompt": prompt,
        "temperature": 0.0,
        "stream": False
    }
    
    data = json.dumps(payload).encode('utf-8')
    
    # Localtunnel requires this exact header to bypass the warning screen!
    headers = {
        'Content-Type': 'application/json',
        'Bypass-Tunnel-Reminder': 'true',  # <--- THE MAGIC KEY
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
