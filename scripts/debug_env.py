# scripts/debug_env.py
import os

key = os.getenv("OPENROUTER_API_KEY")
print("KEY FOUND:", key is not None)
print("KEY PREFIX:", key[:10] if key else None)
