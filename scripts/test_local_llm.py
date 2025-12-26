from scripts.local_llm_client import call_local_llm

prompt = "Reply with exactly one word: OK"
print(call_local_llm(prompt))
