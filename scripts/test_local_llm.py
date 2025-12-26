from scripts.local_llm_client import call_local_llm

prompt = (
    "Output exactly the following text, with no extra words:\n"
    "OK"
)
print(call_local_llm(prompt))
