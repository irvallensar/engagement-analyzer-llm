import mlx_lm

print("Loading model and adapters into Unified Memory...")
model_id = "mlx-community/Qwen2.5-32B-Instruct-4bit"
model, tokenizer = mlx_lm.load(model_id, adapter_path="adapters/adapters.safetensors")
print("Model loaded successfully! Ready for inference.")

def call_local_llm(sentence_text):

    with open("prompts/candidate_labeling.txt", "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()

    prompt_text = SYSTEM_PROMPT.replace("{sentence}", sentence_text)

    messages = [
        {
            "role": "system",
            "content": prompt_text
        }
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    response = mlx_lm.generate(
        model,
        tokenizer,
        prompt=formatted_prompt,
        max_tokens=1024,
        verbose=False
    )

    return response
