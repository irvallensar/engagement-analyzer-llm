import outlines
import mlx_lm
from pydantic import BaseModel
from typing import List

# 1. Define the Strict Pydantic Schema
class MarkerSpan(BaseModel):
    text: str
    label: str
    context_before: str

class ExtractionResult(BaseModel):
    thought_process: str
    spans: List[MarkerSpan]

# Globals — loaded once, reused across calls
mlx_model = None
mlx_tokenizer = None
outlines_model = None
generator = None


def call_local_llm(prompt_text):
    global mlx_model, mlx_tokenizer, outlines_model, generator

    # 2. Load the Model & Outlines Generator (Outlines 1.x API)
    if generator is None:
        print("\n[SYSTEM] Loading Llama-3.3-70B-Instruct-4bit into memory...")
        model_id = "mlx-community/Llama-3.3-70B-Instruct-4bit"

        # Outlines 1.x requires an already-loaded mlx_lm model + tokenizer,
        # NOT a model ID string passed directly to outlines.
        # outlines.models.mlxlm(model_id) no longer exists in 1.x.
        mlx_model, mlx_tokenizer = mlx_lm.load(model_id)

        # Wrap the loaded mlx_lm model for Outlines using the new from_mlxlm() function
        outlines_model = outlines.from_mlxlm(mlx_model, mlx_tokenizer)

        # Outlines 1.x: generate.json(model, Schema) is replaced by Generator(model, Schema)
        generator = outlines.Generator(outlines_model, ExtractionResult)

        print("[SYSTEM] Model loaded successfully.")

    # 3. Format the prompt for Qwen's ChatML
    messages = [
        {"role": "system", "content": "You are an expert computational linguist."},
        {"role": "user", "content": prompt_text}
    ]

    try:
        # Disable Qwen3's native thinking mode — the prompt already has its own
        # explicit chain-of-thought via the "thought_process" field, so stacking
        # native thinking on top adds unnecessary generation drift risk.
        formatted_prompt = mlx_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
    except (AttributeError, TypeError):
        # Fallback if tokenizer lacks apply_chat_template or enable_thinking kwarg
        try:
            formatted_prompt = mlx_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except AttributeError:
            formatted_prompt = (
                f"<|im_start|>system\nYou are an expert computational linguist.<|im_end|>\n"
                f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
            )

    # 4. Generate the response
    try:
        # Outlines 1.x Generator instances are called directly on the prompt string.
        raw_result = generator(formatted_prompt, max_tokens=4096)

        # Outlines 1.x Generator returns a JSON string by default, not a parsed
        # Pydantic object — model_validate_json is required here.
        if isinstance(raw_result, ExtractionResult):
            return raw_result
        elif isinstance(raw_result, str):
            return ExtractionResult.model_validate_json(raw_result)
        else:
            print(f"  [!] Unexpected type from outlines: {type(raw_result)}")
            return ExtractionResult(thought_process="Error", spans=[])

    except Exception as e:
        print(f"  [!] LLM Crash or Parsing Error: {e}")
        # Returning an empty ExtractionResult prevents evaluate_iob.py from crashing
        return ExtractionResult(thought_process="Error", spans=[])
