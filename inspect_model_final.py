# inspect_model_final.py
from mlx_lm import load
MODEL_ID = "mlx-community/Qwen2.5-32B-Instruct-4bit"

model_obj, tokenizer = load(MODEL_ID)

# unwrap common wrappers
model = model_obj
for attr in ("model", "module", "base_model", "wrapped_model", "transformer"):
    if hasattr(model, attr):
        candidate = getattr(model, attr)
        if candidate is not None and not isinstance(candidate, str):
            model = candidate
            break

print("Loaded model type:", type(model))

# list matching module names
matches = []
for name, module in model.named_modules():
    if any(k in name for k in ("q_proj", "k_proj", "v_proj", "gate_proj", "down_proj", "up_proj")):
        matches.append(name)
print("Matches found:", len(matches))
for n in matches[:80]:
    print(n)

# safe parameter iteration: only keep objects that look like tensors
params = []
try:
    for p in model.parameters():
        # skip non-tensor items
        if hasattr(p, "numel"):
            params.append(p)
except Exception:
    # fallback to named_parameters
    for _, p in model.named_parameters():
        if hasattr(p, "numel"):
            params.append(p)

if not params:
    print("No tensor parameters found on model.")
    raise SystemExit(1)

total = sum(int(p.numel()) for p in params)
trainable = sum(int(p.numel()) for p in params if p.requires_grad)
print("Total params:", total)
print("Trainable params:", trainable, f"({100*trainable/total:.6f}%)")

if trainable > 0:
    print("\nFirst trainable parameter names (up to 40):")
    c = 0
    for name, p in model.named_parameters():
        if hasattr(p, "numel") and p.requires_grad:
            print(name)
            c += 1
            if c >= 40:
                break
else:
    print("\nNo trainable parameters detected. Likely LoRA adapters were not attached.")

