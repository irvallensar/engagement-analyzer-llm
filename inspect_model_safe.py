# inspect_model_safe.py
from mlx_lm import load
MODEL_ID = "mlx-community/Qwen2.5-32B-Instruct-4bit"

model_obj, tokenizer = load(MODEL_ID)

# If load returned a tuple or wrapper, try to get the model
model = model_obj
for attr in ("model", "module", "base_model", "wrapped_model", "transformer"):
    if hasattr(model, attr):
        candidate = getattr(model, attr)
        if candidate is not None and not isinstance(candidate, str):
            model = candidate
            break

print("Loaded model type:", type(model))
matches = []
for name, module in model.named_modules():
    if any(k in name for k in ("q_proj", "k_proj", "v_proj", "gate_proj", "down_proj", "up_proj")):
        matches.append(name)

print("Matches found:", len(matches))
for n in matches[:80]:
    print(n)

# Safe parameter iteration
params = []
try:
    params = list(model.parameters())
except Exception:
    try:
        params = [p for _, p in model.named_parameters()]
    except Exception as e:
        print("Could not iterate parameters:", repr(e))
        raise SystemExit(1)

# Validate param objects
bad = [p for p in params if not hasattr(p, "numel")]
if bad:
    print("Found non-tensor parameters. Example types:")
    for i, p in enumerate(bad[:10]):
        print(i, type(p), repr(p)[:200])
    raise SystemExit(1)

total = sum(int(p.numel()) for p in params)
trainable = sum(int(p.numel()) for p in params if p.requires_grad)
print("Total params:", total)
print("Trainable params:", trainable, f"({100*trainable/total:.6f}%)")

if trainable > 0:
    print("\nFirst trainable parameter names (up to 40):")
    c = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(name)
            c += 1
            if c >= 40:
                break
else:
    print("\nNo trainable parameters detected. Likely LoRA adapters were not attached.")

