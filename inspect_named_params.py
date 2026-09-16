# inspect_named_params.py
from mlx_lm import load
import inspect
MODEL = "mlx-community/Qwen2.5-32B-Instruct-4bit"

model_obj, tokenizer = load(MODEL)
model = model_obj
for attr in ("model","module","base_model","wrapped_model","transformer"):
    if hasattr(model, attr):
        candidate = getattr(model, attr)
        if candidate is not None and not isinstance(candidate, str):
            model = candidate
            break

print("Model class:", type(model))
print("Has named_modules:", hasattr(model, "named_modules"))
print("Has named_parameters:", hasattr(model, "named_parameters"))

count = 0
for name, p in model.named_parameters():
    print("PARAM:", name, "type:", type(p), "is_tensor:", hasattr(p, "numel"))
    count += 1
    if count >= 200:
        break
print("Total named_parameters printed:", count)

# Also list module classes that look like quantized/linear layers
mods = {}
for name, m in model.named_modules():
    cls = type(m).__name__
    if "Linear" in cls or "Quant" in cls or "4bit" in cls.lower() or "qlinear" in cls.lower():
        mods.setdefault(cls, 0)
        mods[cls] += 1
print("Quant/Linear-like module classes found:", mods)

