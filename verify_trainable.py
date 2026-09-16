# verify_trainable.py
from mlx_lm import load
MODEL = "local_qwen2.5_float"  # or "mlx-community/Qwen2.5-32B-Instruct" if you used Option A
model_obj, _ = load(MODEL)
model = model_obj
for attr in ("model","module","base_model","wrapped_model","transformer"):
    if hasattr(model, attr):
        candidate = getattr(model, attr)
        if candidate is not None and not isinstance(candidate, str):
            model = candidate
            break

total = sum(int(p.numel()) for p in model.parameters())
trainable = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
print("Total params:", total)
print("Trainable params:", trainable, f"({100*trainable/total:.6f}%)")
# print a few trainable names if any
c = 0
for name, p in model.named_parameters():
    if p.requires_grad:
        print("trainable:", name)
        c += 1
        if c >= 10:
            break

