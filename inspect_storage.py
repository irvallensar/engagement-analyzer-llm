# inspect_storage.py
from mlx_lm import load
MODEL = "mlx-community/Qwen2.5-32B-Instruct-4bit"

m, _ = load(MODEL)

# try to unwrap common wrappers
for attr in ("model","module","base_model","wrapped_model","transformer"):
    if hasattr(m, attr):
        candidate = getattr(m, attr)
        if candidate is not None and not isinstance(candidate, str):
            m = candidate
            break

print("Model class:", type(m))
print("Has named_modules:", hasattr(m, "named_modules"))
print("Has named_parameters:", hasattr(m, "named_parameters"))
print("Has state_dict:", hasattr(m, "state_dict"))

# Print top-level attributes that are not callables
print("\nTop-level non-callable attributes (first 80):")
attrs = [a for a in dir(m) if not callable(getattr(m, a)) and not a.startswith("__")]
for a in attrs[:80]:
    print(a)

# If state_dict exists, print first 60 keys
if hasattr(m, "state_dict"):
    try:
        sd = m.state_dict()
        print("\nstate_dict keys (first 120):")
        for i, k in enumerate(list(sd.keys())[:120]):
            print(k)
    except Exception as e:
        print("\nstate_dict call failed:", repr(e))


