import torch

ckpt_path = "models/best_model.pth"
ckpt = torch.load(ckpt_path, map_location="cpu")

print("Type:", type(ckpt))
if isinstance(ckpt, dict):
    print("Top-level keys:", list(ckpt.keys())[:30])

    # common patterns
    state = None
    for k in ["state_dict", "model", "net", "model_state_dict"]:
        if k in ckpt and isinstance(ckpt[k], dict):
            state = ckpt[k]
            print(f"Using inner key: {k}")
            break

    if state is None:
        # maybe ckpt itself is the state_dict
        state = ckpt

    keys = list(state.keys())
    print("Num params:", len(keys))
    print("First 30 param keys:")
    for k in keys[:30]:
        print(" ", k)
else:
    print("Not a dict checkpoint.")
