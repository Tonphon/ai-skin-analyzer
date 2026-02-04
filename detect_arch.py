import torch
import torch.nn as nn
from torchvision import models

CKPT_PATH = "models/best_model.pth"

state = torch.load(CKPT_PATH, map_location="cpu")

# infer num_classes from checkpoint head
if "classifier.1.weight" in state:
    num_classes = state["classifier.1.weight"].shape[0]
    head_key = "classifier.1.weight"
elif "classifier.0.weight" in state:
    num_classes = state["classifier.0.weight"].shape[0]
    head_key = "classifier.0.weight"
else:
    raise RuntimeError("Can't find classifier weight in checkpoint. Print tail keys and shapes to locate head.")

print("Inferred num_classes:", num_classes, "from", head_key)

def patch_head(m):
    # torchvision efficientnet + efficientnet_v2 both expose m.classifier as Sequential
    in_features = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_features, num_classes)
    return m

candidates = [
    ("efficientnet_v2_s", models.efficientnet_v2_s),
    ("efficientnet_v2_m", models.efficientnet_v2_m),
    ("efficientnet_v2_l", models.efficientnet_v2_l),
    ("efficientnet_b0", models.efficientnet_b0),
    ("efficientnet_b1", models.efficientnet_b1),
    ("efficientnet_b2", models.efficientnet_b2),
    ("efficientnet_b3", models.efficientnet_b3),
    ("efficientnet_b4", models.efficientnet_b4),
    ("efficientnet_b5", models.efficientnet_b5),
    ("efficientnet_b6", models.efficientnet_b6),
    ("efficientnet_b7", models.efficientnet_b7),
]

for name, fn in candidates:
    try:
        m = patch_head(fn(weights=None))
        m.load_state_dict(state, strict=True)
        print("✅ MATCH:", name)
        break
    except Exception as e:
        print("❌", name, "-", str(e).splitlines()[0])
