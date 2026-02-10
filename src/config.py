# src/config.py
MODEL_ARCH = "efficientnet_v2_s"


CLASS_NAMES = [
    "Dark Circle",
    "PIH",
    "blackhead",
    "papule",
    "pustule",
    "whitehead",
    "wrinkle",
]


LABEL_TO_CONCERN_ID = {
    "Dark Circle": 1,   # Whitening
    "PIH": 1,           # Whitening
    "blackhead": 2,     # Anti-Aging
    "papule": 3,        # Acne & Oily
    "pustule": 3,       # Acne & Oily
    "whitehead": 4,     # Eye Care
    "wrinkle": 5        # Sensitive Skin
}


MODEL_MODE = "multilabel"
MULTILABEL_THRESHOLD = 0.40
IMG_SIZE = 224

# Concern ID -> display name
CONCERN_ID_TO_NAME = {
    1: "Whitening",
    2: "Anti-Aging",
    3: "Acne & Oily",
    4: "Eye Care",
    5: "Sensitive Skin",
}
