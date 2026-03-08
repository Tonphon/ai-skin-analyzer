# src/config.py

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
    "PIH": 1,           # Whitening
    "wrinkle": 2,       # Anti-Aging
    "blackhead": 3,     # Acne & Oily
    "whitehead": 3,     # Acne & Oily
    "Dark Circle": 4,   # Eye Care
    "papule": 5,        # Sensitive Skin
    "pustule": 5        # Sensitive Skin
}

CLASS_THRESHOLDS = {
    "Dark Circle": 0.50,
    "PIH": 0.50,
    "blackhead": 0.30,
    "papule": 0.35,
    "pustule": 0.40,
    "whitehead": 0.50,
    "wrinkle": 0.45,
}

MODEL_MODE = "multilabel"
IMG_SIZE = 224

# Concern ID -> display name
CONCERN_ID_TO_NAME = {
    1: "Whitening",
    2: "Anti-Aging",
    3: "Acne & Oily",
    4: "Eye Care",
    5: "Sensitive Skin",
}
