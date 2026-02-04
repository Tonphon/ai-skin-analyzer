# src/config.py


CLASS_NAMES = [
    "acne", "oily", "dark_spots", "wrinkles", "sensitive", "eye_bags", "dullness"
]


LABEL_TO_CONCERN_ID = {
    "dullness": 1,      # Whitening
    "dark_spots": 1,    # Whitening
    "wrinkles": 2,      # Anti-Aging
    "acne": 3,          # Acne & Oily
    "oily": 3,          # Acne & Oily
    "eye_bags": 4,      # Eye Care
    "sensitive": 5      # Sensitive Skin
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
