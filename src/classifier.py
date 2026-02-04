from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from src.config import (
    CLASS_NAMES,
    MODEL_MODE,
    MULTILABEL_THRESHOLD,
    IMG_SIZE,
    LABEL_TO_CONCERN_ID,
)


@dataclass
class Prediction:
    # used by app.py
    label_scores: Dict[str, float]
    concern_ids: List[int]

    # extra (doesn't hurt)
    labels: List[str]
    scores: List[float]
    positive_labels: List[str]
    topk_labels: List[str]
    topk_scores: List[float]


class SkinConcernClassifier:
    def __init__(
        self,
        ckpt_path: str,
        device: str = "cpu",
        class_names: Optional[List[str]] = None,
        img_size: int = IMG_SIZE,
    ):
        self.device = torch.device(device)
        self.class_names = class_names if class_names is not None else list(CLASS_NAMES)

        if not self.class_names:
            raise ValueError("CLASS_NAMES is empty. Check src/config.py")

        # Build model
        self.model = models.efficientnet_v2_s(weights=None)

        # Replace final layer to match your #labels
        if not isinstance(self.model.classifier, nn.Sequential) or not isinstance(self.model.classifier[-1], nn.Linear):
            raise RuntimeError("Unexpected EfficientNetV2 classifier structure.")
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, len(self.class_names))

        # Load checkpoint
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise RuntimeError(f"Checkpoint format not supported. Got: {type(state)}")

        cleaned = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            if nk.startswith("model."):
                nk = nk[len("model.") :]
            cleaned[nk] = v

        self.model.load_state_dict(cleaned, strict=True)

        self.model.to(self.device)
        self.model.eval()

        # Preprocess
        self.preprocess = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def _labels_to_concern_ids(labels: List[str]) -> List[int]:
        # Map labels -> concern IDs, unique + sorted for stable UI
        ids = []
        for lb in labels:
            cid = LABEL_TO_CONCERN_ID.get(lb)
            if cid is not None:
                ids.append(int(cid))
        return sorted(set(ids))

    @torch.inference_mode()
    def predict(self, image: Union[str, Image.Image], topk: int = 7) -> Prediction:
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB")

        x = self.preprocess(img).unsqueeze(0).to(self.device)
        logits = self.model(x).squeeze(0)

        mode = (MODEL_MODE or "").lower().strip()

        if mode == "multilabel":
            scores_t = torch.sigmoid(logits)
            threshold = float(MULTILABEL_THRESHOLD)
            positive_idx = (scores_t >= threshold).nonzero(as_tuple=False).flatten().tolist()
            positive_labels = [self.class_names[i] for i in positive_idx]
        else:
            # multiclass fallback
            scores_t = torch.softmax(logits, dim=0)
            positive_labels = []

        scores = scores_t.detach().cpu().tolist()
        label_scores = {label: float(score) for label, score in zip(self.class_names, scores)}

        # Top-k
        k = min(max(1, int(topk)), len(self.class_names))
        top_scores_t, top_idx_t = torch.topk(scores_t, k=k)
        top_scores = top_scores_t.detach().cpu().tolist()
        top_idx = top_idx_t.detach().cpu().tolist()
        top_labels = [self.class_names[i] for i in top_idx]

        # concern_ids: for multilabel use positives; for multiclass use top1
        if mode == "multilabel":
            concern_ids = self._labels_to_concern_ids(positive_labels)
        else:
            concern_ids = self._labels_to_concern_ids([top_labels[0]])

        return Prediction(
            label_scores=label_scores,
            concern_ids=concern_ids,
            labels=self.class_names,
            scores=[float(s) for s in scores],
            positive_labels=positive_labels,
            topk_labels=top_labels,
            topk_scores=[float(s) for s in top_scores],
        )

    def __call__(self, image: Union[str, Image.Image], topk: int = 7) -> Prediction:
        return self.predict(image, topk=topk)
