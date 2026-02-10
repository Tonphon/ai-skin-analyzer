from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from src.config import (
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

    # extra (optional)
    labels: List[str]
    scores: List[float]
    positive_labels: List[str]
    topk_labels: List[str]
    topk_scores: List[float]


def _load_labels_json(path: str) -> List[str]:
    """
    Accepts models/labels.json in any of these shapes:
      - ["a", "b", "c"]
      - {"classes": ["a","b","c"]}
      - {"class_names": ["a","b","c"]}
      - {"labels": ["a","b","c"]}
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        labels = data
    elif isinstance(data, dict):
        for key in ("classes", "class_names", "labels"):
            if key in data and isinstance(data[key], list):
                labels = data[key]
                break
        else:
            raise ValueError(f"{path} must contain a list or a dict with classes/class_names/labels.")
    else:
        raise ValueError(f"{path} must be a list or dict, got: {type(data)}")

    labels = [str(x).strip() for x in labels if str(x).strip()]
    if not labels:
        raise ValueError(f"{path} is empty.")
    return labels


def _extract_state_dict(ckpt: object) -> Dict[str, torch.Tensor]:
    """
    Handles checkpoints saved as:
      - OrderedDict / plain state_dict
      - dict with keys like model_state_dict / state_dict / model / net
    """
    if isinstance(ckpt, dict):
        # common training save formats
        for key in ("model_state_dict", "state_dict", "model", "net", "model_state"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        # some people save {"...weights...": tensor} directly but also include metadata
        # If it looks like a state dict (many tensor values), keep only tensor entries.
        tensor_items = {k: v for k, v in ckpt.items() if torch.is_tensor(v)}
        if tensor_items:
            return tensor_items

    if isinstance(ckpt, dict):
        raise RuntimeError(f"Checkpoint dict format not recognized. Top-level keys: {list(ckpt.keys())[:20]}")
    raise RuntimeError(f"Checkpoint format not supported. Got: {type(ckpt)}")


def _clean_state_dict_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        nk = k
        for prefix in ("module.", "model."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        cleaned[nk] = v
    return cleaned


class SkinConcernClassifier:
    def __init__(
        self,
        ckpt_path: str,
        device: str = "cpu",
        labels_path: str = "models/labels.json",
        class_names: Optional[List[str]] = None,
        img_size: int = IMG_SIZE,
        arch: str = "efficientnet_v2_s",
    ):
        self.device = torch.device(device)

        # 1) Labels: prefer explicit class_names, else labels.json, else fallback to empty (error)
        if class_names is not None:
            self.class_names = [str(x).strip() for x in class_names if str(x).strip()]
        else:
            if labels_path and os.path.exists(labels_path):
                self.class_names = _load_labels_json(labels_path)
            else:
                raise ValueError(
                    f"labels_path not found: {labels_path}. "
                    f"Create it (models/labels.json) or pass class_names=[...]."
                )

        if not self.class_names:
            raise ValueError("No class names found.")

        # 2) Build model (keep your current choice; supports b0 too if you switch later)
        arch = (arch or "").lower().strip()
        if arch == "efficientnet_v2_s":
            self.model = models.efficientnet_v2_s(weights=None)
        elif arch == "efficientnet_b0":
            self.model = models.efficientnet_b0(weights=None)
        else:
            raise ValueError(f"Unsupported arch: {arch}. Use 'efficientnet_v2_s' or 'efficientnet_b0'.")

        # 3) Replace final layer to match #labels
        if not isinstance(self.model.classifier, nn.Sequential) or not isinstance(self.model.classifier[-1], nn.Linear):
            raise RuntimeError("Unexpected EfficientNet classifier structure.")
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, len(self.class_names))

        # 4) Load checkpoint robustly
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = _extract_state_dict(ckpt)
        state = _clean_state_dict_keys(state)

        # strict=True because we *want* to fail if arch/labels mismatch
        self.model.load_state_dict(state, strict=True)

        self.model.to(self.device)
        self.model.eval()

        # 5) Preprocess
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def _labels_to_concern_ids(labels: List[str]) -> List[int]:
        # Map labels -> concern IDs, unique + sorted for stable UI
        ids: List[int] = []
        for lb in labels:
            cid = LABEL_TO_CONCERN_ID.get(lb)
            if cid is not None:
                ids.append(int(cid))
        return sorted(set(ids))

    @torch.inference_mode()
    def predict(self, image: Union[str, Image.Image], topk: Optional[int] = None) -> Prediction:
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
            scores_t = torch.softmax(logits, dim=0)
            positive_labels = []

        scores = scores_t.detach().cpu().tolist()
        label_scores = {label: float(score) for label, score in zip(self.class_names, scores)}

        # Top-k (default = all labels)
        if topk is None:
            k = len(self.class_names)
        else:
            k = min(max(1, int(topk)), len(self.class_names))

        top_scores_t, top_idx_t = torch.topk(scores_t, k=k)
        top_scores = top_scores_t.detach().cpu().tolist()
        top_idx = top_idx_t.detach().cpu().tolist()
        top_labels = [self.class_names[i] for i in top_idx]

        # concern_ids: multilabel uses positives; multiclass uses top1
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

    def __call__(self, image: Union[str, Image.Image], topk: Optional[int] = None) -> Prediction:
        return self.predict(image, topk=topk)
