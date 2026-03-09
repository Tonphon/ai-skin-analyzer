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
    CLASS_THRESHOLDS,
    IMG_SIZE,
    LABEL_TO_CONCERN_ID,
)


@dataclass
class Prediction:
    label_scores: Dict[str, float]
    concern_ids: List[int]
    labels: List[str]
    scores: List[float]
    positive_labels: List[str]
    topk_labels: List[str]
    topk_scores: List[float]


def _load_labels_json(path: str) -> List[str]:
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
    Supports:
      - raw state_dict (OrderedDict / dict of tensors)
      - checkpoint dict containing model_state_dict / state_dict / model
    """
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model", "net", "model_state"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]

        tensor_items = {k: v for k, v in ckpt.items() if torch.is_tensor(v)}
        if tensor_items:
            return tensor_items

        raise RuntimeError(
            "Checkpoint dict format not recognized. "
            f"Top-level keys: {list(ckpt.keys())[:30]}"
        )

    raise RuntimeError(f"Checkpoint format not supported. Got: {type(ckpt)}")


def _clean_state_dict_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Strips common wrappers/prefixes so torchvision EfficientNet can load:
      - module. (DDP)
      - model.  (some training scripts)
      - efficientnet. (some wrappers save under self.efficientnet)
    """
    cleaned: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        nk = k
        for prefix in ("module.", "model."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]

        if nk.startswith("efficientnet."):
            nk = nk[len("efficientnet."):]

        cleaned[nk] = v
    return cleaned


def _infer_num_classes_from_state(state: Dict[str, torch.Tensor]) -> int:
    """
    For torchvision EfficientNet_v2_s: model.classifier is Sequential and last layer is Linear.
    Common key: classifier.1.weight (or classifier.0.weight depending on how you replaced head).
    """
    for key in ("classifier.1.weight", "classifier.0.weight"):
        w = state.get(key)
        if isinstance(w, torch.Tensor) and w.ndim == 2:
            return int(w.shape[0])
    raise RuntimeError(
        "Can't infer num_classes from checkpoint. "
        "Expected classifier.1.weight or classifier.0.weight in state_dict."
    )


class SkinConcernClassifier:
    def __init__(
        self,
        ckpt_path: str,
        device: str = "cpu",
        labels_path: str = "models/labels.json",
        class_names: Optional[List[str]] = None,
        img_size: int = IMG_SIZE,
        arch: str = "efficientnet_v2_s",  # final
    ):
        self.device = torch.device(device)

        # 1) Labels
        if class_names is not None:
            self.class_names = [str(x).strip() for x in class_names if str(x).strip()]
        else:
            if labels_path and os.path.exists(labels_path):
                self.class_names = _load_labels_json(labels_path)
            else:
                raise ValueError(
                    f"labels_path not found: {labels_path}. "
                    "Create models/labels.json or pass class_names=[...]."
                )

        if not self.class_names:
            raise ValueError("No class names found.")

        # 2) Load checkpoint (handle PyTorch 2.6+ weights_only change)
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
        except Exception:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        state = _extract_state_dict(ckpt)
        state = _clean_state_dict_keys(state)

        # 3) Build model (EfficientNet only)
        arch = (arch or "").lower().strip()
        if arch != "efficientnet_v2_s":
            raise ValueError(f"This classifier.py is finalized for efficientnet_v2_s only. Got: {arch}")

        self.model = models.efficientnet_v2_s(weights=None)

        # Replace head to match checkpoint/classes
        num_classes_ckpt = _infer_num_classes_from_state(state)
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes_ckpt)

        # Safety check: checkpoint classes should match labels.json count
        if num_classes_ckpt != len(self.class_names):
            raise RuntimeError(
                f"Checkpoint num_classes={num_classes_ckpt} but labels.json has {len(self.class_names)}. "
                "Fix models/labels.json to match the checkpoint head."
            )

        # 4) Load weights strictly
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

            # Per-class thresholds (NO fallback)
            thresholds_t = torch.tensor(
                [float(CLASS_THRESHOLDS[name]) for name in self.class_names],
                device=scores_t.device,
                dtype=scores_t.dtype,
            )

            positive_idx = (scores_t >= thresholds_t).nonzero(as_tuple=False).flatten().tolist()
            positive_labels = [self.class_names[i] for i in positive_idx]
        else:
            scores_t = torch.softmax(logits, dim=0)
            positive_labels = []

        scores = scores_t.detach().cpu().tolist()
        label_scores = {label: float(score) for label, score in zip(self.class_names, scores)}

        k = len(self.class_names) if topk is None else min(max(1, int(topk)), len(self.class_names))
        top_scores_t, top_idx_t = torch.topk(scores_t, k=k)
        top_scores = top_scores_t.detach().cpu().tolist()
        top_idx = top_idx_t.detach().cpu().tolist()
        top_labels = [self.class_names[i] for i in top_idx]

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