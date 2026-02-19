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

# NEW: timm for ConvNeXt-v2
import timm


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
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model", "net", "model_state"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]

        tensor_items = {k: v for k, v in ckpt.items() if torch.is_tensor(v)}
        if tensor_items:
            return tensor_items

        raise RuntimeError(f"Checkpoint dict format not recognized. Top-level keys: {list(ckpt.keys())[:20]}")

    if isinstance(ckpt, (dict,)):
        raise RuntimeError(f"Checkpoint format not supported. Got: {type(ckpt)}")

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


def _infer_arch_from_state_keys(state: Dict[str, torch.Tensor]) -> Optional[str]:
    # ConvNeXt-v2 checkpoints in your error look like: backbone.stem..., backbone.stages..., backbone.head.norm...
    keys = list(state.keys())
    if any(k.startswith("backbone.stem") or k.startswith("backbone.stages") for k in keys):
        # guess tiny/base/large by classifier in_features if available
        w = state.get("classifier.1.weight")
        if isinstance(w, torch.Tensor) and w.ndim == 2:
            in_features = w.shape[1]
            if in_features == 768:
                return "convnextv2_tiny"
            if in_features == 1024:
                return "convnextv2_base"
            if in_features == 1536:
                return "convnextv2_large"
        # fallback
        return "convnextv2_tiny"
    return None


class _ConvNeXtV2Wrapper(nn.Module):
    """
    Matches checkpoints that store:
      backbone.*  +  classifier.1.weight/bias
    (no classifier.0 params in the checkpoint)
    """
    def __init__(self, backbone_name: str, num_classes: int):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)

        feat_dim = getattr(self.backbone, "num_features", None)
        if feat_dim is None:
            raise RuntimeError("timm model missing num_features; check timm version / backbone name.")

        self.classifier = nn.Sequential(
            nn.Identity(),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_features(x)
        feats = self.backbone.forward_head(feats, pre_logits=True)
        return self.classifier(feats)


class SkinConcernClassifier:
    def __init__(
        self,
        ckpt_path: str,
        device: str = "cpu",
        labels_path: str = "models/labels.json",
        class_names: Optional[List[str]] = None,
        img_size: int = IMG_SIZE,
        arch: Optional[str] = None,  # allow auto-detect
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
                    f"labels_path not found: {labels_path}. Create it (models/labels.json) or pass class_names=[...]."
                )

        if not self.class_names:
            raise ValueError("No class names found.")

        # 2) Load checkpoint first (so we can auto-detect arch cleanly)
        # PyTorch 2.6+ changed weights_only default; we try safe load then fallback (only if you trust the ckpt).
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")  # works for plain state_dict .pth
        except Exception:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        state = _extract_state_dict(ckpt)
        state = _clean_state_dict_keys(state)

        # 3) Decide arch
        arch = (arch or "").lower().strip() or _infer_arch_from_state_keys(state) or "efficientnet_v2_s"

        # 4) Build model
        if arch.startswith("convnextv2"):
            # timm uses names like "convnextv2_tiny"
            self.model = _ConvNeXtV2Wrapper(arch, num_classes=len(self.class_names))
        elif arch == "efficientnet_v2_s":
            self.model = models.efficientnet_v2_s(weights=None)
            in_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(in_features, len(self.class_names))
        elif arch == "efficientnet_b0":
            self.model = models.efficientnet_b0(weights=None)
            in_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(in_features, len(self.class_names))
        else:
            raise ValueError(f"Unsupported arch: {arch}")

        # 5) Load weights (strict=True so you immediately know if model/ckpt mismatch)
        self.model.load_state_dict(state, strict=True)

        self.model.to(self.device)
        self.model.eval()

        # 6) Preprocess (keep same unless your training used different normalization)
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
            threshold = float(MULTILABEL_THRESHOLD)
            positive_idx = (scores_t >= threshold).nonzero(as_tuple=False).flatten().tolist()
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
