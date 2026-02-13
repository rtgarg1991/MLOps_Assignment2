from __future__ import annotations

from typing import Optional


def _import_torch_modules():
    import torch
    import torch.nn as nn

    return torch, nn


def _simple_cnn_cls(nn):
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes: int = 2):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(64, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            return self.classifier(x)

    return SimpleCNN


def _wider_cnn_cls(nn):
    class WiderCNN(nn.Module):
        def __init__(self, num_classes: int = 2):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(128, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            return self.classifier(x)

    return WiderCNN


def torch_available() -> bool:
    try:
        _import_torch_modules()
        return True
    except Exception:
        return False


def build_model(variant: str, num_classes: int = 2):
    _, nn = _import_torch_modules()

    variant = variant.lower()
    if variant == "baseline":
        return _simple_cnn_cls(nn)(num_classes=num_classes)
    if variant == "wide":
        return _wider_cnn_cls(nn)(num_classes=num_classes)
    raise ValueError(f"Unsupported model variant: {variant}")


def choose_device(force_cpu: bool = False) -> str:
    if force_cpu:
        return "cpu"

    try:
        torch, _ = _import_torch_modules()
    except Exception:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model_bundle(model_path: str, model_variant: Optional[str] = None):
    torch, _ = _import_torch_modules()

    payload = torch.load(model_path, map_location="cpu")
    class_to_idx = payload["class_to_idx"]
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    variant = model_variant or payload.get("model_variant", "baseline")
    model = build_model(variant=variant, num_classes=len(class_to_idx))
    model.load_state_dict(payload["state_dict"])
    model.eval()

    return {
        "model": model,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "image_size": int(payload.get("image_size", 224)),
        "model_variant": variant,
    }
