import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "224"))
DEFAULT_MODEL_VARIANT = os.getenv("MODEL_VARIANT", "baseline")
DEFAULT_MODEL_PATH = Path(os.getenv("MODEL_PATH", PROJECT_ROOT / "models" / "model.pt"))

CLASS_NAMES = ("cat", "dog")

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
