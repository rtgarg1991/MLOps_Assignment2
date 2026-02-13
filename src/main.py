from __future__ import annotations

import io
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Dict

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from src.config import DEFAULT_IMAGE_SIZE, DEFAULT_MODEL_PATH, MEAN, STD
from src.inference import format_probabilities, top_prediction
from src.model import load_model_bundle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("cats-dogs-api")

MODEL_PATH = os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))
ENABLE_REAL_MODEL = os.getenv("ENABLE_REAL_MODEL", "0") == "1"

model_bundle = {
    "is_real_model": False,
    "class_names": ["cat", "dog"],
    "image_size": DEFAULT_IMAGE_SIZE,
    "model": None,
}

REQUEST_COUNTER = Counter(
    "cats_dogs_requests_total", "Total requests", ["path", "method", "status"]
)
PREDICTION_COUNTER = Counter(
    "cats_dogs_predictions_total", "Total prediction calls", ["label"]
)
REQUEST_LATENCY = Histogram(
    "cats_dogs_request_latency_seconds",
    "Request latency in seconds",
    ["path", "method"],
)
INFERENCE_LATENCY = Histogram(
    "cats_dogs_inference_latency_seconds", "Model inference latency"
)


class DummyImageModel:
    def predict(self, image_array: np.ndarray) -> np.ndarray:
        mean_val = float(np.mean(image_array))
        if mean_val > 127:
            return np.array([0.65, 0.35], dtype=np.float32)
        return np.array([0.35, 0.65], dtype=np.float32)


_dummy_model = DummyImageModel()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_bundle

    if ENABLE_REAL_MODEL and os.path.exists(MODEL_PATH):
        try:
            payload = load_model_bundle(MODEL_PATH)
            class_names = [
                payload["idx_to_class"][idx]
                for idx in sorted(payload["idx_to_class"].keys())
            ]
            model_bundle = {
                "is_real_model": True,
                "class_names": class_names,
                "image_size": payload["image_size"],
                "model": payload["model"],
            }
            logger.info("Loaded trained model from %s", MODEL_PATH)
        except Exception as exc:
            logger.exception("Failed loading trained model, using dummy: %s", exc)
    else:
        logger.warning(
            "Using dummy model. Set ENABLE_REAL_MODEL=1 and provide MODEL_PATH to load trained weights."
        )

    yield


app = FastAPI(title="Cats vs Dogs Classifier API", lifespan=lifespan)
Instrumentator().instrument(app).expose(app)


@app.middleware("http")
async def log_and_measure(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    REQUEST_COUNTER.labels(
        path=request.url.path, method=request.method, status=str(response.status_code)
    ).inc()
    REQUEST_LATENCY.labels(path=request.url.path, method=request.method).observe(duration)
    logger.info(
        "path=%s method=%s status=%s duration_ms=%.2f",
        request.url.path,
        request.method,
        response.status_code,
        duration * 1000,
    )
    return response


@app.get("/health")
def health() -> Dict[str, object]:
    return {
        "status": "healthy",
        "model_loaded": bool(model_bundle["is_real_model"]),
        "model_path": MODEL_PATH,
        "real_model_enabled": ENABLE_REAL_MODEL,
    }


def _predict_probabilities(image: Image.Image):
    if model_bundle["is_real_model"]:
        import torch
        from torchvision import transforms

        transform = transforms.Compose(
            [
                transforms.Resize((model_bundle["image_size"], model_bundle["image_size"])),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ]
        )
        tensor = transform(image).unsqueeze(0)

        with INFERENCE_LATENCY.time():
            with torch.no_grad():
                logits = model_bundle["model"](tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return probs.tolist()

    arr = np.array(image)
    with INFERENCE_LATENCY.time():
        probs = _dummy_model.predict(arr)
    return probs.tolist()


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, object]:
    content_type = (file.content_type or "").lower()
    if "image" not in content_type:
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image payload") from exc

    probs = _predict_probabilities(image)
    prob_map = format_probabilities(model_bundle["class_names"], probs)
    label, confidence = top_prediction(prob_map)

    PREDICTION_COUNTER.labels(label=label).inc()

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": prob_map,
        "model_loaded": bool(model_bundle["is_real_model"]),
    }
