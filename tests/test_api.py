from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

import src.main as main


def make_test_image_bytes() -> bytes:
    img = Image.new("RGB", (128, 128), color=(200, 100, 50))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_health_endpoint():
    with TestClient(main.app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert "model_loaded" in body


def test_predict_endpoint_structure():
    with TestClient(main.app) as client:
        response = client.post(
            "/predict",
            files={"file": ("sample.jpg", make_test_image_bytes(), "image/jpeg")},
        )
    assert response.status_code == 200
    body = response.json()
    assert "label" in body
    assert "confidence" in body
    assert "probabilities" in body
    assert body["label"] in {"cat", "dog"}


def test_predict_rejects_non_image():
    with TestClient(main.app) as client:
        response = client.post(
            "/predict",
            files={"file": ("sample.txt", b"hello", "text/plain")},
        )
    assert response.status_code == 400
