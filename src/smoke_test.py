from __future__ import annotations

import argparse
import io
import sys
import time

import requests
from PIL import Image


def build_sample_image() -> bytes:
    image = Image.new("RGB", (224, 224), color=(120, 80, 200))
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return buf.getvalue()


def _request_with_retry(method, url, max_retries=5, backoff=3, **kwargs):
    """Make an HTTP request with exponential backoff retry."""
    for attempt in range(1, max_retries + 1):
        try:
            resp = method(url, **kwargs)
            if resp.status_code < 500:
                return resp
            print(
                f"[smoke] Attempt {attempt}/{max_retries}: "
                f"got {resp.status_code}, retrying in {backoff}s..."
            )
        except requests.exceptions.ConnectionError as e:
            print(
                f"[smoke] Attempt {attempt}/{max_retries}: "
                f"connection error ({e}), retrying in {backoff}s..."
            )
        except requests.exceptions.Timeout:
            print(
                f"[smoke] Attempt {attempt}/{max_retries}: "
                f"timeout, retrying in {backoff}s..."
            )
        if attempt < max_retries:
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)
    print(f"[smoke] All {max_retries} attempts failed for {url}")
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", type=str, required=True)
    args = parser.parse_args()

    health_url = f"{args.base_url.rstrip('/')}/health"
    predict_url = f"{args.base_url.rstrip('/')}/predict"

    print(f"[smoke] Checking health: {health_url}")
    health_resp = _request_with_retry(requests.get, health_url, timeout=20)
    if health_resp.status_code != 200:
        print(
            f"Health check failed: {health_resp.status_code} {health_resp.text}"
        )
        sys.exit(1)

    health_body = health_resp.json()
    if not health_body.get("model_loaded"):
        print(
            f"Real model not loaded - serving dummy predictions: {health_body}"
        )
        sys.exit(1)

    files = {
        "file": ("smoke.jpg", build_sample_image(), "image/jpeg"),
    }
    print(f"[smoke] Sending prediction request to: {predict_url}")
    pred_resp = _request_with_retry(
        requests.post, predict_url, files=files, timeout=30
    )
    if pred_resp.status_code != 200:
        print(
            f"Prediction check failed: {pred_resp.status_code} {pred_resp.text}"
        )
        sys.exit(1)

    body = pred_resp.json()
    required = {"label", "confidence", "probabilities"}
    if not required.issubset(set(body.keys())):
        print(f"Prediction response missing keys: {body}")
        sys.exit(1)

    print(
        f"[smoke] Prediction result: label={body.get('label')} "
        f"confidence={body.get('confidence')}"
    )
    print("Smoke test passed")


if __name__ == "__main__":
    main()
