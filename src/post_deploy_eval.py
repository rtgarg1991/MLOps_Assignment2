from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import requests


def evaluate(base_url: str, csv_path: Path) -> dict:
    total = 0
    correct = 0
    details = []

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_path = Path(row["image_path"])
            true_label = row["label"].strip().lower()

            with image_path.open("rb") as imgf:
                files = {"file": (image_path.name, imgf, "image/jpeg")}
                resp = requests.post(f"{base_url.rstrip('/')}/predict", files=files, timeout=30)
                resp.raise_for_status()
                pred = resp.json()["label"].strip().lower()

            total += 1
            hit = pred == true_label
            correct += int(hit)
            details.append({"image": str(image_path), "label": true_label, "prediction": pred, "correct": hit})

    accuracy = (correct / total) if total else 0.0
    return {"total": total, "correct": correct, "accuracy": accuracy, "details": details}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/post_deploy_eval.json"))
    args = parser.parse_args()

    result = evaluate(args.base_url, args.input_csv)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps({"accuracy": result["accuracy"], "total": result["total"]}, indent=2))


if __name__ == "__main__":
    main()
