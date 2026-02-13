from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def build_rows(processed_dir: Path, per_class: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []

    for label in ["cat", "dog"]:
        candidates = sorted((processed_dir / "test" / label).glob("*.jpg"))
        if not candidates:
            continue
        k = min(per_class, len(candidates))
        chosen = rng.sample(candidates, k)
        rows.extend(
            {"image_path": str(path.resolve()), "label": label} for path in chosen
        )

    rng.shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-csv", type=Path, default=Path("artifacts/eval_batch.csv"))
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = build_rows(args.processed_dir, args.per_class, args.seed)

    if not rows:
        raise RuntimeError(
            f"No test images found under {args.processed_dir / 'test'}"
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote eval batch to {args.output_csv} with {len(rows)} rows")


if __name__ == "__main__":
    main()
