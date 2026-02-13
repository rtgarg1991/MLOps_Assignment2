from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

LABELS = ("cat", "dog")


def split_indices(
    total: int, train_ratio: float, val_ratio: float, seed: int
) -> Tuple[List[int], List[int], List[int]]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")

    indices = list(range(total))
    random.Random(seed).shuffle(indices)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    return train_idx, val_idx, test_idx


def preprocess_image(
    source: Path, destination: Path, image_size: int
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as img:
        img = img.convert("RGB")
        img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
        img.save(destination, quality=95)


def preprocess_dataset(
    input_dir: Path,
    output_dir: Path,
    image_size: int = 224,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, Dict[str, int]]:
    print(f"[preprocess] Input: {input_dir}")
    print(f"[preprocess] Output: {output_dir}")
    print(
        f"[preprocess] Image size: {image_size}, split: {train_ratio}/{val_ratio}/{1.0 - train_ratio - val_ratio:.1f}"
    )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Dict[str, int]] = {
        "train": {},
        "val": {},
        "test": {},
    }

    for label in LABELS:
        label_dir = input_dir / label
        if not label_dir.exists():
            raise FileNotFoundError(
                f"Label directory not found: {label_dir} "
                f"- available: {[d.name for d in input_dir.iterdir() if d.is_dir()]}"
            )
        files = sorted(label_dir.glob("*"))
        files = [f for f in files if f.is_file()]
        print(f"[preprocess] {label}: {len(files)} images found")

        train_idx, val_idx, test_idx = split_indices(
            len(files), train_ratio, val_ratio, seed
        )

        split_map = {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        }

        for split_name, idxs in split_map.items():
            summary[split_name][label] = len(idxs)
            for idx in idxs:
                src = files[idx]
                dst = output_dir / split_name / label / src.name
                preprocess_image(src, dst, image_size=image_size)

    metadata = {
        "image_size": image_size,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": 1.0 - train_ratio - val_ratio,
        "counts": summary,
    }

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed")
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = preprocess_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print("Preprocessing complete")
    for split_name, counts in summary.items():
        total = sum(counts.values())
        print(f"  {split_name}: {total} ({counts})")


if __name__ == "__main__":
    main()
