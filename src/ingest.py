from __future__ import annotations

import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABELS = ("cat", "dog")


def _label_from_path(path: Path) -> str | None:
    parts = [p.lower() for p in path.parts]
    if any("cat" == p or p.startswith("cat") for p in parts):
        return "cat"
    if any("dog" == p or p.startswith("dog") for p in parts):
        return "dog"
    return None


def collect_labeled_images(source_dir: Path) -> Dict[str, List[Path]]:
    items = {label: [] for label in LABELS}
    for path in source_dir.rglob("*"):
        if (
            not path.is_file()
            or path.suffix.lower() not in SUPPORTED_EXTENSIONS
        ):
            continue
        label = _label_from_path(path)
        if label:
            items[label].append(path)
    return items


def copy_dataset(
    source_dir: Path, output_dir: Path, max_per_class: int | None = None
) -> Dict[str, int]:
    images = collect_labeled_images(source_dir)
    counts = {}

    for label, paths in images.items():
        dest_dir = output_dir / label
        dest_dir.mkdir(parents=True, exist_ok=True)

        selected = paths[:max_per_class] if max_per_class else paths
        for src in selected:
            digest = hashlib.sha1(str(src).encode("utf-8")).hexdigest()[:16]
            dst = dest_dir / f"{digest}{src.suffix.lower()}"
            shutil.copy2(src, dst)

        counts[label] = len(selected)

    return counts


def extract_zip(zip_path: Path, temp_dir: Path) -> Path:
    temp_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(temp_dir)
    return temp_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir", type=Path, help="Dataset root directory"
    )
    parser.add_argument("--zip-path", type=Path, help="Optional dataset zip")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--max-per-class", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.source_dir and not args.zip_path:
        raise ValueError("Provide either --source-dir or --zip-path")

    source = args.source_dir
    cleanup_dir = None

    if args.zip_path:
        cleanup_dir = Path("/tmp/cats_dogs_raw")
        source = extract_zip(args.zip_path, cleanup_dir)

    assert source is not None
    print(f"[ingest] Source: {source}")
    print(f"[ingest] Output: {args.output_dir}")

    if not source.exists():
        raise FileNotFoundError(f"Source directory not found: {source}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    counts = copy_dataset(source, args.output_dir, args.max_per_class)
    print("Ingestion complete")
    for label, count in counts.items():
        print(f"  {label}: {count}")
    total = sum(counts.values())
    if total == 0:
        raise RuntimeError(
            "No images found - check source directory structure"
        )
    print(f"  total: {total}")

    if cleanup_dir and cleanup_dir.exists():
        shutil.rmtree(cleanup_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
