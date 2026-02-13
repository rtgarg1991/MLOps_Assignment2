from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps


def generate_augmented_preview(
    input_image: Path,
    output_dir: Path,
    image_size: int = 224,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(input_image) as img:
        img = img.convert("RGB").resize((image_size, image_size), Image.Resampling.LANCZOS)

        variants = {
            "original": img,
            "flip_horizontal": ImageOps.mirror(img),
            "flip_vertical": ImageOps.flip(img),
            "rotate_10": img.rotate(10),
            "rotate_minus_10": img.rotate(-10),
        }

        for name, variant in variants.items():
            variant.save(output_dir / f"{name}.jpg", quality=95)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/augmentation_preview"))
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    generate_augmented_preview(args.input_image, args.output_dir, args.image_size)
    print(f"Augmentation preview generated at {args.output_dir}")


if __name__ == "__main__":
    main()
