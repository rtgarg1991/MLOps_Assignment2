from pathlib import Path

from PIL import Image

from src.feature_engineering import generate_augmented_preview


def test_generate_augmented_preview_outputs_files(tmp_path: Path):
    input_image = tmp_path / "input.jpg"
    Image.new("RGB", (64, 64), color=(150, 120, 90)).save(input_image)

    output_dir = tmp_path / "preview"
    generate_augmented_preview(input_image, output_dir, image_size=128)

    generated = list(output_dir.glob("*.jpg"))
    assert len(generated) >= 4
