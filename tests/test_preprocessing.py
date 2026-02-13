from pathlib import Path

from PIL import Image

from src.preprocessing import preprocess_image, split_indices


def test_split_indices_shape():
    train, val, test = split_indices(total=10, train_ratio=0.8, val_ratio=0.1, seed=42)
    assert len(train) == 8
    assert len(val) == 1
    assert len(test) == 1


def test_preprocess_image_resizes(tmp_path: Path):
    source = tmp_path / "in.jpg"
    target = tmp_path / "out.jpg"

    Image.new("RGB", (60, 40), color=(10, 10, 10)).save(source)
    preprocess_image(source, target, image_size=224)

    with Image.open(target) as img:
        assert img.size == (224, 224)
