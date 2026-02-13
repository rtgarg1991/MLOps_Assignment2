from pathlib import Path

from PIL import Image

from src.ingest import copy_dataset


def create_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=(120, 10, 20)).save(path)


def test_copy_dataset_detects_cat_dog(tmp_path: Path):
    source = tmp_path / "source"
    create_image(source / "Cat" / "cat_1.jpg")
    create_image(source / "Dog" / "dog_1.jpg")

    out_dir = tmp_path / "raw"
    counts = copy_dataset(source, out_dir)

    assert counts["cat"] == 1
    assert counts["dog"] == 1
    assert len(list((out_dir / "cat").glob("*"))) == 1
    assert len(list((out_dir / "dog").glob("*"))) == 1
