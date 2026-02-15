from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

try:
    import mlflow
except ImportError:  # pragma: no cover
    mlflow = None

try:
    from google.cloud import storage
except ImportError:  # pragma: no cover
    storage = None

from src.config import MEAN, STD
from src.model import build_model, choose_device


def _load_torch_stack():
    import torch
    from torch import nn
    from torch.optim import Adam
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    return torch, nn, Adam, DataLoader, datasets, transforms


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def compute_accuracy(preds: Iterable[int], targets: Iterable[int]) -> float:
    preds_list = list(preds)
    targets_list = list(targets)
    if not targets_list:
        return 0.0
    correct = sum(int(p == t) for p, t in zip(preds_list, targets_list))
    return correct / len(targets_list)


def upload_file_to_gcs(
    bucket_name: str, source_path: Path, dest_path: str
) -> None:
    if storage is None:
        raise RuntimeError("google-cloud-storage is not installed")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(dest_path)
    blob.upload_from_filename(str(source_path))


def download_prefix_from_gcs(
    bucket_name: str,
    prefix: str,
    destination_dir: Path,
    max_workers: int = 16,
) -> None:
    if storage is None:
        raise RuntimeError("google-cloud-storage is not installed")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    client = storage.Client()
    blobs = list(client.list_blobs(bucket_name, prefix=prefix))
    blobs = [b for b in blobs if not b.name.endswith("/")]

    if not blobs:
        raise FileNotFoundError(
            f"No objects found in gs://{bucket_name}/{prefix}"
        )

    print(
        f"[gcs] Downloading {len(blobs)} files with {max_workers} threads..."
    )
    bucket = client.bucket(bucket_name)

    def _download_one(blob_name: str) -> str:
        relative = Path(blob_name).relative_to(prefix)
        local_path = destination_dir / relative
        local_path.parent.mkdir(parents=True, exist_ok=True)
        bucket.blob(blob_name).download_to_filename(str(local_path))
        return blob_name

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_download_one, b.name): b.name for b in blobs}
        for future in as_completed(futures):
            future.result()
            done += 1
            if done % 500 == 0 or done == len(blobs):
                print(f"[gcs]   downloaded {done}/{len(blobs)}")

    print(f"[gcs] Download complete: {done} files")


def upload_directory_to_gcs(
    bucket_name: str,
    local_dir: Path,
    gcs_prefix: str,
    max_workers: int = 16,
) -> None:
    if storage is None:
        raise RuntimeError("google-cloud-storage is not installed")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import os

    files = []
    for root, _dirs, fnames in os.walk(local_dir):
        for fname in fnames:
            files.append(Path(root) / fname)

    if not files:
        print(f"[gcs] No files to upload from {local_dir}")
        return

    print(f"[gcs] Uploading {len(files)} files with {max_workers} threads...")
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    def _upload_one(local_path: Path) -> str:
        rel = local_path.relative_to(local_dir)
        gcs_key = f"{gcs_prefix.rstrip('/')}/{rel}"
        bucket.blob(gcs_key).upload_from_filename(str(local_path))
        return gcs_key

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_upload_one, f): f for f in files}
        for future in as_completed(futures):
            future.result()
            done += 1
            if done % 500 == 0 or done == len(files):
                print(f"[gcs]   uploaded {done}/{len(files)}")

    print(f"[gcs] Upload complete: {done} files")


def make_transforms(image_size: int):
    _, _, _, _, _, transforms = _load_torch_stack()
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )

    return train_transform, eval_transform


def build_dataloaders(
    data_dir: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[object, object, object, Dict[int, str]]:
    torch, _, _, DataLoader, datasets, _ = _load_torch_stack()
    train_transform, eval_transform = make_transforms(image_size)

    train_ds = datasets.ImageFolder(
        str(data_dir / "train"), transform=train_transform
    )
    val_ds = datasets.ImageFolder(
        str(data_dir / "val"), transform=eval_transform
    )
    test_ds = datasets.ImageFolder(
        str(data_dir / "test"), transform=eval_transform
    )

    if set(train_ds.classes) != {"cat", "dog"}:
        raise ValueError(
            f"Expected classes {{'cat','dog'}} but got {train_ds.classes}"
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    idx_to_class = {idx: cls for cls, idx in train_ds.class_to_idx.items()}
    return train_loader, val_loader, test_loader, idx_to_class


def run_one_epoch(
    model,
    loader,
    criterion,
    device: str,
    optimizer=None,
) -> Tuple[float, float, List[int], List[int]]:
    torch, _, _, _, _, _ = _load_torch_stack()

    is_training = optimizer is not None
    model.train(is_training)

    losses: List[float] = []
    preds_all: List[int] = []
    targets_all: List[int] = []

    total_batches = len(loader)
    one_third = max(1, total_batches // 3)

    for i, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            logits = model(inputs)
            loss = criterion(logits, targets)
            preds = torch.argmax(logits, dim=1)

            if is_training:
                loss.backward()
                optimizer.step()

        losses.append(float(loss.item()))
        preds_all.extend(preds.detach().cpu().numpy().tolist())
        targets_all.extend(targets.detach().cpu().numpy().tolist())

        if (i + 1) % one_third == 0 or (i + 1) == total_batches:
            status = "Training" if is_training else "Evaluating"
            percent = ((i + 1) / total_batches) * 100
            current_loss = float(np.mean(losses))
            print(
                f"[{status}] Batch {i + 1}/{total_batches} ({percent:.0f}%) | "
                f"Avg Loss: {current_loss:.4f}"
            )

    avg_loss = float(np.mean(losses)) if losses else 0.0
    acc = compute_accuracy(preds_all, targets_all)
    return avg_loss, acc, preds_all, targets_all


def save_artifacts(
    artifact_dir: Path,
    history: Dict[str, List[float]],
    y_true: List[int],
    y_pred: List[int],
    class_names: Dict[int, str],
    metrics: Dict[str, float],
) -> Tuple[Path, Path, Path]:
    artifact_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = artifact_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred, labels=list(class_names.keys()))
    cm_fig, cm_ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        cm, display_labels=[class_names[i] for i in class_names]
    )
    disp.plot(ax=cm_ax, cmap="Blues", colorbar=True)
    cm_ax.set_title("Cats vs Dogs - Confusion Matrix")
    cm_fig.tight_layout()
    cm_path = artifact_dir / "confusion_matrix.png"
    cm_fig.savefig(cm_path, dpi=200)
    plt.close(cm_fig)

    loss_fig, loss_ax = plt.subplots(figsize=(7, 5))
    loss_ax.plot(history["train_loss"], label="train_loss")
    loss_ax.plot(history["val_loss"], label="val_loss")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_ax.set_title("Training vs Validation Loss")
    loss_ax.grid(alpha=0.3)
    loss_ax.legend()
    loss_fig.tight_layout()
    loss_path = artifact_dir / "loss_curve.png"
    loss_fig.savefig(loss_path, dpi=200)
    plt.close(loss_fig)

    return metrics_path, cm_path, loss_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/processed")
    )
    parser.add_argument("--dataset-gcs-prefix", type=str, default="")
    parser.add_argument(
        "--local-cache-dir", type=Path, default=Path("/tmp/cats-dogs-data")
    )
    parser.add_argument(
        "--output-model", type=Path, default=Path("models/model.pt")
    )
    parser.add_argument(
        "--artifact-dir", type=Path, default=Path("artifacts/training")
    )
    parser.add_argument(
        "--model-variant",
        type=str,
        default="baseline",
        choices=["baseline", "wide"],
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bucket", type=str, default="")
    parser.add_argument(
        "--model-gcs-prefix", type=str, default="models/latest"
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"),
    )
    parser.add_argument("--experiment-name", type=str, default="cats-vs-dogs")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--disable-mlflow", action="store_true")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--git-sha", type=str, default="")
    return parser.parse_args()


def main() -> None:
    torch, nn, Adam, _, _, _ = _load_torch_stack()

    args = parse_args()
    set_seed(args.seed)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    training_data_dir = args.data_dir

    if args.dataset_gcs_prefix:
        if not args.bucket:
            raise ValueError(
                "--bucket is required when --dataset-gcs-prefix is used"
            )
        args.local_cache_dir.mkdir(parents=True, exist_ok=True)
        download_prefix_from_gcs(
            bucket_name=args.bucket,
            prefix=args.dataset_gcs_prefix.rstrip("/"),
            destination_dir=args.local_cache_dir,
        )
        training_data_dir = args.local_cache_dir

    train_loader, val_loader, test_loader, idx_to_class = build_dataloaders(
        training_data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_model(args.model_variant, num_classes=len(idx_to_class))
    device = choose_device(force_cpu=args.force_cpu)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    import time as _time

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = -1.0
    best_state = None
    train_start = _time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, _, _ = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )
        val_loss, val_acc, _, _ = run_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

        print(
            f"epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    training_duration_sec = _time.time() - train_start

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc, test_preds, test_targets = run_one_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
    )

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_to_idx": {name: idx for idx, name in idx_to_class.items()},
            "image_size": args.image_size,
            "model_variant": args.model_variant,
        },
        args.output_model,
    )

    metrics = {
        "best_val_accuracy": best_val_acc,
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "model_variant": args.model_variant,
        "image_size": args.image_size,
    }

    metrics_path, cm_path, loss_path = save_artifacts(
        artifact_dir=args.artifact_dir,
        history=history,
        y_true=test_targets,
        y_pred=test_preds,
        class_names=idx_to_class,
        metrics=metrics,
    )

    if not args.disable_mlflow and mlflow is not None:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(args.experiment_name)
        with mlflow.start_run(run_name=args.run_name or None):
            # -- params --
            mlflow.log_params(
                {
                    "model_variant": args.model_variant,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "image_size": args.image_size,
                    "num_workers": args.num_workers,
                    "seed": args.seed,
                }
            )

            # -- dataset info --
            mlflow.log_params(
                {
                    "train_samples": len(train_loader.dataset),
                    "val_samples": len(val_loader.dataset),
                    "test_samples": len(test_loader.dataset),
                    "num_classes": len(idx_to_class),
                    "class_names": ",".join(
                        idx_to_class[i] for i in sorted(idx_to_class)
                    ),
                }
            )

            # -- per-epoch metrics (for MLflow charts) --
            for step, (tl, ta, vl, va) in enumerate(
                zip(
                    history["train_loss"],
                    history["train_acc"],
                    history["val_loss"],
                    history["val_acc"],
                )
            ):
                mlflow.log_metrics(
                    {
                        "train_loss": tl,
                        "train_acc": ta,
                        "val_loss": vl,
                        "val_acc": va,
                    },
                    step=step + 1,
                )

            # -- final metrics --
            mlflow.log_metrics(
                {
                    "best_val_accuracy": best_val_acc,
                    "test_accuracy": test_acc,
                    "test_loss": test_loss,
                    "training_duration_sec": round(training_duration_sec, 1),
                }
            )

            # -- tags --
            tags = {
                "run_type": "ci" if args.git_sha else "local",
                "device": device,
            }
            if args.git_sha:
                tags["git_sha"] = args.git_sha
            if args.bucket:
                tags["dataset_source"] = (
                    f"gs://{args.bucket}/{args.dataset_gcs_prefix}"
                    if args.dataset_gcs_prefix
                    else f"gs://{args.bucket}"
                )
            mlflow.set_tags(tags)

            # -- artifacts --
            mlflow.log_artifact(str(args.output_model), artifact_path="model")
            mlflow.log_artifact(str(metrics_path), artifact_path="metrics")
            mlflow.log_artifact(str(cm_path), artifact_path="metrics")
            mlflow.log_artifact(str(loss_path), artifact_path="metrics")

    if args.bucket:
        upload_file_to_gcs(
            args.bucket,
            args.output_model,
            f"{args.model_gcs_prefix.rstrip('/')}/model.pt",
        )
        upload_file_to_gcs(
            args.bucket,
            metrics_path,
            f"{args.model_gcs_prefix.rstrip('/')}/metrics.json",
        )
        upload_file_to_gcs(
            args.bucket,
            cm_path,
            f"{args.model_gcs_prefix.rstrip('/')}/confusion_matrix.png",
        )
        upload_file_to_gcs(
            args.bucket,
            loss_path,
            f"{args.model_gcs_prefix.rstrip('/')}/loss_curve.png",
        )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
