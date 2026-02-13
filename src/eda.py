from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def collect_counts(data_dir: Path) -> dict:
    summary = {}
    for split in ["train", "val", "test"]:
        split_path = data_dir / split
        if not split_path.exists():
            continue
        summary[split] = {}
        for label_dir in split_path.iterdir():
            if label_dir.is_dir():
                summary[split][label_dir.name] = len(list(label_dir.glob("*")))
    return summary


def plot_counts(summary: dict, output_path: Path) -> None:
    labels = sorted({label for split in summary.values() for label in split.keys()})
    splits = ["train", "val", "test"]

    x = range(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, split in enumerate(splits):
        values = [summary.get(split, {}).get(label, 0) for label in labels]
        ax.bar([v + idx * width for v in x], values, width=width, label=split)

    ax.set_xticks([v + width for v in x])
    ax.set_xticklabels(labels)
    ax.set_ylabel("Count")
    ax.set_title("Cats vs Dogs split distribution")
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/eda/class_balance.png"))
    args = parser.parse_args()

    summary = collect_counts(args.data_dir)
    print(summary)
    plot_counts(summary, args.output)


if __name__ == "__main__":
    main()
