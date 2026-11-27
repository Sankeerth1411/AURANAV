"""Unit tests for the read-only YOLO dataset auditor."""

from pathlib import Path

import yaml

from src.dataset_audit import audit_dataset


def make_image(path: Path, content: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def make_label(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_audit_counts_valid_labels_and_detects_missing_labels(tmp_path):
    root = tmp_path / "dataset"
    make_image(root / "train/images/a.jpg", b"train-image-a")
    make_image(root / "train/images/b.jpg", b"train-image-b")
    make_label(root / "train/labels/a.txt", "0 0.5 0.5 0.2 0.2\n")
    data_yaml = root / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(root),
        "train": "train/images",
        "names": ["pothole"],
    }), encoding="utf-8")

    report = audit_dataset(str(data_yaml))
    train = report["splits"]["train"]
    assert train["images"] == 2
    assert train["missing_labels"] == 1
    assert train["class_counts"] == {"pothole": 1}


def test_audit_detects_invalid_boxes_and_class_ids(tmp_path):
    root = tmp_path / "dataset"
    make_image(root / "train/images/a.jpg", b"image")
    make_label(
        root / "train/labels/a.txt",
        "0 0.5 0.5 0.2 0.2\n"
        "3 0.5 0.5 0.2 0.2\n"
        "0 0.99 0.5 0.2 0.2\n",
    )
    data_yaml = root / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(root),
        "train": "train/images",
        "names": ["pothole"],
    }), encoding="utf-8")

    report = audit_dataset(str(data_yaml))
    train = report["splits"]["train"]
    assert train["invalid_class_ids"] == 1
    assert train["invalid_boxes"] == 1


def test_audit_flags_identical_images_across_splits(tmp_path):
    root = tmp_path / "dataset"
    content = b"same-image-content"
    for split in ("train", "val"):
        make_image(root / split / "images/a.jpg", content)
        make_label(root / split / "labels/a.txt", "0 0.5 0.5 0.2 0.2\n")
    data_yaml = root / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(root),
        "train": "train/images",
        "val": "val/images",
        "names": ["pothole"],
    }), encoding="utf-8")

    report = audit_dataset(str(data_yaml))
    assert report["summary"]["cross_split_duplicate_groups"] == 1
