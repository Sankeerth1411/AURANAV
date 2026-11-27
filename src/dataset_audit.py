#!/usr/bin/env python3
"""Audit a YOLO detection dataset before training.

Checks split paths, image/label pairing, malformed annotations, class IDs,
normalised box bounds, empty labels, and byte-identical images across splits.
Does not modify the dataset.
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _class_names(data: Dict) -> List[str]:
    names = data.get("names", [])
    if isinstance(names, dict):
        try:
            return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
        except (TypeError, ValueError):
            return [str(names[k]) for k in sorted(names)]
    if isinstance(names, list):
        return [str(name) for name in names]
    return []


def _split_images(value, dataset_root: Path) -> List[Path]:
    if not value:
        return []
    entries = value if isinstance(value, list) else [value]
    found = []
    for entry in entries:
        path = Path(str(entry))
        if not path.is_absolute():
            path = dataset_root / path
        if path.is_file() and path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8").splitlines():
                item = line.strip()
                if item:
                    image_path = Path(item)
                    found.append(image_path if image_path.is_absolute() else dataset_root / image_path)
        elif path.is_dir():
            found.extend(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            found.append(path)
    return sorted(set(p.resolve() for p in found))


def _label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    # Standard Ultralytics layout: .../images/name.jpg -> .../labels/name.txt
    if "images" in parts:
        index = len(parts) - 1 - parts[::-1].index("images")
        parts[index] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_dataset(data_yaml: str) -> Dict:
    yaml_path = Path(data_yaml).expanduser().resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {yaml_path}")

    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    dataset_root = Path(data.get("path", yaml_path.parent))
    if not dataset_root.is_absolute():
        dataset_root = (yaml_path.parent / dataset_root).resolve()

    names = _class_names(data)
    split_reports = {}
    global_hashes = defaultdict(list)
    total_class_counts = Counter()

    for split in ("train", "val", "valid", "test"):
        if split == "valid" and "val" in data:
            continue
        images = _split_images(data.get(split), dataset_root)
        if not images:
            if split in data:
                split_reports[split] = {
                    "images": 0, "missing_labels": 0, "empty_labels": 0,
                    "malformed_lines": 0, "invalid_class_ids": 0,
                    "invalid_boxes": 0, "class_counts": {},
                    "errors": [f"No images found for configured split: {data.get(split)}"],
                }
            continue

        counts = Counter()
        missing_labels = empty_labels = malformed = invalid_ids = invalid_boxes = 0
        errors = []
        for image in images:
            label = _label_path(image)
            if not label.is_file():
                missing_labels += 1
                errors.append(f"Missing label: {label}")
            else:
                lines = label.read_text(encoding="utf-8", errors="replace").splitlines()
                valid_rows = 0
                for line_number, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue
                    fields = line.split()
                    if len(fields) != 5:
                        malformed += 1
                        errors.append(f"{label}:{line_number}: expected 5 fields, got {len(fields)}")
                        continue
                    try:
                        class_float = float(fields[0])
                        box = [float(value) for value in fields[1:]]
                    except ValueError:
                        malformed += 1
                        errors.append(f"{label}:{line_number}: non-numeric annotation")
                        continue
                    if not class_float.is_integer():
                        invalid_ids += 1
                        errors.append(f"{label}:{line_number}: class id is not an integer")
                        continue
                    class_id = int(class_float)
                    if class_id < 0 or class_id >= len(names):
                        invalid_ids += 1
                        errors.append(f"{label}:{line_number}: class id {class_id} outside [0, {len(names)-1}]")
                        continue
                    x, y, width, height = box
                    if (
                        any(not (0.0 <= v <= 1.0) for v in box)
                        or width <= 0 or height <= 0
                        or x - width / 2 < 0 or x + width / 2 > 1
                        or y - height / 2 < 0 or y + height / 2 > 1
                    ):
                        invalid_boxes += 1
                        errors.append(f"{label}:{line_number}: invalid or out-of-frame box")
                        continue
                    counts[class_id] += 1
                    total_class_counts[class_id] += 1
                    valid_rows += 1
                if lines and valid_rows == 0:
                    empty_labels += 1
                elif not lines:
                    empty_labels += 1

            try:
                image_hash = _sha256(image)
                global_hashes[image_hash].append({"split": split, "path": str(image)})
            except OSError as exc:
                errors.append(f"Cannot read image {image}: {exc}")

        split_reports[split] = {
            "images": len(images),
            "missing_labels": missing_labels,
            "empty_labels": empty_labels,
            "malformed_lines": malformed,
            "invalid_class_ids": invalid_ids,
            "invalid_boxes": invalid_boxes,
            "class_counts": {
                names[class_id]: count for class_id, count in sorted(counts.items())
            },
            "errors": errors[:200],
        }

    duplicates = []
    for entries in global_hashes.values():
        splits = {entry["split"] for entry in entries}
        if len(splits) > 1:
            duplicates.append(entries)

    report = {
        "dataset_yaml": str(yaml_path),
        "dataset_root": str(dataset_root),
        "class_names": names,
        "splits": split_reports,
        "total_class_counts": {
            names[class_id]: count for class_id, count in sorted(total_class_counts.items())
        },
        "cross_split_duplicate_groups": duplicates,
        "summary": {
            "configured_splits_found": len(split_reports),
            "cross_split_duplicate_groups": len(duplicates),
            "annotation_errors": sum(
                split["missing_labels"] + split["malformed_lines"]
                + split["invalid_class_ids"] + split["invalid_boxes"]
                for split in split_reports.values()
            ),
        },
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Audit YOLO dataset structure and labels")
    parser.add_argument("--data", required=True, help="Path to dataset data.yaml")
    parser.add_argument("--output", default="artifacts/test_results/dataset_audit.json")
    parser.add_argument("--fail-on-errors", action="store_true")
    args = parser.parse_args()

    report = audit_dataset(args.data)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Full report saved to: {output}")

    if args.fail_on_errors and (
        report["summary"]["annotation_errors"] > 0
        or report["summary"]["cross_split_duplicate_groups"] > 0
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
