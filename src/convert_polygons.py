
from pathlib import Path
import shutil

SOURCE = Path("data/raw")
DEST = Path("data/detection")

def convert_label_line(line):
    values = line.split()
    if not values:
        return line

    class_id = values[0]
    coords = [float(v) for v in values[1:]]

    # Already a YOLO bounding box
    if len(coords) == 4:
        return f"{class_id} " + " ".join(map(str, coords))

    # Polygon needs at least 3 points (6 coordinates)
    if len(coords) < 6 or len(coords) % 2 != 0:
        raise ValueError(f"Invalid annotation: {line}")

    xs = coords[0::2]
    ys = coords[1::2]

    if any(not 0 <= v <= 1 for v in coords):
        raise ValueError(f"Coordinates outside 0–1: {line}")

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    return (
        f"{class_id} {x_center:.8f} {y_center:.8f} "
        f"{width:.8f} {height:.8f}"
    )

def main():
    if not SOURCE.is_dir():
        raise FileNotFoundError(f"Dataset not found: {SOURCE}")
    if DEST.exists():
        raise FileExistsError(
            f"{DEST} already exists. Rename it or choose another destination."
        )

    # Copy the original dataset; never modify its labels.
    shutil.copytree(SOURCE, DEST)
    converted = 0

    try:
        for split in ("train", "valid", "test"):
            labels_dir = DEST / split / "labels"
            if not labels_dir.is_dir():
                raise FileNotFoundError(f"Missing folder: {labels_dir}")

            for path in labels_dir.glob("*.txt"):
                original = path.read_text(encoding="utf-8")
                lines = original.splitlines()
                output = [convert_label_line(line) for line in lines]
                path.write_text(
                    "\n".join(output) + ("\n" if output else ""),
                    encoding="utf-8",
                )
                converted += 1

        print(f"Converted labels in {converted} files.")
        print(f"Original dataset preserved at: {SOURCE}")
        print(f"Converted dataset saved at: {DEST}")
        print("Next: update data.yaml paths and validate the converted dataset.")

    except Exception:
        shutil.rmtree(DEST)
        raise

if __name__ == "__main__":
    main()
