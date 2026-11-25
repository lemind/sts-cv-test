"""Small helpers shared by the dataset scripts."""

from pathlib import Path

import yaml

# Same set Ultralytics accepts for detection (minus formats we never expect here).
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def read_class_names(classes_file: Path) -> list[str]:
    """Class names in file order; the line index (0-based) is the YOLO class id.

    A blank line inside the list would shift every id after it, so it is an error.
    Trailing blank lines and a UTF-8 BOM are allowed.
    """
    lines = [line.strip() for line in classes_file.read_text(encoding="utf-8-sig").splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    if any(not line for line in lines):
        raise ValueError(f"blank line inside {classes_file}: class ids would shift")
    if len(lines) != len(set(lines)):
        raise ValueError(f"duplicate class names in {classes_file}")
    return lines


def read_names_file(names_file: Path) -> list[str]:
    """Class names from a CVAT export: `data.yaml` (`names:` list or dict) or `obj.names`."""
    if names_file.suffix.lower() in (".yaml", ".yml"):
        names = yaml.safe_load(names_file.read_text(encoding="utf-8-sig"))["names"]
        if isinstance(names, dict):
            return [str(names[key]) for key in sorted(names, key=int)]
        return [str(name) for name in names]
    return read_class_names(names_file)


def list_images(images_dir: Path) -> list[Path]:
    """Image files sorted by name, so every run sees the same order."""
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def yolo_to_corners(
    x_center: float, y_center: float, width: float, height: float
) -> tuple[float, float, float, float]:
    """Normalized YOLO box -> normalized (x1, y1, x2, y2)."""
    return (
        x_center - width / 2,
        y_center - height / 2,
        x_center + width / 2,
        y_center + height / 2,
    )
