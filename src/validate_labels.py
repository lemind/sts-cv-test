"""Check images and YOLO labels; report problems, never fix them.

Usage:
    python -m src.validate_labels --input test_data/traffic_v5_mini
    python -m src.validate_labels --input test_data/traffic_v5_mini --labels annotations/final \
        --strict --json reports/validation.json
    python -m src.validate_labels --input test_data/traffic_v5_mini --labels cvat/export/labels \
        --names cvat/export/data.yaml

Exit code 1 when any error is found, 2 when an input path is missing,
so other scripts can stop on bad labels.
"""

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image

from src.utils import (
    IMAGE_EXTENSIONS,
    list_images,
    read_class_names,
    read_names_file,
    yolo_to_corners,
)

# Triage only: a box crossing the border by more than this (fraction of the image side) is an
# error, less is a warning. With --strict (final labels) any real overflow is an error.
OUT_OF_BOUNDS_TOLERANCE = 0.01
# Labels are written with 6 decimals; a box touching the edge can land a fraction of a pixel
# outside it. Used in pixels when the image size is known, normalized otherwise.
ROUNDING_TOLERANCE_PIXELS = 0.5
ROUNDING_TOLERANCE_NORMALIZED = 1e-4
TINY_BOX_PIXELS = 2
NEAR_DUPLICATE_IOU = 0.9
# Ultralytics drops images smaller than this or in other formats as corrupt.
MIN_IMAGE_SIDE = 10
ACCEPTED_IMAGE_FORMATS = {"JPEG", "MPO", "PNG", "BMP", "WEBP", "TIFF"}
JPEG_END_MARKER = b"\xff\xd9"

ERROR = "error"
WARNING = "warning"


@dataclass
class Issue:
    severity: str
    code: str
    image: str
    message: str
    line: int | None = None  # 1-based line in the label file


@dataclass
class Box:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    line: int


@dataclass
class ImageReport:
    image: str
    width: int | None = None
    height: int | None = None
    mode: str | None = None
    label_file: str | None = None
    boxes: list[Box] = field(default_factory=list)


def inspect_image(image_path: Path, issues: list[Issue]) -> ImageReport:
    report = ImageReport(image=image_path.name)
    try:
        with Image.open(image_path) as image:
            image.verify()  # structure check; the image object is unusable afterwards
        with Image.open(image_path) as image:
            image.load()  # decode pixel data: catches truncated files verify() misses
            report.width, report.height = image.size
            report.mode = image.mode
            image_format = image.format
            orientation = image.getexif().get(0x0112, 1)
    # PIL reports broken files as OSError, SyntaxError (bad PNG chunks) or ValueError.
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as error:
        issues.append(
            Issue(ERROR, "image_unreadable", image_path.name, f"{type(error).__name__}: {error}")
        )
        return report

    if image_format not in ACCEPTED_IMAGE_FORMATS:
        issues.append(
            Issue(ERROR, "image_format", image_path.name, f"format {image_format} is not accepted")
        )
    if min(report.width, report.height) < MIN_IMAGE_SIDE:
        issues.append(
            Issue(
                ERROR,
                "image_too_small",
                image_path.name,
                f"{report.width}x{report.height}, min side {MIN_IMAGE_SIDE}",
            )
        )
    # PIL decodes a JPEG without its end marker; Ultralytics "restores" it by rewriting the file.
    if image_format in ("JPEG", "MPO") and not image_path.read_bytes().endswith(JPEG_END_MARKER):
        issues.append(
            Issue(ERROR, "jpeg_no_end_marker", image_path.name, "JPEG does not end with FFD9")
        )
    if orientation != 1:
        # Ultralytics swaps width/height for this image, our pixel checks would not.
        issues.append(
            Issue(
                ERROR,
                "exif_rotation",
                image_path.name,
                f"EXIF orientation {orientation}: boxes may refer to the rotated image",
            )
        )
    if report.mode not in ("RGB", "L"):
        issues.append(
            Issue(WARNING, "image_mode", image_path.name, f"mode {report.mode}, convert to RGB")
        )
    return report


def parse_label_file(
    label_path: Path, image_name: str, num_classes: int, issues: list[Issue]
) -> list[Box]:
    raw = label_path.read_bytes()
    if b"\r\n" in raw:
        issues.append(Issue(WARNING, "crlf_line_endings", image_name, "CRLF line endings"))
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        issues.append(Issue(ERROR, "label_not_utf8", image_name, str(error)))
        return []
    return parse_label_text(text, image_name, num_classes, issues)


def parse_label_text(
    text: str, image_name: str, num_classes: int, issues: list[Issue]
) -> list[Box]:
    boxes = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        fields = line.split()
        if not fields:
            issues.append(Issue(WARNING, "empty_line", image_name, "empty line", line_number))
            continue
        if len(fields) != 5:
            issues.append(
                Issue(
                    ERROR,
                    "wrong_field_count",
                    image_name,
                    f"{len(fields)} fields: {line!r}",
                    line_number,
                )
            )
            continue
        try:
            class_value = float(fields[0])
            x_center, y_center, width, height = (float(value) for value in fields[1:])
        except ValueError:
            issues.append(Issue(ERROR, "not_a_number", image_name, repr(line), line_number))
            continue
        if not all(
            math.isfinite(value) for value in (class_value, x_center, y_center, width, height)
        ):
            issues.append(Issue(ERROR, "not_a_number", image_name, repr(line), line_number))
            continue
        if not class_value.is_integer():
            issues.append(
                Issue(ERROR, "class_not_integer", image_name, f"class {fields[0]}", line_number)
            )
            continue
        class_id = int(class_value)
        if not 0 <= class_id < num_classes:
            issues.append(
                Issue(
                    ERROR,
                    "unknown_class",
                    image_name,
                    f"class {class_id}, valid 0..{num_classes - 1}",
                    line_number,
                )
            )
            continue
        boxes.append(Box(class_id, x_center, y_center, width, height, line_number))
    return boxes


def check_box_geometry(
    box: Box, report: ImageReport, issues: list[Issue], strict: bool = False
) -> bool:
    """Report geometry problems. False when the box is not a usable box at all.

    A box crossing the border is still a real object (fixable), so it stays usable.
    """
    image_name = report.image
    if box.width <= 0 or box.height <= 0:
        issues.append(
            Issue(ERROR, "non_positive_size", image_name, f"w={box.width} h={box.height}", box.line)
        )
        return False
    if not all(0 <= value <= 1 for value in (box.x_center, box.y_center, box.width, box.height)):
        issues.append(
            Issue(
                ERROR,
                "value_out_of_range",
                image_name,
                f"values must be in [0,1]: {box.x_center} {box.y_center} {box.width} {box.height}",
                box.line,
            )
        )
        return False

    x1, y1, x2, y2 = yolo_to_corners(box.x_center, box.y_center, box.width, box.height)
    overflow_x = max(-x1, x2 - 1, 0.0)
    overflow_y = max(-y1, y2 - 1, 0.0)
    overflow = max(overflow_x, overflow_y)
    if report.width and report.height:
        overflow_px = max(overflow_x * report.width, overflow_y * report.height)
        is_rounding = overflow_px <= ROUNDING_TOLERANCE_PIXELS
        size_text = f", {overflow_px:.1f} px"
    else:
        is_rounding = overflow <= ROUNDING_TOLERANCE_NORMALIZED
        size_text = ""
    if not is_rounding:
        severity = ERROR if strict or overflow > OUT_OF_BOUNDS_TOLERANCE else WARNING
        issues.append(
            Issue(
                severity,
                "box_out_of_bounds",
                image_name,
                f"corners x1={x1:.4f} y1={y1:.4f} x2={x2:.4f} y2={y2:.4f}, "
                f"overflow {overflow:.4f}{size_text}",
                box.line,
            )
        )

    if report.width and report.height:
        width_px, height_px = box.width * report.width, box.height * report.height
        if width_px < TINY_BOX_PIXELS or height_px < TINY_BOX_PIXELS:
            issues.append(
                Issue(
                    WARNING, "tiny_box", image_name, f"{width_px:.1f}x{height_px:.1f} px", box.line
                )
            )
    return True


def box_iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = yolo_to_corners(a.x_center, a.y_center, a.width, a.height)
    bx1, by1, bx2, by2 = yolo_to_corners(b.x_center, b.y_center, b.width, b.height)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = a.width * a.height + b.width * b.height - intersection
    return intersection / union if union > 0 else 0.0


def find_duplicate_boxes(report: ImageReport, issues: list[Issue]) -> None:
    for index, first in enumerate(report.boxes):
        for second in report.boxes[index + 1 :]:
            if first.class_id != second.class_id:
                continue
            iou = box_iou(first, second)
            if iou >= NEAR_DUPLICATE_IOU:
                issues.append(
                    Issue(
                        WARNING,
                        "duplicate_box",
                        report.image,
                        f"same class as line {first.line}, IoU {iou:.2f}",
                        second.line,
                    )
                )


def find_unsupported_files(
    directory: Path, accepted_suffixes: set[str], issues: list[Issue]
) -> None:
    """Every entry the validator does not check is reported, never skipped silently.

    Dotfiles (`.gitkeep`) are ignored. Suffixes are compared case-sensitively for labels,
    because Ultralytics looks for exactly `<stem>.txt`.
    """
    for path in sorted(directory.iterdir()):
        if path.name.startswith("."):
            continue
        if path.is_dir():
            issues.append(
                Issue(ERROR, "unsupported_file", path.name, f"subdirectory in {directory.name}/")
            )
        elif path.suffix not in accepted_suffixes:
            issues.append(
                Issue(
                    ERROR, "unsupported_file", path.name, f"not checked: file in {directory.name}/"
                )
            )


def validate_dataset(
    images_dir: Path,
    labels_dir: Path,
    classes_file: Path,
    strict: bool = False,
    names_file: Path | None = None,
) -> tuple[list[ImageReport], list[Issue], list[str]]:
    """strict: final-label acceptance — a missing label file and any box crossing the image
    border (beyond rounding) are errors.
    names_file: class names of a CVAT export, compared to classes_file.
    """
    class_names = read_class_names(classes_file)
    issues: list[Issue] = []
    reports: list[ImageReport] = []

    if names_file is not None:
        export_names = read_names_file(names_file)
        if export_names != class_names:
            issues.append(
                Issue(
                    ERROR,
                    "class_names_mismatch",
                    names_file.name,
                    f"{export_names} != {class_names} ({classes_file.name})",
                )
            )

    image_suffixes = IMAGE_EXTENSIONS | {suffix.upper() for suffix in IMAGE_EXTENSIONS}
    find_unsupported_files(images_dir, image_suffixes, issues)
    find_unsupported_files(labels_dir, {".txt"}, issues)

    images = list_images(images_dir)
    stems = Counter(image.stem for image in images)
    label_files = {path.stem: path for path in labels_dir.glob("*.txt") if path.is_file()}

    for image_path in images:
        if stems[image_path.stem] > 1:
            issues.append(
                Issue(
                    ERROR, "ambiguous_stem", image_path.name, "several images share this name stem"
                )
            )
        report = inspect_image(image_path, issues)
        label_path = label_files.get(image_path.stem)
        if label_path is None:
            issues.append(
                Issue(
                    ERROR if strict else WARNING,
                    "label_missing",
                    image_path.name,
                    "no label file: needs annotation, not treated as background",
                )
            )
        else:
            report.label_file = label_path.name
            parsed = parse_label_file(label_path, image_path.name, len(class_names), issues)
            if not parsed:
                issues.append(
                    Issue(WARNING, "label_empty", image_path.name, "label file has no boxes")
                )
            # Only usable boxes go into the statistics; the rest are already reported.
            report.boxes = [
                box for box in parsed if check_box_geometry(box, report, issues, strict)
            ]
            find_duplicate_boxes(report, issues)
        reports.append(report)

    image_stems = set(stems)
    for stem, label_path in sorted(label_files.items()):
        if stem not in image_stems:
            issues.append(Issue(ERROR, "label_without_image", label_path.name, "no matching image"))

    return reports, issues, class_names


def summarize(reports: list[ImageReport], issues: list[Issue], class_names: list[str]) -> dict:
    boxes_per_class = Counter()
    images_per_class = Counter()
    for report in reports:
        classes_in_image = {box.class_id for box in report.boxes}
        boxes_per_class.update(box.class_id for box in report.boxes)
        images_per_class.update(classes_in_image)
    return {
        "images": len(reports),
        "images_with_labels": sum(report.label_file is not None for report in reports),
        "boxes": sum(len(report.boxes) for report in reports),
        "classes": [
            {
                "id": class_id,
                "name": name,
                "boxes": boxes_per_class[class_id],
                "images": images_per_class[class_id],
            }
            for class_id, name in enumerate(class_names)
        ],
        "errors": sum(issue.severity == ERROR for issue in issues),
        "warnings": sum(issue.severity == WARNING for issue in issues),
        "issues_by_code": dict(sorted(Counter(issue.code for issue in issues).items())),
    }


def format_text_report(summary: dict, issues: list[Issue]) -> str:
    lines = [
        (
            f"images: {summary['images']}, with labels: {summary['images_with_labels']}, "
            f"boxes: {summary['boxes']}"
        ),
        f"errors: {summary['errors']}, warnings: {summary['warnings']}",
        "",
        f"{'id':>3} {'class':<12} {'boxes':>6} {'images':>7}",
    ]
    for row in summary["classes"]:
        lines.append(f"{row['id']:>3} {row['name']:<12} {row['boxes']:>6} {row['images']:>7}")
    lines += ["", "issues by code:"]
    lines += [f"  {code}: {count}" for code, count in summary["issues_by_code"].items()]
    lines += ["", "issues:"]
    for issue in sorted(
        issues, key=lambda item: (item.severity != ERROR, item.image, item.line or 0)
    ):
        where = f"{issue.image}:{issue.line}" if issue.line else issue.image
        lines.append(f"  [{issue.severity}] {issue.code} {where} — {issue.message}")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="dataset dir with images/ and classes.txt"
    )
    parser.add_argument("--labels", type=Path, help="labels dir (default: <input>/labels)")
    parser.add_argument("--classes", type=Path, help="classes file (default: <input>/classes.txt)")
    parser.add_argument("--json", type=Path, help="also write the full result as JSON")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="final labels: missing label file and any border overflow are errors",
    )
    parser.add_argument(
        "--names",
        type=Path,
        help="class names of a CVAT export (data.yaml or obj.names) to compare",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    images_dir = args.input / "images"
    labels_dir = args.labels or args.input / "labels"
    classes_file = args.classes or args.input / "classes.txt"
    for directory in (images_dir, labels_dir):
        if not directory.is_dir():
            print(f"directory not found: {directory}", file=sys.stderr)
            return 2
    for path in (classes_file, args.names):
        if path is not None and not path.is_file():
            print(f"file not found: {path}", file=sys.stderr)
            return 2

    try:
        reports, issues, class_names = validate_dataset(
            images_dir, labels_dir, classes_file, strict=args.strict, names_file=args.names
        )
    except (ValueError, KeyError, TypeError) as error:  # bad classes.txt / names file
        print(f"error: {error}", file=sys.stderr)
        return 2
    summary = summarize(reports, issues, class_names)
    print(format_text_report(summary, issues))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": summary,
            "issues": [asdict(issue) for issue in issues],
            "images": [asdict(report) for report in reports],
        }
        args.json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
