"""Pre-label images with a COCO-pretrained YOLO and build a CVAT import.

Output (gitignored, under --output):
    import/data.yaml, import/labels/train/<stem>.txt, empty import/images/train/
                                                       -> zipped to cvat_import.zip
        original labels with confirmed class fixes applied (--class-fixes),
        predictions only for images without a label file
    review/<stem>.jpg          original boxes + predictions coloured by category (legend on image)
    prediction_review.csv      every prediction with its category against the original labels
    dropped_lines.csv          original lines still not importable (unknown class with no fix)

Predictions are triage hints for the manual pass in CVAT, never final labels. The matching is
geometric and one-to-one; "overlapping" means only that, not that the object is labeled correctly.

Usage:
    python -m src.prelabel --input test_data/traffic_v5_mini --output cvat
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

from src.utils import list_images, read_class_names, yolo_to_corners
from src.validate_labels import Box, box_iou, parse_label_text

# COCO id -> our class name. police/ambulance/fire_truck/lorry/trailer have no COCO class.
COCO_TO_OURS = {0: "pedestrian", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
OVERLAP_IOU = 0.5
AMBIGUOUS_IOU = 0.3

# category -> review colour; "overlapping" predictions are not drawn.
UNMATCHED = "unmatched"  # IoU < 0.3 with every free original box: possibly missing label
AMBIGUOUS = "ambiguous_overlap"  # 0.3 <= IoU < 0.5: loose box or a different object
CLASS_DISAGREEMENT = "class_disagreement"  # IoU >= 0.5 but another class: possible class error
OVERLAPPING = "overlapping_existing_annotation"  # IoU >= 0.5, same class
NO_LABEL_FILE = "no_label_file"
COLORS = {
    "original": (0, 220, 0),
    UNMATCHED: (255, 0, 0),
    NO_LABEL_FILE: (255, 0, 0),
    AMBIGUOUS: (255, 160, 0),
    CLASS_DISAGREEMENT: (255, 0, 255),
}


def predict_boxes(
    model, image_path: Path, class_ids: dict[str, int], conf: float, imgsz: int
) -> list[Box]:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")  # RGBA screenshots, alpha is fully opaque
    result = model.predict(rgb, conf=conf, imgsz=imgsz, verbose=False)[0]
    boxes = []
    for coco_id, xywhn in zip(result.boxes.cls.tolist(), result.boxes.xywhn.tolist(), strict=True):
        name = COCO_TO_OURS.get(int(coco_id))
        if name is not None:
            boxes.append(Box(class_ids[name], *xywhn, line=0))
    return boxes


def match_predictions(
    predicted: list[Box], labeled: list[Box]
) -> list[tuple[Box, str, Box | None, float]]:
    """Greedy one-to-one matching by descending IoU, class-agnostic.

    Returns (prediction, category, matched original box or None, IoU) per prediction.
    """
    pairs = sorted(
        (
            (box_iou(prediction, original), p_index, o_index)
            for p_index, prediction in enumerate(predicted)
            for o_index, original in enumerate(labeled)
        ),
        reverse=True,
    )
    partner: dict[int, tuple[int, float]] = {}
    used_originals: set[int] = set()
    for iou, p_index, o_index in pairs:
        if iou < AMBIGUOUS_IOU:
            break
        if p_index in partner or o_index in used_originals:
            continue
        partner[p_index] = (o_index, iou)
        used_originals.add(o_index)

    matches = []
    for p_index, prediction in enumerate(predicted):
        if p_index not in partner:
            matches.append((prediction, UNMATCHED, None, 0.0))
            continue
        o_index, iou = partner[p_index]
        original = labeled[o_index]
        if iou < OVERLAP_IOU:
            category = AMBIGUOUS
        elif original.class_id != prediction.class_id:
            category = CLASS_DISAGREEMENT
        else:
            category = OVERLAPPING
        matches.append((prediction, category, original, iou))
    return matches


def read_class_fixes(fixes_file: Path | None) -> dict[tuple[str, int], tuple[int, int]]:
    """(image, 1-based line) -> (old_class, new_class), visually confirmed fixes."""
    if fixes_file is None:
        return {}
    with open(fixes_file, encoding="utf-8") as file:
        return {
            (row["image"], int(row["line"])): (int(row["old_class"]), int(row["new_class"]))
            for row in csv.DictReader(file)
        }


def apply_class_fixes(
    text: str, image_name: str, fixes: dict[tuple[str, int], tuple[int, int]]
) -> str:
    lines = text.splitlines()
    for (fix_image, line_number), (old_class, new_class) in fixes.items():
        if fix_image != image_name:
            continue
        fields = lines[line_number - 1].split()
        if int(float(fields[0])) != old_class:
            raise ValueError(
                f"{image_name}:{line_number}: expected class {old_class}, got {fields[0]}"
            )
        lines[line_number - 1] = " ".join([str(new_class), *fields[1:]])
    return "\n".join(lines) + "\n"


def format_label_lines(boxes: list[Box]) -> str:
    return "".join(
        f"{box.class_id} {box.x_center:.6f} {box.y_center:.6f} {box.width:.6f} {box.height:.6f}\n"
        for box in boxes
    )


def draw_review_image(
    image_path: Path,
    labeled: list[Box],
    flagged: list[tuple[Box, str]],
    names: list[str],
    out: Path,
) -> None:
    with Image.open(image_path) as image:
        canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    line_width = max(2, width // 640)
    to_draw = [(box, "original") for box in labeled] + flagged
    for box, category in to_draw:
        color = COLORS[category]
        x1, y1, x2, y2 = yolo_to_corners(box.x_center, box.y_center, box.width, box.height)
        corners = (x1 * width, y1 * height, x2 * width, y2 * height)
        draw.rectangle(corners, outline=color, width=line_width)
        draw.text((corners[0] + 2, corners[1] + 2), names[box.class_id], fill=color)
    legend = ["original label", "unmatched prediction", "ambiguous overlap", "class disagreement"]
    legend_colors = [
        COLORS["original"],
        COLORS[UNMATCHED],
        COLORS[AMBIGUOUS],
        COLORS[CLASS_DISAGREEMENT],
    ]
    draw.rectangle((0, 0, 190, 16 * len(legend) + 6), fill=(0, 0, 0))
    for index, (text, color) in enumerate(zip(legend, legend_colors, strict=True)):
        draw.text((6, 4 + 16 * index), text, fill=color)
    canvas.save(out, quality=90)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="dataset dir with images/, labels/, classes.txt"
    )
    parser.add_argument("--output", type=Path, default=Path("cvat"))
    parser.add_argument(
        "--class-fixes",
        type=Path,
        default=Path("annotations/class_fixes.csv"),
        help="visually confirmed class corrections applied before import",
    )
    parser.add_argument(
        "--model", default="weights/yolo11x.pt", help="COCO-pretrained YOLO weights"
    )
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument(
        "--imgsz", type=int, default=1280, help="large: CCTV frames have small cars"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from ultralytics import YOLO  # heavy import, only when actually predicting

    names = read_class_names(args.input / "classes.txt")
    class_ids = {name: index for index, name in enumerate(names)}
    fixes = read_class_fixes(args.class_fixes if args.class_fixes.is_file() else None)
    if args.output.exists():
        shutil.rmtree(args.output / "import", ignore_errors=True)
        shutil.rmtree(args.output / "review", ignore_errors=True)
    labels_out = args.output / "import" / "labels" / "train"
    review_out = args.output / "review"
    labels_out.mkdir(parents=True)
    # CVAT's importer requires images/train to exist; left empty, it matches labels to task frames by name.
    (args.output / "import" / "images" / "train").mkdir(parents=True)
    review_out.mkdir(parents=True)

    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)

    review_rows, dropped_rows = [], []
    for image_path in list_images(args.input / "images"):
        label_path = args.input / "labels" / f"{image_path.stem}.txt"
        predicted = predict_boxes(model, image_path, class_ids, args.conf, args.imgsz)

        if label_path.is_file():
            text = apply_class_fixes(label_path.read_text(encoding="utf-8"), image_path.name, fixes)
            issues = []
            labeled = parse_label_text(text, image_path.name, len(names), issues)
            dropped_rows += [
                [image_path.name, issue.line, issue.message]
                for issue in issues
                if issue.code == "unknown_class"
            ]
            matches = match_predictions(predicted, labeled)
            import_boxes = labeled
        else:
            labeled, import_boxes = [], predicted
            matches = [(box, NO_LABEL_FILE, None, 0.0) for box in predicted]

        (labels_out / f"{image_path.stem}.txt").write_text(
            format_label_lines(import_boxes), encoding="utf-8"
        )
        for prediction, category, original, iou in matches:
            review_rows.append(
                [
                    image_path.name,
                    category,
                    names[prediction.class_id],
                    names[original.class_id] if original else "",
                    original.line if original else "",
                    f"{iou:.2f}",
                    *(
                        f"{value:.4f}"
                        for value in (
                            prediction.x_center,
                            prediction.y_center,
                            prediction.width,
                            prediction.height,
                        )
                    ),
                ]
            )
        flagged = [(box, category) for box, category, _, _ in matches if category != OVERLAPPING]
        draw_review_image(
            image_path, labeled, flagged, names, review_out / f"{image_path.stem}.jpg"
        )
        counts = {
            category: sum(match[1] == category for match in matches)
            for category in COLORS
            if category != "original"
        }
        print(
            f"{image_path.name}: labeled {len(labeled)}, predicted {len(predicted)}, "
            + ", ".join(f"{k} {v}" for k, v in counts.items() if v)
        )

    data_yaml = {"path": ".", "train": "images/train", "names": dict(enumerate(names))}
    (args.output / "import" / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    with open(args.output / "prediction_review.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "image",
                "category",
                "predicted_class",
                "labeled_class",
                "labeled_line",
                "iou",
                "x_center",
                "y_center",
                "width",
                "height",
            ]
        )
        writer.writerows(review_rows)
    with open(args.output / "dropped_lines.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "line", "reason"])
        writer.writerows(dropped_rows)
    archive = shutil.make_archive(str(args.output / "cvat_import"), "zip", args.output / "import")
    print(f"\nCVAT import: {archive}\nreview images: {review_out}")
    for category in (UNMATCHED, NO_LABEL_FILE, AMBIGUOUS, CLASS_DISAGREEMENT, OVERLAPPING):
        print(f"  {category}: {sum(row[1] == category for row in review_rows)}")
    print(f"  dropped lines: {len(dropped_rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
