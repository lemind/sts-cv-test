"""Pre-label images with a COCO-pretrained YOLO and build a CVAT import.

Output (gitignored, under --output):
    import/data.yaml, import/labels/train/<stem>.txt   -> zipped to cvat_import.zip
        existing labels as they are (minus unknown classes), predictions for images without labels
    review/<stem>.jpg            original boxes green, predictions not matching any box red
    missed_candidates.csv        those red predictions: likely objects missing from the labels
    dropped_lines.csv            original lines CVAT cannot import (unknown class), to redo by hand

Predictions are a starting point for manual review in CVAT, never final labels.

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
from src.validate_labels import Box, box_iou, parse_label_file

# COCO id -> our class name. police/ambulance/fire_truck/lorry/trailer have no COCO class.
COCO_TO_OURS = {0: "pedestrian", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
# A prediction overlapping an existing box this much (any class) is "already labeled".
MATCH_IOU = 0.3


def predict_boxes(
    model, image_path: Path, class_ids: dict[str, int], conf: float, imgsz: int
) -> list[Box]:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")  # RGBA screenshots
    result = model.predict(rgb, conf=conf, imgsz=imgsz, verbose=False)[0]
    boxes = []
    for coco_id, xywhn in zip(result.boxes.cls.tolist(), result.boxes.xywhn.tolist(), strict=True):
        name = COCO_TO_OURS.get(int(coco_id))
        if name is not None:
            boxes.append(Box(class_ids[name], *xywhn, line=0))
    return boxes


def format_label_lines(boxes: list[Box]) -> str:
    return "".join(
        f"{box.class_id} {box.x_center:.6f} {box.y_center:.6f} {box.width:.6f} {box.height:.6f}\n"
        for box in boxes
    )


def draw_review_image(
    image_path: Path, labeled: list[Box], missed: list[Box], names: list[str], out: Path
) -> None:
    with Image.open(image_path) as image:
        canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    line_width = max(2, width // 640)
    for boxes, color in ((labeled, (0, 220, 0)), (missed, (255, 0, 0))):
        for box in boxes:
            x1, y1, x2, y2 = yolo_to_corners(box.x_center, box.y_center, box.width, box.height)
            corners = (x1 * width, y1 * height, x2 * width, y2 * height)
            draw.rectangle(corners, outline=color, width=line_width)
            draw.text((corners[0] + 2, corners[1] + 2), names[box.class_id], fill=color)
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
    if args.output.exists():
        shutil.rmtree(args.output / "import", ignore_errors=True)
        shutil.rmtree(args.output / "review", ignore_errors=True)
    labels_out = args.output / "import" / "labels" / "train"
    review_out = args.output / "review"
    labels_out.mkdir(parents=True)
    review_out.mkdir(parents=True)

    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)

    missed_rows, dropped_rows = [], []
    for image_path in list_images(args.input / "images"):
        label_path = args.input / "labels" / f"{image_path.stem}.txt"
        predicted = predict_boxes(model, image_path, class_ids, args.conf, args.imgsz)

        if label_path.is_file():
            issues = []
            labeled = parse_label_file(label_path, image_path.name, len(names), issues)
            dropped_rows += [
                [image_path.name, issue.line, issue.message]
                for issue in issues
                if issue.code == "unknown_class"
            ]
            missed = [
                box
                for box in predicted
                if all(box_iou(box, other) < MATCH_IOU for other in labeled)
            ]
            import_boxes = labeled
        else:
            labeled, missed, import_boxes = [], predicted, predicted

        (labels_out / f"{image_path.stem}.txt").write_text(
            format_label_lines(import_boxes), encoding="utf-8"
        )
        missed_rows += [
            [
                image_path.name,
                names[box.class_id],
                f"{box.x_center:.4f}",
                f"{box.y_center:.4f}",
                f"{box.width:.4f}",
                f"{box.height:.4f}",
                "no_label_file" if not label_path.is_file() else "unmatched",
            ]
            for box in missed
        ]
        draw_review_image(image_path, labeled, missed, names, review_out / f"{image_path.stem}.jpg")
        print(
            f"{image_path.name}: labeled {len(labeled)}, predicted {len(predicted)}, red {len(missed)}"
        )

    data_yaml = {"path": ".", "train": "images/train", "names": dict(enumerate(names))}
    (args.output / "import" / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    with open(args.output / "missed_candidates.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "class", "x_center", "y_center", "width", "height", "reason"])
        writer.writerows(missed_rows)
    with open(args.output / "dropped_lines.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "line", "reason"])
        writer.writerows(dropped_rows)
    archive = shutil.make_archive(str(args.output / "cvat_import"), "zip", args.output / "import")
    print(f"\nCVAT import: {archive}\nreview images: {review_out}\ncandidates: {len(missed_rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
