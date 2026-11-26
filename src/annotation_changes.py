"""Build annotations/changes.csv: every difference between the source labels and annotations/final.

The source labels are compared after the confirmed class fixes (annotations/class_fixes.csv);
those fixes get their own rows with the reason from that file. Boxes are paired one-to-one by IoU
(same matching as the pre-labeling review):
    class_fix   listed in class_fixes.csv
    unchanged   same class, same coordinates (not written)
    adjusted    same class, IoU >= 0.5, coordinates differ
    reclassified IoU >= 0.5, another class
    moved       0.3 <= IoU < 0.5, or a far box of the same class and width/height
    removed     source box with no partner in final
    added       final box with no partner in the source (or the image had no label file)

Usage:
    python -m src.annotation_changes --input test_data/traffic_v5_mini --final annotations/final \
        --output annotations/changes.csv
"""

import argparse
import csv
import sys
from pathlib import Path

from src.prelabel import (
    AMBIGUOUS,
    CLASS_DISAGREEMENT,
    OVERLAPPING,
    apply_class_fixes,
    match_predictions,
    read_class_fixes,
)
from src.utils import list_images, read_class_names
from src.validate_labels import Box, parse_label_text

SAME_COORDINATES = 1e-4  # CVAT rewrites values with 6 decimals
SAME_SIZE = 1e-3  # a dragged box keeps its width or height
REASONS = {
    "adjusted": "box fitted to the object in manual CVAT review",
    "reclassified": "class corrected in manual CVAT review",
    "moved": "box shifted off its object in the source, moved onto it in manual CVAT review",
    "removed": "no matching object in the image: false or duplicate box, removed in manual CVAT review",
    "added": "object missing from the source labels, added in manual CVAT review",
    "added_no_label_file": "image had no label file: COCO pre-label reviewed and completed in CVAT",
}


def format_box(box: Box, names: list[str]) -> str:
    coordinates = " ".join(
        f"{value:.6f}" for value in (box.x_center, box.y_center, box.width, box.height)
    )
    return f"{names[box.class_id]} {coordinates}"


def same_coordinates(first: Box, second: Box) -> bool:
    return all(
        abs(a - b) < SAME_COORDINATES
        for a, b in zip(
            (first.x_center, first.y_center, first.width, first.height),
            (second.x_center, second.y_center, second.width, second.height),
            strict=True,
        )
    )


def diff_boxes(source: list[Box], final: list[Box]) -> list[tuple[str, Box | None, Box | None]]:
    """(change, source box, final box) for every box that differs."""
    changes = []
    partnered: set[int] = set()
    for final_box, category, source_box, _ in match_predictions(final, source):
        if source_box is None:
            changes.append(("added", None, final_box))
            continue
        partnered.add(id(source_box))
        if category == OVERLAPPING:
            if not same_coordinates(source_box, final_box):
                changes.append(("adjusted", source_box, final_box))
        elif category == CLASS_DISAGREEMENT:
            changes.append(("reclassified", source_box, final_box))
        elif category == AMBIGUOUS:
            changes.append(("moved", source_box, final_box))
    removed = [box for box in source if id(box) not in partnered]
    added = [final_box for change, _, final_box in changes if change == "added"]
    changes = [change for change in changes if change[0] != "added"]
    # A box dragged far from its old place keeps its class and width or height: that is a move,
    # not a removal plus an addition. Pair such boxes greedily by nearest centre.
    candidates = sorted(
        (
            (old.x_center - new.x_center) ** 2 + (old.y_center - new.y_center) ** 2,
            old_index,
            new_index,
        )
        for old_index, old in enumerate(removed)
        for new_index, new in enumerate(added)
        if old.class_id == new.class_id
        and (abs(old.width - new.width) < SAME_SIZE or abs(old.height - new.height) < SAME_SIZE)
    )
    used_old: set[int] = set()
    used_new: set[int] = set()
    for _, old_index, new_index in candidates:
        if old_index in used_old or new_index in used_new:
            continue
        used_old.add(old_index)
        used_new.add(new_index)
        changes.append(("moved", removed[old_index], added[new_index]))
    changes += [("removed", box, None) for i, box in enumerate(removed) if i not in used_old]
    changes += [("added", None, box) for i, box in enumerate(added) if i not in used_new]
    return changes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--input", type=Path, required=True, help="source dataset dir")
    parser.add_argument("--final", type=Path, default=Path("annotations/final"))
    parser.add_argument("--class-fixes", type=Path, default=Path("annotations/class_fixes.csv"))
    parser.add_argument("--output", type=Path, default=Path("annotations/changes.csv"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    names = read_class_names(args.input / "classes.txt")
    fixes = read_class_fixes(args.class_fixes)
    fix_reasons = {}
    with open(args.class_fixes, encoding="utf-8") as file:
        for row in csv.DictReader(file):
            fix_reasons[(row["image"], int(row["line"]))] = row["reason"]

    rows = []
    for image_path in list_images(args.input / "images"):
        final_path = args.final / f"{image_path.stem}.txt"
        if not final_path.is_file():
            print(f"error: no final labels for {image_path.name}", file=sys.stderr)
            return 1
        final = parse_label_text(
            final_path.read_text(encoding="utf-8"), image_path.name, len(names), []
        )
        source_path = args.input / "labels" / f"{image_path.stem}.txt"
        if not source_path.is_file():
            rows += [
                [
                    image_path.name,
                    "added",
                    "",
                    "",
                    format_box(box, names),
                    REASONS["added_no_label_file"],
                ]
                for box in final
            ]
            continue

        raw_text = source_path.read_text(encoding="utf-8")
        raw_lines = raw_text.splitlines()
        for (fix_image, line), (old_class, new_class) in sorted(fixes.items()):
            if fix_image == image_path.name:
                coordinates = " ".join(raw_lines[line - 1].split()[1:])
                rows.append(
                    [
                        image_path.name,
                        "class_fix",
                        line,
                        f"{names[old_class] if old_class < len(names) else old_class} {coordinates}",
                        f"{names[new_class]} {coordinates}",
                        fix_reasons[(fix_image, line)],
                    ]
                )
        source = parse_label_text(
            apply_class_fixes(raw_text, image_path.name, fixes), image_path.name, len(names), []
        )
        for change, source_box, final_box in diff_boxes(source, final):
            rows.append(
                [
                    image_path.name,
                    change,
                    source_box.line if source_box else "",
                    format_box(source_box, names) if source_box else "",
                    format_box(final_box, names) if final_box else "",
                    REASONS[change],
                ]
            )

    with open(args.output, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["image", "change", "source_line", "before", "after", "reason"])
        writer.writerows(rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row[1]] = counts.get(row[1], 0) + 1
    print(f"{args.output}: {len(rows)} changes")
    for change, count in sorted(counts.items()):
        print(f"  {change}: {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
