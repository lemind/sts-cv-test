"""Build the processed YOLO dataset from the source images and the corrected labels.

Inputs (read-only):
    <input>/images/, <input>/labels/, <input>/classes.txt   source set (labels only for the report)
    <annotations>/final/<stem>.txt               corrected labels, one file per image
    <annotations>/source_groups.csv              image -> group (+ pin_split, note), split unit
    <annotations>/changes.csv                    audit trail, summarized in the report

Outputs (rewritten on every run):
    <output>/images/{train,val,test}/, <output>/labels/{train,val,test}/, <output>/dataset.yaml
    <reports>/split_manifest.csv                 image, file, group, split, pinned
    <reports>/dataset_report.md                  statistics, fixes, limitations
    <reports>/samples/all/<stem>.jpg             every image with its boxes (gitignored)
    <reports>/samples/<name>.jpg                 a few representative ones (committed)
    <reports>/manifest.sha256                    sha256 of every dataset file, `sha256sum -c` format

Everything is read and checked before the first file is written. Paths in the manifest and report
are relative to the working directory (run from the repo root); --output must be inside it and
must not overlap the source data or the annotations.

The split is done by source group (frames of one camera/video/incident stay together), balanced by
image count, deterministic for a given --seed. Groups with pin_split always go to that split.

Usage:
    python -m src.prepare_dataset --input test_data/traffic_v5_mini --output data/processed --seed 42
    python -m src.prepare_dataset --input test_data/traffic_v5_mini --output data/processed --seed 42 \
        --plan-only        # print the split tables and stop, write nothing
"""

import argparse
import csv
import hashlib
import random
import shutil
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

from src.utils import yolo_to_corners
from src.validate_labels import ImageReport, summarize, validate_dataset

SPLITS = ("train", "val", "test")
VAL_FRACTION = 0.2
TEST_FRACTION = 0.2
# A val/test split made of one or two sources, or dominated by one, says little about
# generalization.
MIN_EVAL_GROUPS = 3
MIN_EVAL_IMAGES = 4
MAX_GROUP_SHARE = 0.5
# Modes Ultralytics reads without conversion; anything else is converted to RGB.
KEPT_MODES = {"RGB", "L"}
# Validator codes for source lines that are not counted as valid boxes.
DROPPED_LINE_CODES = (
    "unknown_class",
    "wrong_field_count",
    "not_a_number",
    "class_not_integer",
    "non_positive_size",
    "value_out_of_range",
)
FILE_CHECK_CODES = (
    "image_unreadable",
    "jpeg_no_end_marker",
    "image_format",
    "image_too_small",
    "exif_rotation",
    "duplicate_box",
    "tiny_box",
)
# Representative images for <reports>/samples/ (committed): image name -> what it shows.
SAMPLE_IMAGES = {
    "01-04-2023 183852.png": "typical_multiclass",
    "Снимок экрана от 2023-06-05 16-05-35.png": "shifted_boxes_and_crowd",
    "Снимок экрана от 2023-06-08 11-38-52.png": "shifted_boxes_2",
    "4d3c2d6s-1920.jpg": "class_fix_and_added_trailer",
    "102_9aug202317009.png": "no_source_labels",
    "Снимок экрана от 2023-06-05 14-29-47.png": "small_cctv_objects",
}
BOX_COLORS = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 212),
    (0, 128, 128),
]


def read_source_groups(groups_file: Path, image_names: list[str]) -> dict[str, dict[str, str]]:
    """image name -> {group, pin, note}. Every image is listed once, every row names a group,
    and a pin applies to the whole group."""
    with open(groups_file, encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    groups: dict[str, dict[str, str]] = {}
    for number, row in enumerate(rows, start=2):
        image, group = (row.get("image") or "").strip(), (row.get("group") or "").strip()
        pin = (row.get("pin_split") or "").strip()
        if not image or not group:
            raise ValueError(f"{groups_file}:{number}: image and group are required")
        if image in groups:
            raise ValueError(f"{groups_file}:{number}: {image} listed twice")
        if pin and pin not in SPLITS:
            raise ValueError(f"{groups_file}:{number}: unknown pin_split {pin!r}")
        groups[image] = {"group": group, "pin": pin, "note": (row.get("note") or "").strip()}
    missing = sorted(set(image_names) - set(groups))
    unknown = sorted(set(groups) - set(image_names))
    if missing or unknown:
        raise ValueError(
            f"{groups_file}: images without a group {missing}, unknown images {unknown}"
        )
    pins_per_group: dict[str, set[str]] = defaultdict(set)
    for entry in groups.values():
        pins_per_group[entry["group"]].add(entry["pin"])
    inconsistent = sorted(group for group, pins in pins_per_group.items() if len(pins) > 1)
    if inconsistent:
        raise ValueError(f"{groups_file}: frames of one group pinned differently: {inconsistent}")
    return groups


def split_groups(
    group_sizes: dict[str, int],
    group_pins: dict[str, str],
    seed: int,
    val_fraction: float = VAL_FRACTION,
    test_fraction: float = TEST_FRACTION,
) -> dict[str, str]:
    """group -> split. Pinned groups go to their split; the rest are shuffled by seed and fill
    test, then val, up to the target image count; everything left goes to train. A group larger
    than MAX_GROUP_SHARE of a target never goes to that split."""
    total = sum(group_sizes.values())
    targets = {
        "test": int(total * test_fraction + 0.5),
        "val": int(total * val_fraction + 0.5),
    }
    assignment = {group: pin for group, pin in group_pins.items() if pin}
    counts: Counter = Counter()
    for group, split in assignment.items():
        counts[split] += group_sizes[group]
    free = sorted(group for group in group_sizes if group not in assignment)
    random.Random(seed).shuffle(free)
    for group in free:
        split = "train"
        for candidate in ("test", "val"):
            # One source must not dominate an evaluation split: bigger groups go to train.
            fits = group_sizes[group] <= targets[candidate] * MAX_GROUP_SHARE
            if counts[candidate] < targets[candidate] and fits:
                split = candidate
                break
        assignment[group] = split
        counts[split] += group_sizes[group]
    return assignment


def find_split_problems(
    split_of_image: dict[str, str],
    group_of_image: dict[str, str],
    classes_of_image: dict[str, list[int]],
) -> list[str]:
    """Problems that make the split unusable (pinned groups included); empty when it is fine."""
    problems = []
    all_classes = {c for classes in classes_of_image.values() for c in classes}
    train_classes = {
        c
        for image, classes in classes_of_image.items()
        if split_of_image[image] == "train"
        for c in classes
    }
    if all_classes - train_classes:
        problems.append(f"classes with no train boxes: {sorted(all_classes - train_classes)}")
    splits_of_group: dict[str, set[str]] = defaultdict(set)
    for image, group in group_of_image.items():
        splits_of_group[group].add(split_of_image[image])
    leaked = sorted(group for group, splits in splits_of_group.items() if len(splits) > 1)
    if leaked:
        problems.append(f"groups in more than one split: {leaked}")
    for split in ("val", "test"):
        images = [image for image, s in split_of_image.items() if s == split]
        group_sizes = Counter(group_of_image[image] for image in images)
        if len(group_sizes) < MIN_EVAL_GROUPS:
            problems.append(f"{split}: {len(group_sizes)} groups, need at least {MIN_EVAL_GROUPS}")
        if len(images) < MIN_EVAL_IMAGES:
            problems.append(f"{split}: {len(images)} images, need at least {MIN_EVAL_IMAGES}")
        dominant = sorted(g for g, n in group_sizes.items() if n > len(images) * MAX_GROUP_SHARE)
        if dominant:
            problems.append(f"{split}: more than {MAX_GROUP_SHARE:.0%} from one group {dominant}")
        if not sum(len(classes_of_image[image]) for image in images):
            problems.append(f"{split}: no labeled objects")
    return problems


def needs_conversion(report: ImageReport) -> bool:
    return report.mode not in KEPT_MODES


def find_transparency(image_path: Path) -> bool:
    """True when the image has an alpha channel (or transparency) that is not fully opaque."""
    with Image.open(image_path) as image:
        if image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info:
            alpha = image.convert("RGBA").getchannel("A")
            return alpha.getextrema() != (255, 255)
    return False


def output_file_name(image_path: Path, report: ImageReport) -> str:
    return f"{image_path.stem}.png" if needs_conversion(report) else image_path.name


def write_image(image_path: Path, report: ImageReport, destination: Path) -> None:
    """Copy the file byte for byte when Ultralytics reads it as is; otherwise convert to RGB and
    save as PNG (lossless). Transparency was rejected before any write."""
    if not needs_conversion(report):
        shutil.copyfile(image_path, destination)
        return
    with Image.open(image_path) as image:
        image.convert("RGB").save(destination, format="PNG")


def write_labels(source: Path, destination: Path) -> None:
    """Labels with LF line endings, no blank lines, trailing newline; empty file stays empty."""
    lines = [
        line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    destination.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8", newline="\n")


def draw_boxes(image_path: Path, report: ImageReport, names: list[str], out: Path) -> None:
    with Image.open(image_path) as image:
        canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    line_width = max(1, width // 800)
    for box in report.boxes:
        x1, y1, x2, y2 = yolo_to_corners(box.x_center, box.y_center, box.width, box.height)
        color = BOX_COLORS[box.class_id % len(BOX_COLORS)]
        corners = (x1 * width, y1 * height, x2 * width, y2 * height)
        draw.rectangle(corners, outline=color, width=line_width)
        draw.text((corners[0] + 2, max(0, corners[1] - 11)), names[box.class_id], fill=color)
    canvas.save(out, format="JPEG", quality=90)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_to_cwd(path: Path) -> Path:
    """Path relative to the working directory; ValueError when it lies outside it."""
    return path.resolve().relative_to(Path.cwd().resolve())


def write_manifest(output: Path, manifest: Path) -> str:
    """Hash images/, labels/ and dataset.yaml under `output` in a stable order, paths relative to
    the working directory. Returns the dataset hash (sha256 of the manifest text)."""
    files = [output / "dataset.yaml"]
    for subdir in ("images", "labels"):
        # labels/<split>.cache is Ultralytics' rebuildable label cache, not dataset content.
        files += [
            path
            for path in (output / subdir).rglob("*")
            if path.is_file() and path.suffix != ".cache"
        ]
    relative_paths = sorted((relative_to_cwd(path).as_posix(), path) for path in files)
    text = "".join(f"{file_sha256(path)}  {relative}\n" for relative, path in relative_paths)
    manifest.write_text(text, encoding="utf-8", newline="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def remove_if_exists(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def check_output_location(output: Path, protected: list[Path]) -> str | None:
    """Reason why `output` may not be written, or None."""
    try:
        relative_to_cwd(output)
    except ValueError:
        return f"--output {output} must be inside the working directory (run from the repo root)"
    target = output.resolve()
    for path in protected:
        guarded = path.resolve()
        if target == guarded or guarded in target.parents or target in guarded.parents:
            return f"--output {output} overlaps {path}, which must stay read-only"
    return None


def format_split_tables(
    split_of_image: dict[str, str],
    group_of_image: dict[str, str],
    reports: dict[str, ImageReport],
    names: list[str],
) -> str:
    images_of_split = {
        split: [image for image, s in split_of_image.items() if s == split] for split in SPLITS
    }
    lines = ["| Split | Groups | Images | Boxes |", "|---|---:|---:|---:|"]
    class_stats = {}
    for split, images in images_of_split.items():
        summary = summarize([reports[image] for image in images], [], names)
        class_stats[split] = summary["classes"]
        groups = {group_of_image[image] for image in images}
        lines.append(f"| {split} | {len(groups)} | {len(images)} | {summary['boxes']} |")
    lines += [
        "",
        "| Id | Class | " + " | ".join(f"{s} boxes / images" for s in SPLITS) + " |",
        "|---:|---|" + "---:|" * len(SPLITS),
    ]
    for class_id, name in enumerate(names):
        cells = [
            f"{class_stats[s][class_id]['boxes']} / {class_stats[s][class_id]['images']}"
            for s in SPLITS
        ]
        lines.append(f"| {class_id} | {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def count_raw_label_lines(labels_dir: Path, image_stems: set[str]) -> int:
    """Non-blank lines in the source label files that belong to an image."""
    return sum(
        1
        for path in sorted(labels_dir.glob("*.txt"))
        if path.stem in image_stems
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def read_changes(changes_file: Path) -> list[dict[str, str]]:
    with open(changes_file, encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    required = {"image", "change", "before", "after"}
    if rows and not required <= set(rows[0]):
        raise ValueError(f"{changes_file}: columns {sorted(required)} are required")
    return rows


def describe_moves(changes: list[dict[str, str]]) -> str:
    """Per image: how many boxes were moved back and the median shift they had in the source."""
    shifts: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in changes:
        if row["change"] == "moved":
            before, after = row["before"].split()[1:3], row["after"].split()[1:3]
            shifts[row["image"]].append(
                (float(before[0]) - float(after[0]), float(before[1]) - float(after[1]))
            )
    return "; ".join(
        f"`{image}`: {len(moves)} boxes, source offset as a share of the image size: "
        f"median x {statistics.median(dx for dx, _ in moves):+.2f} "
        f"y {statistics.median(dy for _, dy in moves):+.2f}, "
        f"largest x {max((dx for dx, _ in moves), key=abs):+.2f} "
        f"y {max((dy for _, dy in moves), key=abs):+.2f}"
        for image, moves in sorted(shifts.items())
    )


def format_report(
    args: argparse.Namespace,
    split_tables: str,
    split_of_image: dict[str, str],
    groups: dict[str, dict[str, str]],
    reports: dict[str, ImageReport],
    source_summary: dict,
    final_summary: dict,
    raw_lines: int,
    changes: list[dict[str, str]],
    converted_modes: Counter,
    dataset_hash: str,
) -> str:
    def relative(path: Path) -> str:
        return relative_to_cwd(path).as_posix()

    class_rows = "\n".join(
        f"| {source['id']} | {source['name']} | {source['boxes']} / {source['images']} "
        f"| {final['boxes']} / {final['images']} |"
        for source, final in zip(source_summary["classes"], final_summary["classes"], strict=True)
    )
    source_codes, final_codes = source_summary["issues_by_code"], final_summary["issues_by_code"]
    dropped = {code: source_codes[code] for code in DROPPED_LINE_CODES if code in source_codes}
    change_counts = Counter(row["change"] for row in changes)
    changed_images: dict[str, set[str]] = defaultdict(set)
    for row in changes:
        changed_images[row["change"]].add(row["image"])
    file_checks = ", ".join(
        f"{code} {source_codes.get(code, 0)} / {final_codes.get(code, 0)}"
        for code in FILE_CHECK_CODES
    )
    modes = ", ".join(f"{mode} × {count}" for mode, count in sorted(converted_modes.items()))
    pinned = defaultdict(set)
    for entry in groups.values():
        if entry["pin"]:
            pinned[(entry["group"], entry["pin"])].add(entry["note"])
    pinned_lines = "\n".join(
        f"  - `{group}` → {pin}: {'; '.join(sorted(notes))}"
        for (group, pin), notes in sorted(pinned.items())
    )
    present = {
        split: {
            box.class_id
            for image, s in split_of_image.items()
            if s == split
            for box in reports[image].boxes
        }
        for split in SPLITS
    }
    names = [entry["name"] for entry in final_summary["classes"]]
    absent = "; ".join(
        f"{split}: "
        + (", ".join(names[c] for c in sorted(set(range(len(names))) - present[split])) or "none")
        for split in ("val", "test")
    )
    samples = "\n".join(f"- `samples/{name}.jpg`: {image}" for image, name in SAMPLE_IMAGES.items())
    annotations = relative(args.annotations)
    return f"""# Dataset report: {args.input.name}

Generated by `python -m src.prepare_dataset --input {relative(args.input)} \
--output {relative(args.output)} --seed {args.seed}`. Do not edit by hand.

Dataset hash (sha256 of `{relative(args.reports)}/manifest.sha256`): `{dataset_hash}`

## Summary

| | Source | Final |
|---|---:|---:|
| Images | {source_summary["images"]} | {final_summary["images"]} |
| Images with a label file | {source_summary["images_with_labels"]} | \
{final_summary["images_with_labels"]} |
| Non-blank label lines | {raw_lines} | {final_summary["boxes"]} |
| Valid boxes | {source_summary["boxes"]} | {final_summary["boxes"]} |
| Validator errors / warnings | {source_summary["errors"]} / {source_summary["warnings"]} \
| {final_summary["errors"]} / {final_summary["warnings"]} |

Source lines not counted as valid boxes: {raw_lines - source_summary["boxes"]} \
({", ".join(f"{code} {count}" for code, count in dropped.items()) or "none"}).
Final = `{annotations}/final/`, validated with `--strict`; its remaining warnings are about the
images themselves (see the table below), not the labels.

## Boxes / images per class

| Id | Class | Source | Final |
|---:|---|---:|---:|
{class_rows}

## Split

By source group (`{annotations}/source_groups.csv`): frames of one camera, video or incident stay
in one split. Balanced by image count (target val {VAL_FRACTION:.0%}, test {TEST_FRACTION:.0%}; a
group is added whole, so the result can overshoot), deterministic for `--seed`. Stops when a class
has no train boxes, a group is in two splits, or val/test has fewer than {MIN_EVAL_GROUPS} groups,
fewer than {MIN_EVAL_IMAGES} images, no objects, or more than {MAX_GROUP_SHARE:.0%} of its images
from one group. Per-image assignment: `{relative(args.reports)}/split_manifest.csv`.

{split_tables}

Pinned groups:
{pinned_lines or "  - none"}

## Problems found and what was done

Audit trail, one row per changed box with the reason: `{annotations}/changes.csv`. Class rules
used for every decision: `{annotations}/README.md`.

| Problem | Found (source) | Decision |
|---|---|---|
| Unknown class id | {source_codes.get("unknown_class", 0)} lines | Checked on the image, \
corrected in `{annotations}/class_fixes.csv` |
| Wrong class | {change_counts["class_fix"]} fixed before CVAT (incl. unknown ids), \
{change_counts["reclassified"]} in CVAT | Per the class rules |
| Boxes shifted off their objects | {change_counts["moved"]} boxes in \
{len(changed_images["moved"])} images | Moved back by hand in CVAT. {describe_moves(changes)} |
| Boxes out of the image | {source_codes.get("box_out_of_bounds", 0)} boxes | Checked on the \
image, never clipped automatically; fixed in CVAT (moved or adjusted) |
| Missing label files | {source_codes.get("label_missing", 0)} images | COCO-pretrained YOLO \
pre-labels, reviewed and completed by hand in CVAT |
| Missing objects | {change_counts["added"]} boxes added in {len(changed_images["added"])} images \
(incl. images without a label file) | Added in CVAT |
| Loose or false boxes | {change_counts["adjusted"]} adjusted, {change_counts["removed"]} removed \
| Fixed in CVAT |
| Images not in RGB/L mode | {modes or "none"} | Alpha checked fully opaque, converted to RGB PNG. \
Other images are copied byte for byte |
| Label files with CRLF line endings | {source_codes.get("crlf_line_endings", 0)} | All labels written with \
LF |
| Other file and box checks (source / final) | {file_checks} | Checked by the validator |

## Limitations

- {final_summary["images"]} images from {len({e["group"] for e in groups.values()})} source \
groups: val/test metrics are a sanity check, not an estimate of generalization.
- Classes absent from a split are not evaluated on it (no AP), which is not the same as AP = 0.
  Absent — {absent}.
- Pinned groups (see above) are kept out of the random split on purpose.
- Dense crowds: every person who can be told apart is labeled, people merged into one blob are not
  (YOLO has no "ignore" regions); see `{annotations}/README.md`.
- Pre-labels came from a COCO model, which has no police/ambulance/fire_truck/lorry/trailer classes;
  those were labeled by hand.

## Samples

Every image with its final boxes: `{relative(args.reports)}/samples/all/` (generated, not
committed). Committed:

{samples}
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--input", type=Path, required=True, help="source dataset dir")
    parser.add_argument("--output", type=Path, required=True, help="processed dataset dir")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--annotations", type=Path, default=Path("annotations"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument(
        "--plan-only", action="store_true", help="print the split tables and stop, write nothing"
    )
    return parser.parse_args(argv)


def fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    images_dir = args.input / "images"
    classes_file = args.input / "classes.txt"
    final_dir = args.annotations / "final"
    for path in (args.reports, args.output):
        try:
            relative_to_cwd(path)
        except ValueError:
            return fail(f"{path} must be inside the working directory (run from the repo root)")
    reason = check_output_location(args.output, [args.input, args.annotations, args.reports])
    if reason:
        return fail(reason)

    # 1. Read and check every input; nothing is written in this part.
    final_reports, final_issues, names = validate_dataset(
        images_dir, final_dir, classes_file, strict=True
    )
    final_summary = summarize(final_reports, final_issues, names)
    if final_summary["errors"]:
        for issue in final_issues:
            if issue.severity == "error":
                print(f"error: {issue.image}: {issue.code}: {issue.message}", file=sys.stderr)
        return fail(f"{final_dir} has errors, fix them first")
    reports = {report.image: report for report in final_reports}
    image_paths = [images_dir / report.image for report in final_reports]

    groups = read_source_groups(args.annotations / "source_groups.csv", list(reports))
    group_of_image = {image: entry["group"] for image, entry in groups.items()}
    group_pins = {entry["group"]: entry["pin"] for entry in groups.values()}
    assignment = split_groups(Counter(group_of_image.values()), group_pins, args.seed)
    split_of_image = {image: assignment[group] for image, group in group_of_image.items()}
    classes_of_image = {
        image: [box.class_id for box in report.boxes] for image, report in reports.items()
    }
    split_tables = format_split_tables(split_of_image, group_of_image, reports, names)
    print(split_tables)
    problems = find_split_problems(split_of_image, group_of_image, classes_of_image)
    if problems:
        for problem in problems:
            print(f"error: split: {problem}", file=sys.stderr)
        return fail("try another --seed or revise source_groups.csv")
    if args.plan_only:
        return 0

    unknown_samples = sorted(set(SAMPLE_IMAGES) - set(reports))
    if unknown_samples:
        return fail(f"SAMPLE_IMAGES lists images that do not exist: {unknown_samples}")
    to_convert = [path for path in image_paths if needs_conversion(reports[path.name])]
    transparent = [path.name for path in to_convert if find_transparency(path)]
    if transparent:
        return fail(f"images with real transparency, converting would change them: {transparent}")
    output_names = [output_file_name(path, reports[path.name]) for path in image_paths]
    if len(set(output_names)) != len(output_names):
        return fail("two images map to the same output file name")
    source_reports, source_issues, _ = validate_dataset(
        images_dir, args.input / "labels", classes_file
    )
    source_summary = summarize(source_reports, source_issues, names)
    raw_lines = count_raw_label_lines(args.input / "labels", {path.stem for path in image_paths})
    changes = read_changes(args.annotations / "changes.csv")

    # 2. Write. Old outputs are removed first so nothing stale is hashed or committed.
    for stale in ("images", "labels", "dataset.yaml"):
        remove_if_exists(args.output / stale)
    for subdir in ("images", "labels"):
        for split in SPLITS:
            (args.output / subdir / split).mkdir(parents=True)
    for image_path, output_name in zip(image_paths, output_names, strict=True):
        split = split_of_image[image_path.name]
        write_image(
            image_path, reports[image_path.name], args.output / "images" / split / output_name
        )
        write_labels(
            final_dir / f"{image_path.stem}.txt",
            args.output / "labels" / split / f"{image_path.stem}.txt",
        )
    # No `path:` key: Ultralytics then resolves train/val/test relative to this file, so the
    # dataset loads from any working directory.
    dataset_yaml = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(names)),
    }
    (args.output / "dataset.yaml").write_text(
        yaml.safe_dump(dataset_yaml, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )

    args.reports.mkdir(parents=True, exist_ok=True)
    with open(args.reports / "split_manifest.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["image", "file", "group", "split", "pinned"])
        for image_path, output_name in zip(image_paths, output_names, strict=True):
            entry = groups[image_path.name]
            writer.writerow(
                [
                    image_path.name,
                    output_name,
                    entry["group"],
                    split_of_image[image_path.name],
                    bool(entry["pin"]),
                ]
            )

    samples_dir = args.reports / "samples"
    remove_if_exists(samples_dir / "all")
    for old_sample in samples_dir.glob("*.jpg"):
        old_sample.unlink()
    (samples_dir / "all").mkdir(parents=True)
    for image_path in image_paths:
        draw_boxes(
            image_path,
            reports[image_path.name],
            names,
            samples_dir / "all" / f"{image_path.stem}.jpg",
        )
    for image_name, sample_name in SAMPLE_IMAGES.items():
        shutil.copyfile(
            samples_dir / "all" / f"{Path(image_name).stem}.jpg", samples_dir / f"{sample_name}.jpg"
        )

    dataset_hash = write_manifest(args.output, args.reports / "manifest.sha256")
    report = format_report(
        args,
        split_tables,
        split_of_image,
        groups,
        reports,
        source_summary,
        final_summary,
        raw_lines,
        changes,
        Counter(reports[path.name].mode for path in to_convert),
        dataset_hash,
    )
    (args.reports / "dataset_report.md").write_text(report, encoding="utf-8", newline="\n")
    print(f"\n{args.output}: {len(image_paths)} images, {len(to_convert)} converted to RGB")
    print(f"dataset hash: {dataset_hash}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
