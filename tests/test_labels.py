import json
from pathlib import Path

import pytest
from PIL import Image

from src.utils import read_class_names
from src.validate_labels import ERROR, WARNING, main, validate_dataset

CLASSES = ["bicycle", "car", "pedestrian"]


def make_dataset(
    root: Path,
    images: dict[str, str],
    labels: dict[str, bytes | str],
    size: tuple[int, int] = (100, 50),
) -> Path:
    """images: file name -> PIL mode; labels: file name -> content."""
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    (root / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
    for name, mode in images.items():
        Image.new(mode, size).save(root / "images" / name)
    for name, content in labels.items():
        data = content.encode("utf-8") if isinstance(content, str) else content
        (root / "labels" / name).write_bytes(data)
    return root


def run(root: Path):
    reports, issues, _ = validate_dataset(root / "images", root / "labels", root / "classes.txt")
    return reports, issues


def codes(issues, severity=None) -> list[str]:
    return [issue.code for issue in issues if severity is None or issue.severity == severity]


def test_clean_dataset_has_no_issues(tmp_path):
    root = make_dataset(
        tmp_path, {"a.jpg": "RGB"}, {"a.txt": "1 0.5 0.5 0.2 0.2\n2 0.3 0.3 0.1 0.1\n"}
    )
    reports, issues = run(root)
    assert issues == []
    assert len(reports[0].boxes) == 2


def test_unknown_class_is_error_and_box_dropped(tmp_path):
    root = make_dataset(
        tmp_path, {"a.jpg": "RGB"}, {"a.txt": "1 0.5 0.5 0.2 0.2\n3 0.5 0.5 0.2 0.2\n"}
    )
    reports, issues = run(root)
    assert codes(issues, ERROR) == ["unknown_class"]
    assert issues[0].line == 2
    assert len(reports[0].boxes) == 1


@pytest.mark.parametrize(
    "line, code",
    [
        ("1 0.5 0.5 0.2", "wrong_field_count"),
        ("1 0.5 0.5 0.2 0.2 0.9", "wrong_field_count"),
        ("1 0.5 abc 0.2 0.2", "not_a_number"),
        ("1.5 0.5 0.5 0.2 0.2", "class_not_integer"),
        ("-1 0.5 0.5 0.2 0.2", "unknown_class"),
        ("1 0.5 0.5 0 0.2", "non_positive_size"),
        ("1 1.5 0.5 0.2 0.2", "value_out_of_range"),
    ],
)
def test_malformed_lines_are_errors(tmp_path, line, code):
    root = make_dataset(tmp_path, {"a.jpg": "RGB"}, {"a.txt": line + "\n"})
    _, issues = run(root)
    assert code in codes(issues, ERROR)


def test_out_of_bounds_uses_corners_not_center(tmp_path):
    # Center is inside the image, but the bottom edge is at 0.95 + 0.15 / 2 = 1.025.
    root = make_dataset(tmp_path, {"a.jpg": "RGB"}, {"a.txt": "1 0.5 0.95 0.2 0.15\n"})
    _, issues = run(root)
    assert codes(issues, ERROR) == ["box_out_of_bounds"]


def test_small_overflow_is_warning_and_rounding_is_ignored(tmp_path):
    # Image is 2000x1000.
    labels = {
        "a.txt": "1 0.5 0.9525 0.2 0.1\n",  # bottom at 1.0025: 2.5 px, under 1% -> warning
        "b.txt": "1 0.5 0.95 0.2 0.100001\n",  # bottom at 1.0000005: rounding -> ignored
        "c.txt": "1 0.5 0.95 0.2 0.1008\n",  # bottom at 1.0004: 0.4 px -> ignored
    }
    images = {"a.jpg": "RGB", "b.jpg": "RGB", "c.jpg": "RGB"}
    root = make_dataset(tmp_path, images, labels, size=(2000, 1000))
    _, issues = run(root)
    assert [(issue.image, issue.code, issue.severity) for issue in issues] == [
        ("a.jpg", "box_out_of_bounds", WARNING)
    ]


def test_strict_makes_any_real_overflow_an_error(tmp_path):
    root = make_dataset(
        tmp_path, {"a.jpg": "RGB"}, {"a.txt": "1 0.5 0.9525 0.2 0.1\n"}, size=(2000, 1000)
    )
    _, issues, _ = validate_dataset(
        root / "images", root / "labels", root / "classes.txt", strict=True
    )
    assert [(issue.code, issue.severity) for issue in issues] == [("box_out_of_bounds", ERROR)]


def test_missing_label_is_reported_not_treated_as_background(tmp_path):
    root = make_dataset(
        tmp_path, {"a.jpg": "RGB", "b.jpg": "RGB"}, {"a.txt": "1 0.5 0.5 0.2 0.2\n"}
    )
    reports, issues = run(root)
    assert codes(issues) == ["label_missing"]
    assert issues[0].image == "b.jpg"
    assert reports[1].label_file is None


def test_label_without_image_is_error(tmp_path):
    root = make_dataset(
        tmp_path, {"a.jpg": "RGB"}, {"a.txt": "1 0.5 0.5 0.2 0.2\n", "ghost.txt": ""}
    )
    _, issues = run(root)
    assert "label_without_image" in codes(issues, ERROR)


def test_crlf_is_warning_and_still_parsed(tmp_path):
    root = make_dataset(
        tmp_path, {"a.jpg": "RGB"}, {"a.txt": b"1 0.5 0.5 0.2 0.2\r\n2 0.3 0.3 0.1 0.1\r\n"}
    )
    reports, issues = run(root)
    assert codes(issues) == ["crlf_line_endings"]
    assert len(reports[0].boxes) == 2


def test_empty_line_and_empty_file(tmp_path):
    labels = {"a.txt": "1 0.5 0.5 0.2 0.2\n\n", "b.txt": ""}
    root = make_dataset(tmp_path, {"a.jpg": "RGB", "b.jpg": "RGB"}, labels)
    _, issues = run(root)
    assert sorted(codes(issues, WARNING)) == ["empty_line", "label_empty"]


def test_duplicate_box_same_class_only(tmp_path):
    content = "1 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.2 0.2\n2 0.5 0.5 0.2 0.2\n"
    root = make_dataset(tmp_path, {"a.jpg": "RGB"}, {"a.txt": content})
    _, issues = run(root)
    assert codes(issues) == ["duplicate_box"]
    assert issues[0].line == 2


def test_tiny_box_in_pixels(tmp_path):
    # Image is 100x50: width 0.01 -> 1 px.
    root = make_dataset(tmp_path, {"a.jpg": "RGB"}, {"a.txt": "1 0.5 0.5 0.01 0.2\n"})
    _, issues = run(root)
    assert codes(issues) == ["tiny_box"]


def test_broken_and_truncated_images(tmp_path):
    root = make_dataset(tmp_path, {"good.png": "RGB"}, {})
    (root / "images" / "garbage.jpg").write_bytes(b"not an image")
    full = (root / "images" / "good.png").read_bytes()
    (root / "images" / "cut.png").write_bytes(full[: len(full) // 2])
    _, issues = run(root)
    unreadable = sorted(issue.image for issue in issues if issue.code == "image_unreadable")
    assert unreadable == ["cut.png", "garbage.jpg"]


def test_rgba_image_is_warning(tmp_path):
    root = make_dataset(tmp_path, {"a.png": "RGBA"}, {"a.txt": "1 0.5 0.5 0.2 0.2\n"})
    _, issues = run(root)
    assert codes(issues) == ["image_mode"]


def test_ambiguous_stem(tmp_path):
    root = make_dataset(
        tmp_path, {"a.jpg": "RGB", "a.png": "RGB"}, {"a.txt": "1 0.5 0.5 0.2 0.2\n"}
    )
    _, issues = run(root)
    assert codes(issues, ERROR).count("ambiguous_stem") == 2


def test_names_with_spaces_cyrillic_and_dots(tmp_path):
    names = ["Снимок экрана 1.png", "1649839828.0003297.jpg"]
    labels = {
        "Снимок экрана 1.txt": "1 0.5 0.5 0.2 0.2\n",
        "1649839828.0003297.txt": "2 0.5 0.5 0.2 0.2\n",
    }
    root = make_dataset(tmp_path, {name: "RGB" for name in names}, labels)
    reports, issues = run(root)
    assert issues == []
    assert all(report.label_file for report in reports)


def test_cli_exit_code_and_json(tmp_path):
    root = make_dataset(tmp_path / "ok", {"a.jpg": "RGB"}, {"a.txt": "1 0.5 0.5 0.2 0.2\n"})
    assert main(["--input", str(root)]) == 0

    bad = make_dataset(tmp_path / "bad", {"a.jpg": "RGB"}, {"a.txt": "9 0.5 0.5 0.2 0.2\n"})
    json_path = tmp_path / "out" / "report.json"
    assert main(["--input", str(bad), "--json", str(json_path)]) == 1
    assert '"unknown_class"' in json_path.read_text(encoding="utf-8")


def test_separate_labels_dir(tmp_path):
    root = make_dataset(tmp_path, {"a.jpg": "RGB"}, {"a.txt": "9 0.5 0.5 0.2 0.2\n"})
    fixed = tmp_path / "fixed"
    fixed.mkdir()
    (fixed / "a.txt").write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    assert main(["--input", str(root), "--labels", str(fixed)]) == 0


# --- review fixes -------------------------------------------------------------------


def test_missing_labels_dir_exits_2(tmp_path):
    root = make_dataset(tmp_path, {"a.jpg": "RGB"}, {"a.txt": "1 0.5 0.5 0.2 0.2\n"})
    assert main(["--input", str(root), "--labels", str(tmp_path / "typo")]) == 2


def test_strict_makes_missing_label_an_error(tmp_path):
    root = make_dataset(
        tmp_path, {"a.jpg": "RGB", "b.jpg": "RGB"}, {"a.txt": "1 0.5 0.5 0.2 0.2\n"}
    )
    assert main(["--input", str(root)]) == 0
    assert main(["--input", str(root), "--strict"]) == 1


def test_blank_line_inside_classes_is_rejected(tmp_path):
    root = make_dataset(tmp_path, {"a.jpg": "RGB"}, {"a.txt": "1 0.5 0.5 0.2 0.2\n"})
    (root / "classes.txt").write_text("bicycle\n\ncar\n", encoding="utf-8")
    with pytest.raises(ValueError, match="blank line"):
        read_class_names(root / "classes.txt")
    assert main(["--input", str(root)]) == 2


def test_classes_with_bom_and_trailing_blank_lines(tmp_path):
    classes_file = tmp_path / "classes.txt"
    classes_file.write_text("﻿bicycle\ncar\n\n\n", encoding="utf-8")
    assert read_class_names(classes_file) == ["bicycle", "car"]


def test_jpeg_without_end_marker(tmp_path):
    root = make_dataset(tmp_path, {"a.jpg": "RGB"}, {"a.txt": "1 0.5 0.5 0.2 0.2\n"})
    image_path = root / "images" / "a.jpg"
    image_path.write_bytes(image_path.read_bytes() + b"trailing")
    _, issues = run(root)
    assert codes(issues, ERROR) == ["jpeg_no_end_marker"]


def test_format_not_accepted_and_too_small(tmp_path):
    root = make_dataset(
        tmp_path, {}, {"g.txt": "1 0.5 0.5 0.2 0.2\n", "s.txt": "1 0.5 0.5 0.5 0.5\n"}
    )
    Image.new("P", (100, 50)).save(root / "images" / "g.jpg", format="GIF")
    Image.new("RGB", (8, 8)).save(root / "images" / "s.png")
    _, issues = run(root)
    by_image = {(issue.image, issue.code) for issue in issues if issue.severity == ERROR}
    assert ("g.jpg", "image_format") in by_image
    assert ("s.png", "image_too_small") in by_image


def test_unhandled_files_are_reported(tmp_path):
    root = make_dataset(tmp_path, {"a.jpg": "RGB"}, {"a.txt": "1 0.5 0.5 0.2 0.2\n", "B.TXT": ""})
    (root / "images" / "sub").mkdir()
    (root / "images" / "notes.md").write_text("x")
    (root / "labels" / ".gitkeep").write_text("")
    _, issues = run(root)
    unsupported = sorted(issue.image for issue in issues if issue.code == "unsupported_file")
    assert unsupported == ["B.TXT", "notes.md", "sub"]


def test_tiff_is_checked(tmp_path):
    root = make_dataset(tmp_path, {"a.tif": "RGB"}, {"a.txt": "1 0.5 0.5 0.2 0.2\n"})
    reports, issues = run(root)
    assert issues == [] and reports[0].label_file == "a.txt"


@pytest.mark.parametrize("line", ["1 nan 0.5 0.2 0.2", "1 0.5 0.5 inf 0.2", "nan 0.5 0.5 0.2 0.2"])
def test_non_finite_values_and_json_stays_valid(tmp_path, line):
    root = make_dataset(tmp_path, {"a.jpg": "RGB"}, {"a.txt": line + "\n"})
    json_path = tmp_path / "report.json"
    assert main(["--input", str(root), "--json", str(json_path)]) == 1
    payload = json.loads(json_path.read_text(encoding="utf-8"), parse_constant=pytest.fail)
    assert "not_a_number" in payload["summary"]["issues_by_code"]


def test_invalid_boxes_not_counted_in_stats(tmp_path):
    content = "1 0.5 0.5 -0.2 0.2\n1 2 2 0.2 0.2\n1 0.5 0.95 0.2 0.15\n"
    root = make_dataset(tmp_path, {"a.jpg": "RGB"}, {"a.txt": content})
    reports, _ = run(root)
    # Out-of-bounds box is a real object and stays; the two broken ones are dropped.
    assert [box.line for box in reports[0].boxes] == [3]


def test_exif_rotation_is_error(tmp_path):
    root = make_dataset(tmp_path, {}, {"a.txt": "1 0.5 0.5 0.2 0.2\n"})
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.new("RGB", (100, 50)).save(root / "images" / "a.jpg", exif=exif)
    _, issues = run(root)
    assert codes(issues, ERROR) == ["exif_rotation"]


@pytest.mark.parametrize(
    "file_name, content, expected_exit",
    [
        ("data.yaml", "names:\n  0: bicycle\n  1: car\n  2: pedestrian\n", 0),
        ("data.yaml", "names: [bicycle, car, pedestrian]\n", 0),
        ("data.yaml", "names: [bicycle, pedestrian, car]\n", 1),
        ("obj.names", "bicycle\ncar\npedestrian\n", 0),
        ("obj.names", "car\nbicycle\npedestrian\n", 1),
    ],
)
def test_cvat_class_names_compared(tmp_path, file_name, content, expected_exit):
    root = make_dataset(tmp_path, {"a.jpg": "RGB"}, {"a.txt": "1 0.5 0.5 0.2 0.2\n"})
    names_file = tmp_path / file_name
    names_file.write_text(content, encoding="utf-8")
    assert main(["--input", str(root), "--names", str(names_file)]) == expected_exit
