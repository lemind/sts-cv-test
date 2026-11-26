import csv
from collections import Counter

import pytest
from PIL import Image

from src import prepare_dataset
from src.prepare_dataset import (
    find_split_problems,
    find_transparency,
    read_source_groups,
    split_groups,
    write_image,
    write_labels,
)
from src.validate_labels import ImageReport

SIZES = {f"g{index:02d}": 1 + index % 3 for index in range(20)}  # 40 images in 20 groups


def test_split_is_deterministic_for_a_seed():
    assert split_groups(SIZES, {}, seed=42) == split_groups(SIZES, {}, seed=42)
    assert any(
        split_groups(SIZES, {}, seed=seed) != split_groups(SIZES, {}, seed=42) for seed in range(5)
    )


def test_every_group_gets_exactly_one_split_and_pins_are_kept():
    pins = {"g00": "train", "g01": "train", "g02": "test"}
    assignment = split_groups(SIZES, pins, seed=1)
    assert set(assignment) == set(SIZES)
    assert set(assignment.values()) <= {"train", "val", "test"}
    assert assignment["g00"] == assignment["g01"] == "train"
    assert assignment["g02"] == "test"


@pytest.mark.parametrize("seed", range(10))
def test_split_is_balanced_by_image_count(seed):
    images = Counter()
    for group, split in split_groups(SIZES, {}, seed=seed).items():
        images[split] += SIZES[group]
    assert 8 <= images["test"] <= 10 and 8 <= images["val"] <= 10


def test_big_group_never_goes_to_an_evaluation_split():
    sizes = {**{f"single{index}": 1 for index in range(30)}, "big": 6}  # 36 images, test target 7
    for seed in range(20):
        assert split_groups(sizes, {}, seed=seed)["big"] == "train"


def test_split_problems_leak_missing_train_class_and_tiny_eval_split():
    groups = {"a1": "a", "a2": "a", "b": "b", "c": "c", "d": "d"}
    splits = {"a1": "train", "a2": "test", "b": "val", "c": "test", "d": "train"}
    classes = {"a1": [1], "a2": [1], "b": [1], "c": [5], "d": [1]}
    problems = find_split_problems(splits, groups, classes)
    assert any("more than one split" in problem for problem in problems)
    assert any("no train boxes: [5]" in problem for problem in problems)
    assert any(problem.startswith("val: 1 groups") for problem in problems)
    assert any(problem.startswith("val: 1 images") for problem in problems)


def test_split_problems_catch_a_pinned_group_dominating_test_and_empty_splits():
    # test: 4 frames of one pinned group + 3 single frames; val: 4 images without objects
    groups = {**{f"p{i}": "pinned" for i in range(4)}, **{f"t{i}": f"t{i}" for i in range(3)}}
    groups |= {f"v{i}": f"v{i}" for i in range(4)} | {"x": "x"}
    splits = {image: "test" for image in groups if image[0] in "pt"}
    splits |= {f"v{i}": "val" for i in range(4)} | {"x": "train"}
    classes = {image: [] if image.startswith("v") else [1] for image in groups}
    problems = find_split_problems(splits, groups, classes)
    assert any("test: more than 50% from one group ['pinned']" in problem for problem in problems)
    assert "val: no labeled objects" in problems


def write_groups(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "group", "pin_split", "note"])
        writer.writerows(rows)


@pytest.mark.parametrize(
    ("rows", "error"),
    [
        ([["a.jpg", "g", "train", ""], ["b.jpg", "g", "", ""]], "pinned differently"),
        ([["a.jpg", "g", "", ""]], "without a group"),
        ([["a.jpg", "g", "", ""], ["b.jpg", "", "", ""]], "group are required"),
        ([["a.jpg", "g", "", ""], ["a.jpg", "h", "", ""]], "listed twice"),
        ([["a.jpg", "g", "", ""], ["b.jpg", "h", "dev", ""]], "unknown pin_split"),
    ],
)
def test_read_source_groups_rejects_bad_rows(tmp_path, rows, error):
    groups_file = tmp_path / "groups.csv"
    write_groups(groups_file, rows)
    with pytest.raises(ValueError, match=error):
        read_source_groups(groups_file, ["a.jpg", "b.jpg"])


def test_rgb_image_is_copied_byte_for_byte_and_rgba_is_converted(tmp_path):
    Image.new("RGB", (32, 16), (10, 20, 30)).save(tmp_path / "photo.jpg", quality=75)
    Image.new("RGBA", (32, 16), (10, 20, 30, 255)).save(tmp_path / "shot.png")

    write_image(tmp_path / "photo.jpg", ImageReport("photo.jpg", mode="RGB"), tmp_path / "a.jpg")
    assert (tmp_path / "a.jpg").read_bytes() == (tmp_path / "photo.jpg").read_bytes()
    write_image(tmp_path / "shot.png", ImageReport("shot.png", mode="RGBA"), tmp_path / "b.png")
    with Image.open(tmp_path / "b.png") as converted:
        assert converted.mode == "RGB" and converted.size == (32, 16)


def test_real_transparency_is_detected(tmp_path):
    Image.new("RGBA", (8, 8), (0, 0, 0, 255)).save(tmp_path / "opaque.png")
    Image.new("RGBA", (8, 8), (0, 0, 0, 128)).save(tmp_path / "glass.png")
    assert not find_transparency(tmp_path / "opaque.png")
    assert find_transparency(tmp_path / "glass.png")


def test_labels_are_written_with_lf_and_without_blank_lines(tmp_path):
    source = tmp_path / "a.txt"
    source.write_bytes(b"1 0.5 0.5 0.1 0.1\r\n\r\n10 0.2 0.2 0.1 0.1")
    write_labels(source, tmp_path / "b.txt")
    assert (tmp_path / "b.txt").read_bytes() == b"1 0.5 0.5 0.1 0.1\n10 0.2 0.2 0.1 0.1\n"


def make_project(root, groups, mode="RGB"):
    """Synthetic source set, corrected labels and groups: one image and one car per frame."""
    images, final = root / "src" / "images", root / "annotations" / "final"
    images.mkdir(parents=True)
    (root / "src" / "labels").mkdir()
    final.mkdir(parents=True)
    (root / "src" / "classes.txt").write_text("bicycle\ncar\n", encoding="utf-8")
    rows = []
    for index, group in enumerate(groups):
        name = f"img{index:02d}"
        color = (index * 8, 0, 0, 255) if mode == "RGBA" else (index * 8, 0, 0)
        Image.new(mode, (64, 48), color).save(images / f"{name}.png")
        (final / f"{name}.txt").write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        rows.append([f"{name}.png", group, "", ""])
    write_groups(root / "annotations" / "source_groups.csv", rows)
    (root / "annotations" / "changes.csv").write_text(
        "image,change,source_line,before,after,reason\n", encoding="utf-8"
    )


PAIRS = [f"group{index // 2}" for index in range(30)]  # 15 groups of 2 frames


def test_pipeline_output_is_reproducible_and_groups_do_not_leak(tmp_path, monkeypatch):
    make_project(tmp_path, PAIRS, mode="RGBA")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(prepare_dataset, "SAMPLE_IMAGES", {"img03.png": "example"})
    args = ["--input", "src", "--output", "data/processed", "--seed", "7"]

    assert prepare_dataset.main(args) == 0
    outputs = ("manifest.sha256", "dataset_report.md", "split_manifest.csv")
    first = {name: (tmp_path / "reports" / name).read_bytes() for name in outputs}
    (tmp_path / "data" / "processed" / "images" / "train" / "stale.png").write_bytes(b"old")
    (tmp_path / "reports" / "samples" / "renamed_sample.jpg").write_bytes(b"old")
    assert prepare_dataset.main(args) == 0
    assert {name: (tmp_path / "reports" / name).read_bytes() for name in outputs} == first

    manifest = first["manifest.sha256"].decode()
    assert manifest.count("  data/processed/") == 61  # 30 images, 30 labels, dataset.yaml
    assert (
        "stale" not in manifest and not (tmp_path / "reports/samples/renamed_sample.jpg").exists()
    )
    assert (tmp_path / "reports" / "samples" / "example.jpg").is_file()
    with open(tmp_path / "reports" / "split_manifest.csv", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    splits_of_group = {}
    processed = tmp_path / "data" / "processed"
    for row in rows:
        splits_of_group.setdefault(row["group"], set()).add(row["split"])
        with Image.open(processed / "images" / row["split"] / row["file"]) as image:
            assert image.mode == "RGB"
        assert (processed / "labels" / row["split"] / f"{row['file'][:-4]}.txt").is_file()
    assert all(len(splits) == 1 for splits in splits_of_group.values())


def test_absolute_output_gives_the_same_manifest(tmp_path, monkeypatch):
    make_project(tmp_path, PAIRS)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(prepare_dataset, "SAMPLE_IMAGES", {})
    assert prepare_dataset.main(["--input", "src", "--output", "data/processed"]) == 0
    relative = (tmp_path / "reports" / "manifest.sha256").read_text(encoding="utf-8")
    absolute_output = str(tmp_path / "data" / ".." / "data" / "processed")
    assert prepare_dataset.main(["--input", "src", "--output", absolute_output]) == 0
    assert (tmp_path / "reports" / "manifest.sha256").read_text(encoding="utf-8") == relative
    assert str(tmp_path) not in (tmp_path / "reports" / "dataset_report.md").read_text("utf-8")


@pytest.mark.parametrize("output", ["src", "src/images", ".", "annotations/out"])
def test_output_overlapping_source_or_annotations_is_refused(tmp_path, monkeypatch, output):
    make_project(tmp_path, PAIRS)
    monkeypatch.chdir(tmp_path)
    before = sorted(path.name for path in (tmp_path / "src" / "images").iterdir())
    assert prepare_dataset.main(["--input", "src", "--output", output]) == 1
    assert sorted(path.name for path in (tmp_path / "src" / "images").iterdir()) == before


def test_pipeline_writes_nothing_on_invalid_final_labels(tmp_path, monkeypatch):
    make_project(tmp_path, PAIRS)
    (tmp_path / "annotations" / "final" / "img00.txt").write_text("7 0.5 0.5 0.2 0.2\n")
    monkeypatch.chdir(tmp_path)
    assert prepare_dataset.main(["--input", "src", "--output", "out"]) == 1
    assert not (tmp_path / "out").exists() and not (tmp_path / "reports").exists()


def test_pipeline_writes_nothing_when_an_image_is_transparent(tmp_path, monkeypatch):
    make_project(tmp_path, PAIRS)
    Image.new("RGBA", (64, 48), (0, 0, 0, 0)).save(tmp_path / "src" / "images" / "img05.png")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(prepare_dataset, "SAMPLE_IMAGES", {})
    assert prepare_dataset.main(["--input", "src", "--output", "out"]) == 1
    assert not (tmp_path / "out").exists()
