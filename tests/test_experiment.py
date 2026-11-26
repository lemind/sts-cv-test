from pathlib import Path

import pytest
import yaml
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.prepare_dataset import write_manifest
from src.run_experiment import (
    alias_pretrained_class_names,
    load_config,
    main,
    train_arguments,
    verify_manifest,
)
from src.tensorboard_logger import TensorBoardLogger

BASE_CONFIG = {
    "name": "exp",
    "model": "weights/yolo11n.pt",
    "data": "data/processed/dataset.yaml",
    "manifest": "reports/manifest.sha256",
    "epochs": 5,
    "imgsz": 640,
    "batch": 4,
    "device": "cpu",
    "workers": 0,
    "seed": 42,
    "deterministic": True,
    "patience": 100,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "warmup_epochs": 1,
    "close_mosaic": 1,
    "freeze": 0,
    "class_aliases": {"person": "pedestrian"},
    "conf": 0.001,
    "iou": 0.7,
    "prediction_examples": 2,
    "smoke_test": {"epochs": 1, "imgsz": 320},
}


def write_config(path, **changes):
    path.write_text(yaml.safe_dump({**BASE_CONFIG, **changes}), encoding="utf-8")
    return path


def test_cli_overrides_and_smoke_test_settings(tmp_path):
    config_file = write_config(tmp_path / "c.yaml")
    config = load_config(config_file, {"epochs": 9, "seed": None}, smoke_test=False)
    assert config["epochs"] == 9 and config["seed"] == 42 and "smoke_test" not in config
    smoke = load_config(config_file, {}, smoke_test=True)
    assert smoke["epochs"] == 1 and smoke["imgsz"] == 320 and smoke["name"] == "exp_smoke"


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"epochs": "5"}, "'epochs' must be int"),
        ({"deterministic": "yes"}, "'deterministic' must be bool"),
        ({"momentum": 0.9}, "unknown keys"),
        ({"data": "/abs/dataset.yaml"}, "relative to the repo root"),
        ({"batch": 0}, "'batch' must be >= 1"),
        ({"iou": 1.5}, "'iou' must be <= 1"),
    ],
)
def test_bad_config_is_rejected(tmp_path, changes, error):
    with pytest.raises(ValueError, match=error):
        load_config(write_config(tmp_path / "c.yaml", **changes), {}, smoke_test=False)


def test_nominal_batch_follows_batch_override(tmp_path):
    # nbs != batch makes Ultralytics accumulate gradients: 3 optimizer steps in a 5-epoch run.
    config = load_config(write_config(tmp_path / "c.yaml"), {"batch": 2}, smoke_test=False)
    arguments = train_arguments(config, Path("d.yaml"), tmp_path / "runs" / "r")
    assert arguments["batch"] == arguments["nbs"] == 2
    assert arguments["optimizer"] == "AdamW" and arguments["lr0"] == 0.001
    assert arguments["close_mosaic"] == 1 and arguments["warmup_bias_lr"] == 0.0


class FakeYOLO:
    def __init__(self, names):
        self.model = type("Net", (), {"names": dict(names)})()


def test_class_aliases_rename_pretrained_classes_and_reject_unknown():
    model = FakeYOLO({0: "person", 1: "bicycle"})
    assert alias_pretrained_class_names(model, {"person": "pedestrian"}) == ["person -> pedestrian"]
    assert model.model.names == {0: "pedestrian", 1: "bicycle"}
    with pytest.raises(ValueError, match="not classes of the pretrained model"):
        alias_pretrained_class_names(FakeYOLO({0: "car"}), {"person": "pedestrian"})


def test_dry_run_rejects_alias_to_unknown_dataset_class(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make_processed_dataset(tmp_path)
    write_config(tmp_path / "c.yaml", class_aliases={"person": "walker"})
    assert main(["--config", "c.yaml", "--dry-run"]) == 1


def make_processed_dataset(root):
    """data/processed with one image per split and its manifest, relative to `root`."""
    data = root / "data" / "processed"
    for split in ("train", "val", "test"):
        (data / "images" / split).mkdir(parents=True)
        (data / "labels" / split).mkdir(parents=True)
        Image.new("RGB", (32, 32)).save(data / "images" / split / f"{split}.png")
        (data / "labels" / split / f"{split}.txt").write_text("1 0.5 0.5 0.2 0.2\n")
    spec = {"train": "images/train", "val": "images/val", "test": "images/test"}
    spec["names"] = {0: "bicycle", 1: "car", 2: "pedestrian"}
    (data / "dataset.yaml").write_text(yaml.safe_dump(spec))
    (root / "reports").mkdir()
    write_manifest(Path("data/processed"), Path("reports/manifest.sha256"))


def test_dry_run_checks_everything_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make_processed_dataset(tmp_path)
    write_config(tmp_path / "c.yaml")
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    assert main(["--config", "c.yaml", "--dry-run"]) == 0
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before
    assert not (tmp_path / "runs").exists() and not (tmp_path / "weights").exists()


def test_dry_run_fails_when_data_differs_from_manifest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make_processed_dataset(tmp_path)
    write_config(tmp_path / "c.yaml")
    (tmp_path / "data/processed/labels/val/val.txt").write_text("1 0.4 0.5 0.2 0.2\n")
    assert main(["--config", "c.yaml", "--dry-run"]) == 1


def test_manifest_check_ignores_ultralytics_cache_but_not_other_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make_processed_dataset(tmp_path)
    manifest, data = Path("reports/manifest.sha256"), Path("data/processed")
    (data / "labels" / "train.cache").write_bytes(b"cache")
    assert verify_manifest(manifest, data)[1] == []
    (data / "images" / "train" / "extra.png").write_bytes(b"x")
    assert verify_manifest(manifest, data)[1] == [
        "not in manifest: data/processed/images/train/extra.png"
    ]


class FakeTrainer:
    """The attributes of an Ultralytics trainer that the logger reads."""

    def __init__(self):
        self.epoch, self.metrics, self.lr = 0, {}, {"lr/pg0": 0.01}
        self.tloss = {"box_loss": 1.0, "cls_loss": 2.0, "dfl_loss": 3.0}

    def label_loss_items(self, loss_items, prefix="train"):
        return {f"{prefix}/{k}": v for k, v in loss_items.items()}


def epoch_metrics(map50):
    return {
        "metrics/precision(B)": 0.5,
        "metrics/recall(B)": 0.4,
        "metrics/mAP50(B)": map50,
        "metrics/mAP50-95(B)": map50 / 2,
        "val/box_loss": 1.5,
        "val/cls_loss": 2.5,
        "val/dfl_loss": 3.5,
    }


def test_logger_writes_one_step_per_epoch_and_best_validation_separately(tmp_path):
    logger = TensorBoardLogger(tmp_path)
    trainer = FakeTrainer()
    for epoch, map50 in enumerate([0.1, 0.3]):
        trainer.epoch, trainer.metrics = epoch, epoch_metrics(map50)
        logger.on_train_epoch_end(trainer)
        logger.on_fit_epoch_end(trainer)
    # Ultralytics' final validation of best.pt: epoch + 1, no on_train_epoch_end before it.
    trainer.epoch, trainer.metrics = (
        2,
        {k: v for k, v in epoch_metrics(0.3).items() if "loss" not in k},
    )
    logger.on_fit_epoch_end(trainer)
    logger.close()

    events = EventAccumulator(str(tmp_path))
    events.Reload()
    steps = {
        tag: [(e.step, round(e.value, 3)) for e in events.Scalars(tag)]
        for tag in events.Tags()["scalars"]
    }
    assert steps["val/mAP50"] == [(1, 0.1), (2, 0.3)]
    assert steps["loss/train_box"] == [(1, 1.0), (2, 1.0)] and steps["loss/val_cls"] == [
        (1, 2.5),
        (2, 2.5),
    ]
    assert steps["lr/pg0"] == [(1, 0.01), (2, 0.01)]
    assert steps["best_val/mAP50"] == [(2, 0.3)]
    assert logger.logged_steps == [1, 2]
