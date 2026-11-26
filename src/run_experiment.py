"""Run a YOLO experiment from a config and log it to TensorBoard with our own SummaryWriter.

Modes:
    --dry-run      check config, dataset (against the manifest), classes, paths and device; print the
                   plan. Writes nothing, creates no model, downloads nothing.
    --smoke-test   the full pipeline with the config's `smoke_test` overrides (one short epoch).
    (default)      train (val split picks best.pt) -> best.pt -> test split, evaluated once.

Run directory runs/<timestamp>_<name>/:
    events.out.tfevents.*      our TensorBoard log (tags: see src/tensorboard_logger.py)
    config.yaml                the resolved config, after CLI overrides
    metadata.json              git commit + dirty flag, dataset hash, versions, seed, device, weights
    args.yaml, results.csv, *.png, weights/{best,last}.pt     written by Ultralytics
    predictions/<split>/       example predictions of best.pt
    test/                      Ultralytics test evaluation plots
    test_metrics.json          test metrics, ground-truth objects and evaluated classes per class

Usage:
    python -m src.run_experiment --config configs/yolo.yaml --dry-run
    python -m src.run_experiment --config configs/yolo.yaml --smoke-test
    python -m src.run_experiment --config configs/yolo.yaml --epochs 10
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml

REQUIRED_KEYS = {
    "name": str,
    "model": str,
    "data": str,
    "manifest": str,
    "epochs": int,
    "imgsz": int,
    "batch": int,
    "device": str,
    "workers": int,
    "seed": int,
    "deterministic": bool,
    "patience": int,
    "conf": float,
    "iou": float,
    "prediction_examples": int,
}
OVERRIDABLE = ("name", "model", "data", "epochs", "imgsz", "batch", "device", "seed")
SPLITS = ("train", "val", "test")
# Ultralytics writes labels/<split>.cache next to the labels; it is rebuilt, not part of the data.
ULTRALYTICS_CACHE = ".cache"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
# Ultralytics integrations switched off for this process only (the settings file is untouched):
# we log to TensorBoard ourselves, and nothing is sent anywhere.
DISABLED_SETTINGS = ("tensorboard", "sync", "clearml", "comet", "dvc", "mlflow", "raytune", "wandb")


def load_config(config_file: Path, overrides: dict, smoke_test: bool) -> dict:
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    smoke_overrides = config.pop("smoke_test", {}) or {}
    if smoke_test:
        config.update(smoke_overrides)
        config["name"] = f"{config['name']}_smoke"
    config.update({key: value for key, value in overrides.items() if value is not None})
    problems = []
    for key, expected in REQUIRED_KEYS.items():
        if key not in config:
            problems.append(f"missing key {key!r}")
        elif (
            expected is float and isinstance(config[key], int) and not isinstance(config[key], bool)
        ):
            config[key] = float(config[key])
        elif not isinstance(config[key], expected) or (
            expected is int and isinstance(config[key], bool)
        ):
            problems.append(f"{key!r} must be {expected.__name__}, got {config[key]!r}")
    unknown = sorted(set(config) - set(REQUIRED_KEYS))
    if unknown:
        problems.append(f"unknown keys {unknown}")
    for key in ("model", "data", "manifest"):
        if isinstance(config.get(key), str) and Path(config[key]).is_absolute():
            problems.append(f"{key!r} must be a path relative to the repo root")
    if problems:
        raise ValueError(f"{config_file}: " + "; ".join(problems))
    return config


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(manifest: Path, data_dir: Path) -> tuple[str, list[str]]:
    """(dataset hash, problems). The dataset must contain exactly the files of the manifest, with the
    same content; the dataset hash is the sha256 of the manifest text."""
    text = manifest.read_text(encoding="utf-8")
    listed = {}
    for line in text.splitlines():
        digest, path = line.split("  ", 1)
        listed[Path(path)] = digest
    problems = [
        f"changed: {path}"
        for path, digest in listed.items()
        if not path.is_file() or file_sha256(path) != digest
    ]
    on_disk = {
        path
        for subdir in ("images", "labels")
        for path in (data_dir / subdir).rglob("*")
        if path.is_file() and path.suffix != ULTRALYTICS_CACHE
    }
    on_disk.add(data_dir / "dataset.yaml")
    problems += [f"not in manifest: {path}" for path in sorted(on_disk - set(listed))]
    return hashlib.sha256(text.encode("utf-8")).hexdigest(), problems


def check_dataset(data_file: Path) -> tuple[dict, dict, list[str]]:
    """(dataset.yaml, images per split, problems): split folders, image/label pairs, class ids."""
    problems = []
    spec = yaml.safe_load(data_file.read_text(encoding="utf-8"))
    names = spec.get("names")
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names)]
    if not names:
        problems.append("dataset.yaml has no class names")
        names = []
    if "path" in spec:
        problems.append("dataset.yaml has `path:`; it must resolve relative to the yaml file")
    counts = {}
    for split in SPLITS:
        if split not in spec:
            problems.append(f"dataset.yaml has no {split!r}")
            continue
        images_dir = data_file.parent / spec[split]
        labels_dir = data_file.parent / spec[split].replace("images", "labels", 1)
        images = sorted(p for p in images_dir.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        counts[split] = len(images)
        if not images:
            problems.append(f"{split}: no images in {images_dir}")
        for image in images:
            label = labels_dir / f"{image.stem}.txt"
            if not label.is_file():
                problems.append(f"{split}: no label file for {image.name}")
                continue
            for number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                class_id = int(line.split()[0])
                if not 0 <= class_id < len(names):
                    problems.append(f"{label}:{number}: class {class_id} not in names")
    return spec, counts, problems


def count_objects(labels_dir: Path, num_classes: int) -> list[int]:
    counts = Counter(
        int(line.split()[0])
        for path in sorted(labels_dir.glob("*.txt"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    return [counts[class_id] for class_id in range(num_classes)]


def git_state() -> dict:
    def git(*args: str) -> str:
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    status = git("status", "--porcelain")
    return {"commit": git("rev-parse", "HEAD"), "dirty": status not in ("", "unknown")}


def disable_ultralytics_integrations() -> None:
    from ultralytics.utils import SETTINGS

    for key in DISABLED_SETTINGS:
        if key in SETTINGS:
            dict.__setitem__(
                SETTINGS, key, False
            )  # in memory: SETTINGS[key] = ... would save to disk


def predict_examples(model, images: list[Path], out_dir: Path, config: dict) -> list[Path]:
    """Draw predictions of `model` on `images`; returns the written files."""
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for image in images:
        result = model.predict(
            str(image), imgsz=config["imgsz"], device=config["device"], conf=0.25, verbose=False
        )[0]
        out = out_dir / f"{image.stem}.jpg"
        Image.fromarray(result.plot()[..., ::-1]).save(out, quality=90)  # plot() returns BGR
        written.append(out)
    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=Path("configs/yolo.yaml"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="check everything, write nothing")
    mode.add_argument("--smoke-test", action="store_true", help="one short epoch, full logging")
    parser.add_argument("--runs", type=Path, default=Path("runs"), help="parent of run directories")
    for key in OVERRIDABLE:
        kind = int if REQUIRED_KEYS[key] is int else str
        parser.add_argument(f"--{key}", type=kind, help=f"override `{key}` from the config")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(
            args.config, {key: getattr(args, key) for key in OVERRIDABLE}, args.smoke_test
        )
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    data_file, manifest = Path(config["data"]), Path(config["manifest"])
    problems = []
    if not data_file.is_file():
        problems.append(f"{data_file} not found: run python -m src.prepare_dataset first")
    if not manifest.is_file():
        problems.append(f"{manifest} not found: run python -m src.prepare_dataset first")
    if problems:
        print("\n".join(f"error: {p}" for p in problems), file=sys.stderr)
        return 1
    spec, image_counts, problems = check_dataset(data_file)
    dataset_hash, manifest_problems = verify_manifest(manifest, data_file.parent)
    problems += manifest_problems[:10]
    if manifest_problems:
        problems.append("data differs from the manifest: rerun python -m src.prepare_dataset")
    if config["device"] != "cpu":
        import torch

        if not torch.cuda.is_available():
            problems.append(f"device {config['device']!r} requested but CUDA is not available")
    if problems:
        print("\n".join(f"error: {p}" for p in problems), file=sys.stderr)
        return 1

    weights = Path(config["model"])
    plan = {
        "config": config,
        "dataset_hash": dataset_hash,
        "images": image_counts,
        "classes": len(spec["names"]),
        "weights": f"{weights} ({'present' if weights.is_file() else 'downloaded on first run'})",
        "git": git_state(),
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if args.dry_run:
        print("dry run: config, dataset and device are OK; nothing written")
        return 0

    import torch
    import ultralytics
    from ultralytics import YOLO

    from src.tensorboard_logger import TensorBoardLogger

    disable_ultralytics_integrations()
    run_dir = args.runs / f"{datetime.now():%Y-%m-%dT%H%M%S}_{config['name']}"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    weights.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))  # downloads the pretrained weights into weights/ if missing
    metadata = {
        "run": run_dir.name,
        "command": " ".join(["python -m src.run_experiment", *(argv or sys.argv[1:])]),
        "git": git_state(),
        "dataset": {
            "manifest": manifest.as_posix(),
            "sha256": dataset_hash,
            "images": image_counts,
        },
        "weights": {"file": weights.as_posix(), "sha256": file_sha256(weights)},
        "seed": config["seed"],
        "device": config["device"],
        "validation_thresholds": {"conf": config["conf"], "iou": config["iou"]},
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
        },
        "platform": platform.platform(),
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger = TensorBoardLogger(run_dir)
    try:
        logger.log_run_info(config, metadata)
        model.add_callback("on_train_epoch_end", logger.on_train_epoch_end)
        model.add_callback("on_fit_epoch_end", logger.on_fit_epoch_end)
        # Ultralytics puts a relative `project` under its own runs_dir setting, so the
        # absolute path is computed at run time; nothing absolute is stored in the config.
        model.train(
            data=str(data_file),
            epochs=config["epochs"],
            imgsz=config["imgsz"],
            batch=config["batch"],
            device=config["device"],
            workers=config["workers"],
            seed=config["seed"],
            deterministic=config["deterministic"],
            patience=config["patience"],
            conf=config["conf"],
            iou=config["iou"],
            project=str(run_dir.parent.resolve()),
            name=run_dir.name,
            exist_ok=True,
            plots=True,
        )
        last_step = logger.logged_steps[-1] if logger.logged_steps else 0
        logger.log_image(
            "val/confusion_matrix", run_dir / "confusion_matrix_normalized.png", last_step
        )

        best = YOLO(str(run_dir / "weights" / "best.pt"))
        for split in ("val", "test"):
            images = sorted(
                p
                for p in (data_file.parent / spec[split]).glob("*")
                if p.suffix.lower() in IMAGE_SUFFIXES
            )[: config["prediction_examples"]]
            for path in predict_examples(best, images, run_dir / "predictions" / split, config):
                logger.log_image(f"predictions/{split}/{path.stem}", path, last_step)

        # Test split: evaluated once, with best.pt chosen on val; nothing is tuned on it.
        test = best.val(
            data=str(data_file),
            split="test",
            imgsz=config["imgsz"],
            batch=config["batch"],
            device=config["device"],
            conf=config["conf"],
            iou=config["iou"],
            project=str(run_dir.resolve()),
            name="test",
            exist_ok=True,
            plots=True,
        )
        test_scalars = {
            "precision": float(test.box.mp),
            "recall": float(test.box.mr),
            "mAP50": float(test.box.map50),
            "mAP50-95": float(test.box.map),
        }
        logger.log_scalars("test", test_scalars, last_step)
        logger.log_image(
            "test/confusion_matrix", run_dir / "test" / "confusion_matrix_normalized.png", last_step
        )

        names = (
            [spec["names"][key] for key in sorted(spec["names"])]
            if isinstance(spec["names"], dict)
            else spec["names"]
        )
        objects = count_objects(
            data_file.parent / spec["test"].replace("images", "labels", 1), len(names)
        )
        evaluated = [int(c) for c in test.box.ap_class_index]
        per_class = {
            names[c]: {
                "objects": objects[c],
                "evaluated": c in evaluated,
                **(
                    {
                        "mAP50": float(test.box.ap50[evaluated.index(c)]),
                        "mAP50-95": float(test.box.ap[evaluated.index(c)]),
                    }
                    if c in evaluated
                    else {}
                ),
            }
            for c in range(len(names))
        }
        not_evaluated = [names[c] for c in range(len(names)) if c not in evaluated]
        test_report = {
            "weights": "weights/best.pt",
            "split": "test",
            "images": image_counts["test"],
            "metrics": test_scalars,
            "best_val_metrics": logger.best_val_metrics,
            "classes": per_class,
            "evaluated_classes": [names[c] for c in evaluated],
            "warning": (
                "mAP is the mean over evaluated classes only; classes without test objects "
                f"({', '.join(not_evaluated) or 'none'}) have no AP, which is not AP = 0. "
                f"{image_counts['test']} test images: a sanity check, not a benchmark."
            ),
        }
        (run_dir / "test_metrics.json").write_text(
            json.dumps(test_report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        hparams = {
            key: config[key] for key in ("model", "epochs", "imgsz", "batch", "seed", "device")
        }
        logger.log_hparams(
            hparams,
            {
                **{f"hparam/val_{k}": v for k, v in logger.best_val_metrics.items()},
                **{f"hparam/test_{k}": v for k, v in test_scalars.items()},
            },
        )
    finally:
        logger.close()
    print(json.dumps(test_report["metrics"], indent=2))
    print(f"run directory: {run_dir}\ntensorboard --logdir {args.runs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
