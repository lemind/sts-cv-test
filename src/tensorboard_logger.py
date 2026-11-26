"""Our own TensorBoard logging for an Ultralytics training run.

Tags written:
    loss/train_box, loss/train_cls, loss/train_dfl     mean training losses per epoch
    loss/val_box, loss/val_cls, loss/val_dfl           validation losses per epoch
    val/precision, val/recall, val/mAP50, val/mAP50-95 validation metrics per epoch
    lr/pg0, lr/pg1, lr/pg2                             learning rate per parameter group
    best_val/*                                         final validation of best.pt (once)
    test/*                                             test split, best.pt, evaluated once
    images: val/confusion_matrix, test/confusion_matrix, predictions/<split>/<image>
    text: config, metadata;  hparams: config values with final val/test metrics

Step = epoch number starting at 1. Ultralytics calls on_fit_epoch_end once more after training to
validate best.pt; that call is recognised (no on_train_epoch_end before it) and logged as best_val/*
instead of as an extra epoch.
"""

import json
import logging
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

LOGGER = logging.getLogger(__name__)

METRIC_TAGS = {
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
    "metrics/mAP50(B)": "mAP50",
    "metrics/mAP50-95(B)": "mAP50-95",
}
LOSS_NAMES = ("box_loss", "cls_loss", "dfl_loss")


class TensorBoardLogger:
    def __init__(self, log_dir: Path) -> None:
        self.writer = SummaryWriter(log_dir=str(log_dir))
        self.last_train_epoch: int | None = None
        self.logged_steps: list[int] = []
        self.best_val_metrics: dict[str, float] = {}
        self.missing_keys: set[str] = set()

    def log_run_info(self, config: dict, metadata: dict) -> None:
        config_text = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
        self.writer.add_text("config", f"```yaml\n{config_text}```", 0)
        metadata_text = json.dumps(metadata, indent=2, ensure_ascii=False)
        self.writer.add_text("metadata", f"```json\n{metadata_text}\n```", 0)

    def on_train_epoch_end(self, trainer) -> None:
        self.last_train_epoch = trainer.epoch

    def on_fit_epoch_end(self, trainer) -> None:
        metrics = trainer.metrics or {}
        if trainer.epoch != self.last_train_epoch:
            # Final validation of best.pt after training, not a new epoch.
            self.best_val_metrics = self._pick_metrics(metrics)
            for name, value in self.best_val_metrics.items():
                self.writer.add_scalar(
                    f"best_val/{name}", value, self.logged_steps[-1] if self.logged_steps else 0
                )
            return
        step = trainer.epoch + 1
        if trainer.tloss is not None:
            for key, value in trainer.label_loss_items(trainer.tloss, prefix="train").items():
                self.writer.add_scalar(
                    f"loss/train_{key.split('/')[1].removesuffix('_loss')}", value, step
                )
        for loss in LOSS_NAMES:
            key = f"val/{loss}"
            if key in metrics:
                self.writer.add_scalar(
                    f"loss/val_{loss.removesuffix('_loss')}", float(metrics[key]), step
                )
            else:
                self._warn_missing(key)
        for name, value in self._pick_metrics(metrics).items():
            self.writer.add_scalar(f"val/{name}", value, step)
        for name, value in (trainer.lr or {}).items():
            self.writer.add_scalar(name, float(value), step)
        self.logged_steps.append(step)
        self.writer.flush()

    def _pick_metrics(self, metrics: dict) -> dict[str, float]:
        picked = {}
        for key, name in METRIC_TAGS.items():
            if key in metrics:
                picked[name] = float(metrics[key])
            else:
                self._warn_missing(key)
        return picked

    def _warn_missing(self, key: str) -> None:
        if key not in self.missing_keys:
            self.missing_keys.add(key)
            LOGGER.warning("metric %s not reported by Ultralytics, not logged", key)

    def log_scalars(self, prefix: str, values: dict[str, float], step: int) -> None:
        for name, value in values.items():
            self.writer.add_scalar(f"{prefix}/{name}", value, step)

    def log_image(self, tag: str, image: Path | np.ndarray, step: int) -> bool:
        """Log an image file or an RGB HWC array. A missing file is warned about, not fatal."""
        if isinstance(image, Path):
            if not image.is_file():
                LOGGER.warning("image %s not found, %s not logged", image, tag)
                return False
            with Image.open(image) as opened:
                image = np.asarray(opened.convert("RGB"))
        self.writer.add_image(tag, image, step, dataformats="HWC")
        return True

    def log_hparams(self, hparams: dict, metrics: dict[str, float]) -> None:
        # run_name keeps the hparams event file in a fixed subfolder of this run.
        self.writer.add_hparams(hparams, metrics, run_name="hparams")

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()
