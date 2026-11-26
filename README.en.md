# sts-cv-test

[Русская версия](README.md)

Validation, correction and splitting of a small traffic detection set (`test_data/traffic_v5_mini`,
40 images, 11 classes), and a reproducible Ultralytics YOLO experiment logged to TensorBoard with
our own `SummaryWriter`.

The focus is the data and the reproducibility of the pipeline, not mAP: 24 training images cannot
train a useful detector; the goal is a sound, reproducible experiment.

## Contents

- [Install](#install)
- [Pipeline](#pipeline): validate → annotations → prepare → dry-run → smoke test → train → TensorBoard
- [TensorBoard logs](#tensorboard-logs)
- [Submitted run](#submitted-run)
- [Repository layout](#repository-layout)
- [Assumptions and limitations](#assumptions-and-limitations)

## Install

Python 3.12, CPU-only PyTorch. All versions are pinned (`pyproject.toml` + `uv.lock`).

```bash
git clone https://github.com/lemind/sts-cv-test.git && cd sts-cv-test
uv sync --frozen                 # creates .venv from uv.lock, dev tools (pytest, ruff) included
```

Without uv, `requirements.txt` is an export of the same lock file (runtime and dev packages):

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt  # pulls CPU torch from download.pytorch.org (index in the file)
```

All commands below are run from the repo root. With uv, prefix them with `uv run`
(or activate `.venv`). Paths are relative: nothing depends on where the repo is cloned.

Tests (96, ~5 s):

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

## Pipeline

### 1. Validate the source set

```bash
uv run python -m src.validate_labels --input test_data/traffic_v5_mini
```

Checks every image (readable, format, mode, EXIF rotation, truncated JPEG) and every label line
(class id against `classes.txt`, coordinate range, boxes past the border, degenerate and duplicate
boxes, CRLF), and lists images without a label file. It reports and never fixes. Exit code 1 means
errors were found; on the source set that is expected: 6 errors / 24 warnings
(`reports/validation_source.json` is the committed result, written with `--json`).

### 2. Annotations: the CVAT round-trip (already done)

The fixes are committed in `annotations/`, so this step does not need to be repeated to rebuild the
dataset. What was done and how to redo it:

| File | |
|---|---|
| `annotations/README.md` | class rules and box rules used for every decision |
| `annotations/class_fixes.csv` | class ids corrected after looking at the image (incl. the 2 unknown ids) |
| `annotations/final/` | final labels for all 40 images, exported from CVAT |
| `annotations/changes.csv` | every box changed from the source labels to `final/`, with the reason |
| `annotations/source_groups.csv` | image → source group (camera/video/incident), the split unit |

1. Pre-label: source labels with the class fixes applied, plus COCO-pretrained YOLO predictions
   for the 10 images without labels. Builds the CVAT import zip (Ultralytics YOLO Detection 1.0)
   and review images. Downloads `yolo11x.pt` (~110 MB) on the first run.
   ```bash
   uv run python -m src.prelabel --input test_data/traffic_v5_mini --output cvat
   ```
2. In CVAT: create a task with the 40 images and the 11 labels in `classes.txt` order, import
   `cvat/cvat_import.zip`, review every frame, export as Ultralytics YOLO Detection 1.0 into
   `cvat/export/`.
3. Re-validate the export and compare the class ids/names with `classes.txt`:
   ```bash
   uv run python -m src.validate_labels --input test_data/traffic_v5_mini \
       --labels cvat/export/labels/train --names cvat/export/data.yaml --strict
   ```
4. Copy `cvat/export/labels/train/*.txt` into `annotations/final/`, validate again
   (`--labels annotations/final --strict`, 0 errors) and rebuild the audit trail:
   ```bash
   uv run python -m src.annotation_changes --input test_data/traffic_v5_mini \
       --final annotations/final --output annotations/changes.csv
   ```

What the review found (details in `reports/dataset_report.md`): 14 boxes shifted off their
objects in 2 images, 160 missing boxes (10 images without labels, others incomplete), 5 class
corrections (4 before CVAT from `class_fixes.csv`, 2 of them unknown class ids, and 1 in CVAT), 5 loose and 3 false boxes, 13 RGBA images, 1 label file with CRLF.
Out-of-image boxes were all shifted boxes and were moved back by hand, never clipped.

### 3. Prepare the dataset

```bash
uv run python -m src.prepare_dataset --input test_data/traffic_v5_mini --output data/processed --seed 42
```

| Output | |
|---|---|
| `data/processed/{images,labels}/{train,val,test}/`, `dataset.yaml` | the YOLO dataset (gitignored) |
| `reports/dataset_report.md` | statistics, split, every problem found and what was done |
| `reports/split_manifest.csv` | image → group → split |
| `reports/samples/*.jpg` | 6 images with their final boxes (committed); all 40 in `samples/all/` (gitignored) |
| `reports/manifest.sha256` | sha256 of every dataset file; its own sha256 is the dataset version |

- **Split by source group:** frames of one camera, video or incident stay in one split.
  Groups are balanced by image count (about 20% val, 20% test), deterministic for `--seed`:
  train 24 images / 444 boxes, val 8 / 204, test 8 / 218.
- **Pinned groups:** the 4 COCO train2017 images go to train, because the pretrained weights have
  seen them. One highway camera also goes to train: it holds one of the only 2 `lorry` boxes.
- **What makes it stop:** a class with no train boxes, a group in two splits, or a val/test
  split with fewer than 3 groups or 4 images, or with more than half of its images from one group.
- **Images:** copied byte for byte. RGBA PNGs (alpha checked fully opaque) are converted to RGB
  PNG.
- **Labels:** written with LF line endings.
- **Order of work:** everything is read and checked before the first file is written. Stale
  outputs are removed.

Use `--plan-only` to print the split tables without writing anything. Rebuilding and checking
against the manifest: [`data/README.md`](data/README.md).

### 4. Dry run

```bash
uv run python -m src.run_experiment --config configs/yolo.yaml --data data/processed/dataset.yaml --dry-run
```

Checks the config (types, unknown keys, absolute paths), the dataset against
`reports/manifest.sha256`, image/label pairs and class ids, the device. Prints the resolved plan.
Writes nothing, builds no model and downloads nothing.

### 5. Smoke test

```bash
uv run python -m src.run_experiment --config configs/yolo.yaml --smoke-test
```

The full pipeline with the config's `smoke_test` overrides (1 epoch, imgsz 320, ~1 min on CPU),
in `runs/<timestamp>_yolo11n_smoke/`. The first real run downloads the COCO-pretrained
`yolo11n.pt` (~5 MB) into `weights/` (gitignored); its sha256 is recorded in `metadata.json`.

### 6. Train

```bash
uv run python -m src.run_experiment --config configs/yolo.yaml
```

50 epochs at imgsz 640 on CPU, ~8 min. All settings are in [`configs/yolo.yaml`](configs/yolo.yaml).
The CLI overrides `--name --model --data --epochs --imgsz --batch --device --seed --lr0
--warmup_epochs --close_mosaic --freeze` take precedence, and the resolved config is saved in the run.

Training settings that differ from Ultralytics defaults, and why:

| Setting | Value | Default | Why |
|---|---|---|---|
| `nbs` (nominal batch) | = `batch` (4), always, also after `--batch` | 64 | With 64, gradients are accumulated over up to 16 batches. With 6 batches per epoch, a short run got 3 optimizer steps and the val metrics stayed flat |
| `optimizer`, `lr0` | AdamW, 0.0003 | `auto` | `auto` ignores `lr0` and picks the optimizer and lr itself, so the run record showed an lr that was not used. 0.0003 was chosen on val |
| `warmup_bias_lr` | 0.0 | 0.1 | 0.1 is meant for SGD; with AdamW it is 300× `lr0` |
| `class_aliases` | `person: pedestrian` | — | Ultralytics copies classifier weights only between classes with the same name. Without this, `pedestrian` would start from scratch instead of from COCO `person` |

**Why 50 epochs.** The short run the task asks for is `--smoke-test` (1 epoch). The 50-epoch run is
an extra check that the pipeline really trains the model; it is not an attempt to get high
accuracy on 40 images.

**Tuning.** A first 5-epoch run with AdamW lr 0.001 scored test mAP50 0.038: fine-tuning clearly
did not work. Then 4 candidates of 50 epochs were trained and one was chosen on val only
(mAP50-95 of `best.pt`):

| Candidate | `lr0` | `freeze` | val mAP50 | val mAP50-95 |
|---|---|---|---:|---:|
| A | 0.001 | 0 (whole network trains) | 0.272 | 0.170 |
| B | 0.001 | 10 (backbone frozen) | 0.425 | 0.258 |
| **C** | **0.0003** | **0** | **0.473** | **0.336** |
| D | 0.0003 | 10 | 0.390 | 0.271 |

The settings of C are in the config. Val has 8 images and was used both to pick the epoch and to
pick the settings, so its metrics are optimistic. One seed was compared; part of the difference
between candidates may be noise. The script also computed test metrics for the candidates (it
evaluates test in every run), but they were not used for the choice.

Ultralytics NMS has a wall-clock limit (2 s + 0.05 s per image). On CPU with conf 0.001 the val
batch of 8 images hit it, and the images left over got no detections. The val metrics then
depended on machine load. The limit is switched off for the process (`disable_nms_time_limit`).

How the run works:
1. Training: the val split picks `best.pt`.
2. `best.pt` is evaluated on the test split (in every run, including the smoke test).
3. Our callbacks write the TensorBoard log, and example predictions of `best.pt` are drawn
   for val and test.

Each run gets its own folder `runs/<timestamp>_<name>/`:

| File | |
|---|---|
| `events.out.tfevents.*`, `hparams/` | our TensorBoard log, see below |
| `config.yaml` | resolved config after CLI overrides |
| `metadata.json` | git commit + dirty flag, dataset sha256 and split sizes, weights sha256, seed, device, thresholds, Python/torch/Ultralytics versions, platform, command |
| `weights/best.pt`, `weights/last.pt` | checkpoints: `last.pt` after every epoch, `best.pt` when the val fitness (mAP50-95) improves |
| `test_metrics.json` | test P/R/mAP, best val metrics, per class: test objects, evaluated or not, AP |
| `predictions/{val,test}/` | example predictions of `best.pt` |
| `test/` | Ultralytics plots of the test evaluation (confusion matrices, PR curves) |
| `args.yaml`, `results.csv`, `*.png` | written by Ultralytics. It stores the absolute `project`/`save_dir` in `args.yaml` and the checkpoints; in the submitted archive they are replaced with relative ones |

Reproducibility: `seed: 42`, `deterministic: true`, `workers: 0`, and no NMS time limit, on CPU.
Two runs of the same commit give the same `results.csv` (only the time column differs) and the
same `test_metrics.json`. This was checked.

Ultralytics' own TensorBoard callback and other integrations (W&B, MLflow, ClearML, ...) are
switched off for the process, in memory only. The user's Ultralytics settings file is not
modified.

### 7. TensorBoard

```bash
uv run tensorboard --logdir runs     # or, with .venv activated: tensorboard --logdir runs
```

Open http://localhost:6006.

## TensorBoard logs

Written by `src/tensorboard_logger.py` through `torch.utils.tensorboard.SummaryWriter`. One step =
one epoch, starting at 1.

| Tab | Tags | When |
|---|---|---|
| Scalars | `loss/train_box`, `loss/train_cls`, `loss/train_dfl` | every epoch, mean over the epoch |
| | `loss/val_box`, `loss/val_cls`, `loss/val_dfl` | every epoch |
| | `val/precision`, `val/recall`, `val/mAP50`, `val/mAP50-95` | every epoch |
| | `lr/pg0`, `lr/pg1`, `lr/pg2` | every epoch, per parameter group |
| | `best_val/*` (same 4 metrics) | once: Ultralytics' final validation of `best.pt` |
| | `test/*` (same 4 metrics) | once: `best.pt` on the test split |
| | `hparam/val_*`, `hparam/test_*` (same 4 metrics) | once, step 0, written by the HParams entry below |
| Images | `val/confusion_matrix`, `test/confusion_matrix` | once, for `best.pt`, normalized by true class |
| | `predictions/val/<image>`, `predictions/test/<image>` | once, 4 images per split, `best.pt` at conf 0.25 |
| Text | `config`, `metadata` | at the start: resolved config and `metadata.json` |
| HParams | config (model, epochs, imgsz, batch, optimizer, lr0, freeze, seed, device) → best val and test metrics | once, at the end |

The final validation of `best.pt` is logged as `best_val/*`. It is not logged as an extra epoch:
Ultralytics calls the epoch callback once more after training, and the logger recognises that
call.

## Submitted run

`runs/` is not committed; only `runs/.gitkeep` is, as the task requires. The run described here is
handed in as an archive with this one run folder in it. It is attached to a [GitHub Release](https://github.com/lemind/sts-cv-test/releases/tag/run-2026-11-26T171458_yolo11n).
Download and unpack it in the repo root:

```bash
curl -LO https://github.com/lemind/sts-cv-test/releases/download/run-2026-11-26T171458_yolo11n/2026-11-26T171458_yolo11n.tar.gz
tar -xzf 2026-11-26T171458_yolo11n.tar.gz   # -> runs/2026-11-26T171458_yolo11n/
sha256sum -c runs/2026-11-26T171458_yolo11n/SHA256SUMS   # integrity of the archived files
uv run tensorboard --logdir runs
```

Run settings:
- **Command:** `python -m src.run_experiment --config configs/yolo.yaml`.
- **Model:** yolo11n, COCO-pretrained; the `person` weights are transferred to `pedestrian`.
- **Training:** 50 epochs, imgsz 640, batch 4, AdamW lr 0.0003, CPU, seed 42.
- **Versions:** `metadata.json` records commit `9cd04b2` (`dirty: false`) and dataset `52ba8e6a…`.

It is the same run as candidate C: rerun, it gave identical `results.csv` and `test_metrics.json`.

Val metrics during training (selected epochs):

| Epoch | train cls loss | val P | val R | val mAP50 | val mAP50-95 |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.225 | 0.375 | 0.394 | 0.206 | 0.140 |
| 10 | 1.143 | 0.158 | 0.524 | 0.233 | 0.146 |
| 20 | 0.893 | 0.336 | 0.314 | 0.342 | 0.244 |
| 30 | 0.798 | 0.433 | 0.493 | 0.411 | 0.293 |
| 37 (best.pt) | 0.685 | 0.480 | 0.435 | 0.473 | 0.336 |
| 50 | 0.653 | 0.348 | 0.352 | 0.353 | 0.198 |

`best.pt` is from epoch 37. On the test split (8 images, 218 boxes, evaluated once) it scores:

| P | R | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| 0.390 | 0.166 | 0.128 | 0.071 |

Per class (test mAP50):

| Class | mAP50 | Test boxes |
|---|---:|---:|
| car | 0.48 | 125 |
| police | 0.29 | 5 |
| pedestrian | 0.21 | 57 |
| truck | 0.15 | 14 |
| motorcycle | 0.02 | 9 |
| bus, lorry, trailer, ambulance | 0 | 1–3 each |

Bicycle and fire_truck are not in test.

What these numbers do and do not show:
- **Not a usable detector.** With 24 training images, classes with 1–5 boxes are not learned. The
  frequent classes (car, pedestrian) and police are learned noticeably.
- **Test is much lower than val.** Each split has 8 images from different cameras, and val was also
  used to pick `best.pt` and the settings, so val is optimistic. Test is a diagnostic estimate, also
  noisy.
- **Val drops after epoch 37.** That is overfitting; `best.pt` is taken at the peak.
- **Test is not untouched.** Settings were chosen on val only, but test metrics were also computed
  in diagnostic runs during development (for example 0.038 for the first 5-epoch run). So test here
  is a diagnostic estimate, not a fully independent holdout.

## Repository layout

```text
configs/yolo.yaml          experiment config
src/validate_labels.py     image and label checks (report only)
src/prelabel.py            COCO pre-labels + CVAT import zip
src/annotation_changes.py  annotations/changes.csv from source vs final labels
src/prepare_dataset.py     split, dataset.yaml, report, samples, manifest
src/run_experiment.py      dry-run / smoke test / train + test evaluation
src/tensorboard_logger.py  our SummaryWriter logging
src/utils.py               label parsing and shared helpers
annotations/               corrected labels, class rules, groups, audit trail
test_data/traffic_v5_mini/ source set, read-only
data/                      processed dataset (generated, gitignored), see data/README.md
reports/                   dataset report, split, samples, manifest
runs/                      experiment runs (gitignored)
tests/                     pytest
```

## Assumptions and limitations

- **Small data.** 40 images from 28 source groups, 8 images each in val and test. Metrics are
  a check that the pipeline works, not an estimate of generalization. Classes absent from a split
  (test: bicycle, fire_truck) are not evaluated there, which is not the same as AP = 0.
  `test_metrics.json` lists which classes were evaluated.
- **Rare classes.** `bicycle` exists only on COCO images, which are pinned to train. `lorry` has 2
  boxes in the whole set.
- **COCO overlap.** The 4 COCO train2017 images are kept in train, since the pretrained weights
  have seen them.
- **Pre-labels.** They are triage hints only. Every final box was reviewed by hand. The COCO
  model has no police, ambulance, fire_truck, lorry or trailer classes; those were labeled by hand
  following `annotations/README.md`.
- **Dense crowds.** People who can be told apart get a box. People merged into one blob do not,
  because YOLO has no ignore regions.
- **Tuned on val.** `lr0` and backbone freezing were chosen among 4 candidates on val (see
  [Train](#6-train)); everything else is Ultralytics defaults or fixes. Nothing was chosen on test,
  but test metrics are visible in diagnostic runs, so test is not a fully independent holdout. There
  is no baseline without fine-tuning (the same evaluation of the untrained COCO model), so no
  comparison with one is made.
- **Classifier head.** COCO classifier weights are transferred to 6 of 11 classes: bicycle, car,
  motorcycle, bus, truck by name, and pedestrian through `class_aliases`. COCO has no police,
  ambulance, fire_truck, lorry or trailer; those are learned from scratch.
- **CPU only.** The lock file pins CPU wheels of torch. On a CUDA machine, install a CUDA build of
  torch 2.15.0 and pass `--device 0`.
