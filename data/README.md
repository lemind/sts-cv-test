# data/

| Folder | What it is | In git |
|---|---|---|
| `raw/` | Place for an external full dataset. Unused by this task | `.gitkeep` only |
| `processed/` | The prepared YOLO dataset: `images/{train,val,test}/`, `labels/{train,val,test}/`, `dataset.yaml` | `.gitkeep` only |

## Rebuild `processed/`

`processed/` is generated from the committed source set and the corrected labels; it is never
edited by hand. From the repo root:

```bash
uv run python -m src.prepare_dataset --input test_data/traffic_v5_mini --output data/processed --seed 42
```

Inputs: `test_data/traffic_v5_mini/` (images, read-only), `annotations/final/` (labels),
`annotations/source_groups.csv` (split groups). The run also rewrites `reports/` (report, split
manifest, samples, `manifest.sha256`).

## Check it against the manifest

`reports/manifest.sha256` lists the sha256 of every file in `processed/` (paths relative to the
repo root, `sha256sum -c` format). Its own sha256 is the dataset version recorded in every
experiment run (`metadata.json`).

```bash
sha256sum -c --quiet reports/manifest.sha256   # silent when all files match
git diff --exit-code reports/                  # rebuild gives byte-identical reports
```

`src.run_experiment` checks the same manifest before training (also in `--dry-run`) and stops if a
file differs, is missing, or is not listed. Ultralytics' `labels/*.cache` files are ignored.

`dataset.yaml` has no `path:` key: Ultralytics resolves `images/...` relative to the yaml file, so
the dataset works from any working directory and contains no absolute paths.
