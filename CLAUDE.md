@AGENTS.md

## Context

- Task: CV test task for STS. Graded on data preparation, engineering care, reproducibility and experiment
  structure — **not** on mAP.
- Read `PLAN.md` first (local, gitignored): current stage, findings in the data, open questions.
- Input: `test_data/traffic_v5_mini/` (40 images, 11 classes in `classes.txt`). Output: `data/processed/`,
  `reports/`, `runs/`.
- Commands (once implemented):
  - `python -m src.prepare_dataset --input test_data/traffic_v5_mini --output data/processed --seed 42`
  - `python -m src.run_experiment --config configs/yolo.yaml --data data/processed/dataset.yaml --dry-run`
  - `tensorboard --logdir runs`
