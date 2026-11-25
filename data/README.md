# data/

- `raw/` — place for an external full dataset (not committed).
- `processed/` — output of `python -m src.prepare_dataset` (not committed). Rebuild it from
  `test_data/traffic_v5_mini/` and verify against `reports/manifest.sha256`.
