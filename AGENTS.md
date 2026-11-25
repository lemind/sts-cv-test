# AGENTS.md — sts-cv-test

Test task: dataset preparation + a reproducible Ultralytics YOLO experiment logged to TensorBoard.
The task statement is the source of truth; the working plan lives in `PLAN.md` (local, not committed).

## Philosophy

- Prefer KISS over DRY.
- Duplication is acceptable if abstraction harms readability.
- Avoid abstractions before the third real use case.
- Prefer boring, maintainable solutions.
- Explicit code over clever code.
- Measure before optimizing.

## Communication

- Ask clarifying questions before acting on ambiguous requirements.
- State assumptions explicitly when information is missing.
- Propose concrete next steps, not general suggestions.

## Project non-negotiables

- **Source data is read-only**: never modify `test_data/traffic_v5_mini/`. Every fix goes to `data/processed/`
  and is recorded in `reports/dataset_report.md` (what changed and why).
- **No absolute paths** anywhere: code, configs, reports, `dataset.yaml`. Paths are relative to the repo root.
- **Reproducible**: every random step takes `--seed`; the same input and seed give byte-identical output
  (checked by `reports/manifest.sha256`).
- **Missing `.txt` is not "background"**: an image without labels is either labeled or excluded and listed,
  never silently used as empty.
- **Class IDs follow `classes.txt`** line order (0-based). After any CVAT round-trip, re-run the validator
  and compare class IDs.
- **Split by source group**, not by random frame: near-identical frames of one video must not land in
  both train and test.
- Never commit weights, run logs, the processed dataset or secrets.

## Code Style & Consistency

- Python, formatted and linted with `ruff`. Type hints on public functions.
- Do not disable linters or skip formatting to "fix" a bug.
- **Naming**: descriptive names that make purpose obvious (`find_out_of_bounds_boxes`, `split_by_group`).
  Avoid generic names like `data`, `result`, `check`, `process`. Don't over-verbose.
- CLI entry points use `argparse` and run as `python -m src.<module>`.

## Git & Version Control

- Commit atomically: one logical change per commit.
- Message format from the task: `feat: add dataset validation`, `docs: add run instructions`,
  `fix: …`, `test: …`, `chore: …`. Short, one line; details go in the body.
- Never commit without explicit user request; show a summary of changes first.
- Never force-push or rewrite history without explicit approval.

## Error Handling & Validation

- Validate at boundaries: image files, label files, CLI args, config.
- Never `catch` and ignore; log context or rethrow. A broken file is reported, not skipped silently.

## Testing

- `pytest`. Focus on what catches real bugs: label parsing and validation (bad class, out-of-bounds,
  malformed lines, CRLF), split determinism and group leakage.
- Don't pad coverage on trivial code.

## Plan (keep in sync)

- `PLAN.md` holds stages, status and findings. After finishing a step, mark it `[x]` and add anything
  discovered to the findings section.
- Once a stage is marked done (✅), don't add new items to it; new work goes to the current stage.

## Scope Discipline

**Do only what was explicitly asked. Everything else is out of scope.**

- If asked to update docs, don't touch code. If asked to fix a bug, don't refactor around it.
- When in doubt whether an action is in scope: **don't do it, ask first.**

## Autonomy

### Act without asking
- Fix typos, lint errors, or obvious bugs.
- Add missing error handling.
- Improve tests within the same module.

### Ask before acting
- Changes affecting >3 files or >2 modules.
- Adding/removing dependencies or changing pinned versions.
- Deleting or relabeling annotations (each such decision goes into the report).
- Long training runs.
- Committing.
- Any work beyond the explicitly stated task.

## Forbidden

- Premature abstractions.
- Global state unless justified.
- Silent failures.
- Hidden magic behavior.
- Modifying source data in place.
- Committing secrets, weights, logs or `.env` files.
