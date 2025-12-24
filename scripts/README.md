# Cleanup models utility

This folder contains `cleanup_models.py`, a utility to remove model artifacts under `outputs/` that are not referenced in a metrics CSV.

Usage examples:

- Dry-run: list candidate model files to delete
```
python scripts/cleanup_models.py --metrics-csv metrics_filtered_ge0_5.csv --base-dir . --dry-run
```

- Actually delete:
```
python scripts/cleanup_models.py --metrics-csv metrics_filtered_ge0_5.csv --base-dir . --force
```

Additional options:

- `--threshold <float>` default 0.5 — require precision/recall/f1 to be >= threshold (use `--strict` for >)
- `--strict` — use strict `>` comparison instead of >=
- `--remove-run-dirs` — remove whole run directories like `Train_model_XGB/outputs/<run>` that don't meet the threshold


You can also call `cleanup` from training scripts by passing the flags `--cleanup` or `--cleanup-force` to optionally list and remove models after training.
