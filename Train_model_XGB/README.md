# Model Serving (FastAPI)

This folder contains scripts to run and serve the XGBoost model trained by `xgboost_tpe.py`.

Prerequisites
```
pip install -r requirements.txt
# or for the server specifically
pip install fastapi uvicorn
```

Files
- `serve.py` - FastAPI web server exposing `/predict` and `/health` endpoints.
 - `serve.py` - FastAPI web server exposing `/predict` and `/health` endpoints. `predict` now supports explainability via SHAP when requested (add `"explain": true` in JSON body).
- `convert_joblib_to_pkl.py` - utility to convert joblib model to pickle (.pkl).
- `inference.py` - simple CLI for batch predictions using a .pkl model.
 - `inference.py` - simple CLI for batch predictions using a .pkl model (the .pkl is a sklearn Pipeline including preprocessing + XGBoost classifier).

How to run
1) Ensure a model and artifacts exist in `Train_model/outputs/readmission_tpe` (e.g., `best_readmission_xgb_tpe.pkl`, `features.json`, `readmission_tpe_results.json`)
2) Start the server:
```
export MODEL_PATH=Train_model/outputs/readmission_tpe/best_readmission_xgb_tpe.pkl
export FEATURES_PATH=Train_model/outputs/readmission_tpe/features.json
export RESULTS_PATH=Train_model/outputs/readmission_tpe/readmission_tpe_results.json
uvicorn Train_model.serve:app --reload --port 8000
```
```
Cleanup note
------------
You can optionally run the training script with a cleanup flag to list (dry-run) or delete model artifacts that are not referenced in `metrics_filtered_ge0_5.csv`:

```
# Dry-run to show candidates
python Train_model/xgboost_tpe.py --cleanup

# Force-delete (dangerous)
python Train_model/xgboost_tpe.py --cleanup-force
```
Additional cleanup options:

- `--cleanup-threshold <float>` (default 0.5) — minimum value for precision, recall and F1 to keep a run
- `--cleanup-strict` — use strict `>` instead of `>=` when comparing to the threshold
- `--cleanup-remove-run-dirs` — remove entire run directory `Train_model/outputs/<run>` if it does not meet threshold


Backend CORS and Frontend connections
```
# If your frontend runs on a non-default origin (e.g., http://localhost:3000), you can set:
export CORS_ORIGINS=http://localhost:3000
# By default, server allows http://localhost:3000.
```

3) Make a request (example with `curl`):
```
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"data": {"GENDER": "M", "feature1": 1.0, ..., "feature58": 0.0}}'
```

To request SHAP explainability, pass `"explain": true` in the request body (note: this requires the `shap` package installed in the environment):

```
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"data": {"GENDER": "M", "feature1": 1.0, ..., "feature58": 0.0}, "explain": true}'
```

Notes
- The server validates feature presence if `features.json` exists. Make sure input JSON exact feature names and types.
- Don't use pickle files from untrusted sources.
- For production, containerize and secure the app (TLS, auth), and consider saving preprocessing pipeline together with model.

Training / saving a model as a pickle (`.pkl`)
```
# Example: run the TPE-based XGBoost optimization and save a pkl model for the app
python Train_model/xgboost_tpe.py --input cohort/new_cohort_icu_readmission.csv --outdir Train_model/outputs/readmission_tpe --model-name best_readmission_xgb_tpe

# This creates the model at Train_model/outputs/readmission_tpe/best_readmission_xgb_tpe.pkl (and joblib / json equivalents). The .pkl is a sklearn Pipeline (preprocessor + XGBClassifier) so the server's `MODEL_PATH` can point directly to it.
```
