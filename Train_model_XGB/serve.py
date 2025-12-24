"""
Serve new (clean) version for XGBoost pipeline with SHAP explainability
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Union, Dict, Any
import uvicorn
import os, pickle, json
import pandas as pd
import numpy as np


class PredictRequest(BaseModel):
    data: Union[Dict[str, Any], List[Dict[str, Any]]]
    explain: bool = False


app = FastAPI(title='XGBoost Readmission Model Server')

BASE = os.path.dirname(__file__)
MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join(BASE, 'outputs', 'readmission_tpe', 'best_readmission_xgb_tpe.pkl'))
FEATURES_PATH = os.environ.get('FEATURES_PATH', os.path.join(BASE, 'outputs', 'readmission_tpe', 'features.json'))
RESULTS_PATH = os.environ.get('RESULTS_PATH', os.path.join(BASE, 'outputs', 'readmission_tpe', 'readmission_tpe_results.json'))

ALLOWED_ORIGINS = [o for o in os.environ.get('CORS_ORIGINS', 'http://localhost:3000').split(',') if o]
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_credentials=True, allow_methods=['*'], allow_headers=['*'])


def _load_json(p):
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def _load_threshold(p):
    if os.path.exists(p):
        try:
            with open(p) as f:
                r = json.load(f)
            return float(r.get('best_threshold', 0.5))
        except Exception:
            return 0.5
    return 0.5


def coerce_and_prep(df: pd.DataFrame):
    if 'Y' in df.columns:
        df = df.drop(columns=['Y'])
    if 'GENDER' in df.columns:
        df['GENDER'] = df['GENDER'].apply(lambda v: 'M' if str(v).lower() in ['male', 'm'] else ('F' if str(v).lower() in ['female', 'f'] else v))
    return df


model = None
features = None
threshold = None


@app.on_event('startup')
def startup_event():
    global model, features, threshold
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError('Model path not found: ' + MODEL_PATH)
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    features = _load_json(FEATURES_PATH)
    threshold = _load_threshold(RESULTS_PATH)


@app.get('/health')
def health():
    return {'status': 'ok', 'model_loaded': model is not None}


@app.get('/features')
def get_feats():
    return {'features': features or [], 'count': len(features or [])}


@app.post('/predict')
def predict(req: PredictRequest):
    global model, features, threshold
    if model is None:
        raise HTTPException(status_code=503, detail='Model not loaded')
    data = req.data
    if isinstance(data, dict):
        data = [data]
    df = pd.DataFrame(data)
    if features is not None:
        missing = [f for f in features if f not in df.columns]
        for m in missing:
            df[m] = np.nan
        df = df[features].copy()
    df_prep = coerce_and_prep(df)
    try:
        proba = model.predict_proba(df_prep)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Prediction failed: {e}')
    labels = (proba >= threshold).astype(int)
    records = df.replace({np.nan: None}).to_dict(orient='records')
    resp = [{'index': i, 'input': r, 'pred_proba': float(proba[i]), 'pred_label': int(labels[i])} for i, r in enumerate(records)]

    if req.explain:
        try:
            import shap
        except Exception:
            return {'predictions': resp, 'threshold': threshold, 'explain_error': 'shap_not_installed'}
        shap_info = None
        try:
            if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps and 'clf' in model.named_steps:
                pre = model.named_steps['preprocessor']
                clf = model.named_steps['clf']
                try:
                    X_trans = pre.transform(df_prep)
                except Exception:
                    X_trans = pre.fit_transform(df_prep)
                try:
                    feat_names = list(pre.get_feature_names_out(df_prep.columns))
                except Exception:
                    feat_names = list(df_prep.columns)
                try:
                    explainer = shap.TreeExplainer(clf)
                    shap_vals = explainer.shap_values(X_trans)
                except Exception:
                    explainer = shap.Explainer(clf.predict_proba, X_trans)
                    shap_res = explainer(X_trans)
                    shap_vals = shap_res.values
                arr = np.array(shap_vals)
                if arr.ndim == 3:
                    if arr.shape[1] == 2:
                        arr = arr[:, 1, :]
                    elif arr.shape[2] == 2:
                        arr = arr[:, :, 1]
                shap_info = []
                for i_row, svals in enumerate(arr):
                    agg = {}
                    for fname, sval in zip(feat_names, svals):
                        base = str(fname).split('__')[-1] if '__' in str(fname) else str(fname)
                        try:
                            contribution = float(sval)
                        except Exception:
                            svalarr = np.asarray(sval)
                            contribution = float(svalarr.sum()) if svalarr.size > 0 else 0.0
                        val = resp[i_row]['input'].get(base) if base in resp[i_row]['input'] else None
                        agg.setdefault(base, {'feature': base, 'value': val, 'contribution': 0.0})
                        agg[base]['contribution'] += contribution
                    factors = [{'feature': v['feature'], 'value': v['value'], 'contribution': float(v['contribution']), 'direction': 'increase' if v['contribution'] > 0 else 'decrease'} for v in agg.values()]
                    factors = sorted(factors, key=lambda x: abs(x['contribution']), reverse=True)[:50]
                    shap_info.append(factors)
            else:
                explainer = shap.Explainer(model.predict_proba, df_prep)
                shap_res = explainer(df_prep)
                arr = np.array(shap_res.values)
                if arr.ndim == 3:
                    if arr.shape[1] == 2:
                        arr = arr[:, 1, :]
                    elif arr.shape[2] == 2:
                        arr = arr[:, :, 1]
                feat_names = list(df_prep.columns)
                shap_info = []
                for i_row, svals in enumerate(arr):
                    factors = []
                    for fname, sval in zip(feat_names, svals):
                        try:
                            contribution = float(sval)
                        except Exception:
                            svalarr = np.asarray(sval)
                            contribution = float(svalarr.sum()) if svalarr.size > 0 else 0.0
                        val = resp[i_row]['input'].get(fname) if fname in resp[i_row]['input'] else None
                        factors.append({'feature': fname, 'value': val, 'contribution': contribution, 'direction': 'increase' if contribution > 0 else 'decrease'})
                    factors = sorted(factors, key=lambda x: abs(x['contribution']), reverse=True)[:50]
                    shap_info.append(factors)
        except Exception as e:
            return {'predictions': resp, 'threshold': threshold, 'explain_error': str(e)}
        if shap_info is not None and len(shap_info) == len(resp):
            for i_row in range(len(resp)):
                resp[i_row]['shapFactors'] = shap_info[i_row]
    return {'predictions': resp, 'threshold': threshold}


if __name__ == '__main__':
    uvicorn.run('Train_model.serve_new:app', host='0.0.0.0', port=8001, reload=True)
