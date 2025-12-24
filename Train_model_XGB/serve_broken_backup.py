"""
Serve the model with SHAP explainability (simple, tested implementation)
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union, Dict, Any
import uvicorn, os, pickle, json
import pandas as pd
import numpy as np


class PredictRequest(BaseModel):
    data: Union[Dict[str, Any], List[Dict[str, Any]]]
    explain: bool = False


def _load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _load_threshold(path):
    if os.path.exists(path):
        try:
            with open(path) as f:
                r = json.load(f)
            return float(r.get('best_threshold', 0.5))
        except Exception:
            return 0.5
    return 0.5


def normalize_inputs(df: pd.DataFrame):
    if 'Y' in df.columns:
        df = df.drop(columns=['Y'])
    if 'GENDER' in df.columns:
        df['GENDER'] = df['GENDER'].apply(lambda v: 'M' if str(v).lower() in ['male', 'm'] else ('F' if str(v).lower() in ['female', 'f'] else v))
    return df


app = FastAPI(title='Readmission Model Server')

BASE = os.path.dirname(__file__)
MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join(BASE, 'outputs', 'readmission_tpe', 'best_readmission_xgb_tpe.pkl'))
FEATURES_PATH = os.environ.get('FEATURES_PATH', os.path.join(BASE, 'outputs', 'readmission_tpe', 'features.json'))
RESULTS_PATH = os.environ.get('RESULTS_PATH', os.path.join(BASE, 'outputs', 'readmission_tpe', 'readmission_tpe_results.json'))

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
        for m in missing: df[m] = np.nan
        df = df[features].copy()
    df_prep = normalize_inputs(df)
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
                            svalarr=np.asarray(sval)
                            contribution = float(svalarr.sum()) if svalarr.size>0 else 0.0
                        val = resp[i_row]['input'].get(base) if base in resp[i_row]['input'] else None
                        agg.setdefault(base, {'feature': base, 'value': val, 'contribution': 0.0})
                        agg[base]['contribution'] += contribution
                    factors = [{'feature': v['feature'], 'value': v['value'], 'contribution': float(v['contribution']), 'direction': 'increase' if v['contribution']>0 else 'decrease'} for v in agg.values()]
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
                    factors=[]
                    for fname, sval in zip(feat_names, svals):
                        try:
                            contribution = float(sval)
                        except Exception:
                            svalarr=np.asarray(sval)
                            contribution = float(svalarr.sum()) if svalarr.size>0 else 0.0
                        val = resp[i_row]['input'].get(fname) if fname in resp[i_row]['input'] else None
                        factors.append({'feature': fname, 'value': val, 'contribution': contribution, 'direction': 'increase' if contribution>0 else 'decrease'})
                    factors = sorted(factors, key=lambda x: abs(x['contribution']), reverse=True)[:50]
                    shap_info.append(factors)
        except Exception as e:
            return {'predictions': resp, 'threshold': threshold, 'explain_error': str(e)}
        if shap_info is not None and len(shap_info) == len(resp):
            for i_row in range(len(resp)):
                resp[i_row]['shapFactors'] = shap_info[i_row]
    return {'predictions': resp, 'threshold': threshold}


if __name__ == '__main__':
    uvicorn.run('Train_model.serve:app', host='0.0.0.0', port=8000, reload=True)
"""
FastAPI server for serving the XGBoost Pipeline with optional SHAP explainability.

Endpoints:
- GET /health
- GET /features
- POST /predict  (body: {"data": <dict | list>, "explain": <bool>})

This is a simplified and robust implementation to return predictions and
optionally aggregated SHAP factors mapped back to input feature names.
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

model = None
features = None
threshold = None

# CORS
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


@app.on_event('startup')
def startup():
    global model, features, threshold
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f'Model not found at {MODEL_PATH}')
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    features = _load_json(FEATURES_PATH)
    threshold = _load_threshold(RESULTS_PATH)


@app.get('/health')
def get_health():
    return {'status': 'ok', 'model_loaded': model is not None}


@app.get('/features')
def get_features():
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
    df_json = df.replace({np.nan: None}).to_dict(orient='records')
    resp = []
    for i, row in enumerate(df_json):
        resp.append({'index': i, 'input': row, 'pred_proba': float(proba[i]), 'pred_label': int(labels[i])})
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
                shap_arr = np.array(shap_vals)
                if shap_arr.ndim == 3:
                    if shap_arr.shape[1] == 2:
                        shap_arr = shap_arr[:, 1, :]
                    elif shap_arr.shape[2] == 2:
                        shap_arr = shap_arr[:, :, 1]
                try:
                    if np.allclose(shap_arr, 0.0):
                        explainer = shap.Explainer(clf.predict_proba, X_trans)
                        shap_res = explainer(X_trans)
                        shap_arr = np.array(shap_res.values)
                        if shap_arr.ndim == 3:
                            if shap_arr.shape[1] == 2:
                                shap_arr = shap_arr[:, 1, :]
                            elif shap_arr.shape[2] == 2:
                                shap_arr = shap_arr[:, :, 1]
                except Exception:
                    pass
                shap_info = []
                for i_row, svals in enumerate(shap_arr):
                    row_vals = resp[i_row]['input'] if i_row < len(resp) else None
                    agg = {}
                    s = np.asarray(svals)
                    if s.ndim == 1:
                        iter_vals = zip(feat_names, s)
                    else:
                        iter_vals = zip(feat_names, s.reshape(s.shape[-1],))
                    for fname, sval in iter_vals:
                        base_name = str(fname).split('__')[-1] if '__' in str(fname) else str(fname)
                        val = row_vals.get(base_name) if row_vals and base_name in row_vals else None
                        try:
                            contribution = float(sval)
                        except Exception:
                            sval_a = np.asarray(sval)
                            contribution = float(sval_a.sum()) if sval_a.size > 0 else 0.0
                        if base_name not in agg:
                            agg[base_name] = {'feature': base_name, 'value': val, 'contribution': 0.0}
                        agg[base_name]['contribution'] += contribution
                    factors = []
                    for k, v in agg.items():
                        factors.append({'feature': v['feature'], 'value': v['value'], 'contribution': float(v['contribution']), 'direction': 'increase' if v['contribution'] > 0 else 'decrease'})
                    factors = sorted(factors, key=lambda x: abs(x['contribution']), reverse=True)[:50]
                    shap_info.append(factors)
            else:
                explainer = shap.Explainer(model.predict_proba, df_prep)
                shap_res = explainer(df_prep)
                shap_arr = np.array(shap_res.values)
                if shap_arr.ndim == 3:
                    if shap_arr.shape[1] == 2:
                        shap_arr = shap_arr[:, 1, :]
                    elif shap_arr.shape[2] == 2:
                        shap_arr = shap_arr[:, :, 1]
                feat_names = list(df_prep.columns)
                shap_info = []
                for i_row, svals in enumerate(shap_arr):
                    row_vals = resp[i_row]['input'] if i_row < len(resp) else None
                    factors = []
                    for fname, sval in zip(feat_names, svals):
                        try:
                            contribution = float(sval)
                        except Exception:
                            svalarr = np.asarray(sval)
                            contribution = float(svalarr.sum()) if svalarr.size > 0 else 0.0
                        val = row_vals.get(fname) if row_vals and fname in row_vals else None
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
    uvicorn.run('Train_model.serve:app', host='0.0.0.0', port=8000, reload=True)
"""
FastAPI server for serving the XGBoost Pipeline with optional SHAP explainability.

Endpoints:
- GET /health
- GET /features
- POST /predict  (body: {"data": <dict | list>, "explain": <bool>})

This is a simplified and robust implementation to return predictions and
optionally aggregated SHAP factors mapped back to input feature names.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union, Dict, Any
import uvicorn
import os, pickle, json
import pandas as pd
import numpy as np


class PredictRequest(BaseModel):
    data: Union[Dict[str, Any], List[Dict[str, Any]]]
    explain: bool = False


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


app = FastAPI(title='XGBoost Readmission Model Server')

BASE = os.path.dirname(__file__)
MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join(BASE, 'outputs', 'readmission_tpe', 'best_readmission_xgb_tpe.pkl'))
FEATURES_PATH = os.environ.get('FEATURES_PATH', os.path.join(BASE, 'outputs', 'readmission_tpe', 'features.json'))
RESULTS_PATH = os.environ.get('RESULTS_PATH', os.path.join(BASE, 'outputs', 'readmission_tpe', 'readmission_tpe_results.json'))

model = None
features = None
threshold = None

# CORS
ALLOWED_ORIGINS = [o for o in os.environ.get('CORS_ORIGINS', 'http://localhost:3000').split(',') if o]
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_credentials=True, allow_methods=['*'], allow_headers=['*'])


@app.on_event('startup')
def startup():
    global model, features, threshold
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f'Model not found at {MODEL_PATH}')
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    features = _load_json(FEATURES_PATH)
    threshold = _load_threshold(RESULTS_PATH)


@app.get('/health')
def get_health():
    return {'status': 'ok', 'model_loaded': model is not None}


@app.get('/features')
def get_features():
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
    df_json = df.replace({np.nan: None}).to_dict(orient='records')
    resp = []
    for i, row in enumerate(df_json):
        resp.append({'index': i, 'input': row, 'pred_proba': float(proba[i]), 'pred_label': int(labels[i])})

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
                shap_arr = np.array(shap_vals)
                if shap_arr.ndim == 3:
                    if shap_arr.shape[1] == 2:
                        shap_arr = shap_arr[:, 1, :]
                    elif shap_arr.shape[2] == 2:
                        shap_arr = shap_arr[:, :, 1]
                try:
                    if np.allclose(shap_arr, 0.0):
                        explainer = shap.Explainer(clf.predict_proba, X_trans)
                        shap_res = explainer(X_trans)
                        shap_arr = np.array(shap_res.values)
                        if shap_arr.ndim == 3:
                            if shap_arr.shape[1] == 2:
                                shap_arr = shap_arr[:, 1, :]
                            elif shap_arr.shape[2] == 2:
                                shap_arr = shap_arr[:, :, 1]
                except Exception:
                    pass
                shap_info = []
                for i_row, svals in enumerate(shap_arr):
                    row_vals = resp[i_row]['input'] if i_row < len(resp) else None
                    agg = {}
                    s = np.asarray(svals)
                    if s.ndim == 1:
                        iter_vals = zip(feat_names, s)
                    else:
                        iter_vals = zip(feat_names, s.reshape(s.shape[-1],))
                    for fname, sval in iter_vals:
                        base_name = str(fname).split('__')[-1] if '__' in str(fname) else str(fname)
                        val = row_vals.get(base_name) if row_vals and base_name in row_vals else None
                        try:
                            contribution = float(sval)
                        except Exception:
                            sval_a = np.asarray(sval)
                            contribution = float(sval_a.sum()) if sval_a.size > 0 else 0.0
                        if base_name not in agg:
                            agg[base_name] = {'feature': base_name, 'value': val, 'contribution': 0.0}
                        agg[base_name]['contribution'] += contribution
                    factors = []
                    for k, v in agg.items():
                        factors.append({'feature': v['feature'], 'value': v['value'], 'contribution': float(v['contribution']), 'direction': 'increase' if v['contribution'] > 0 else 'decrease'})
                    factors = sorted(factors, key=lambda x: abs(x['contribution']), reverse=True)[:50]
                    shap_info.append(factors)
            else:
                explainer = shap.Explainer(model.predict_proba, df_prep)
                shap_res = explainer(df_prep)
                shap_arr = np.array(shap_res.values)
                if shap_arr.ndim == 3:
                    if shap_arr.shape[1] == 2:
                        shap_arr = shap_arr[:, 1, :]
                    elif shap_arr.shape[2] == 2:
                        shap_arr = shap_arr[:, :, 1]
                feat_names = list(df_prep.columns)
                shap_info = []
                for i_row, svals in enumerate(shap_arr):
                    row_vals = resp[i_row]['input'] if i_row < len(resp) else None
                    factors = []
                    for fname, sval in zip(feat_names, svals):
                        try:
                            contribution = float(sval)
                        except Exception:
                            svalarr = np.asarray(sval)
                            contribution = float(svalarr.sum()) if svalarr.size > 0 else 0.0
                        val = row_vals.get(fname) if row_vals and fname in row_vals else None
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
    uvicorn.run('Train_model.serve:app', host='0.0.0.0', port=8000, reload=True)
"""
FastAPI server for serving the XGBoost Pipeline with optional SHAP explainability.

Endpoints:
- GET /health
- GET /features
- POST /predict  (body: {"data": <dict | list>, "explain": <bool>})

This is a simplified and robust implementation to return predictions and
optionally aggregated SHAP factors mapped back to input feature names.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Union, Dict, Any
import uvicorn
import os
import pickle
import json
import pandas as pd
import numpy as np


class PredictRequest(BaseModel):
    data: Union[Dict[str, Any], List[Dict[str, Any]]]
    explain: bool = False


def _load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _load_threshold(path):
    if os.path.exists(path):
        try:
            with open(path) as f:
                r = json.load(f)
            return float(r.get('best_threshold', 0.5))
        except Exception:
            return 0.5
    return 0.5


def coerce_and_prep(df: pd.DataFrame):
    # Keep inputs minimal; drop label column if present and normalize gender
    if 'Y' in df.columns:
        df = df.drop(columns=['Y'])
    if 'GENDER' in df.columns:
        df['GENDER'] = df['GENDER'].apply(lambda v: 'M' if str(v).lower() in ['male', 'm'] else ('F' if str(v).lower() in ['female', 'f'] else v))
    return df


app = FastAPI(title='XGBoost Readmission Model Server')

BASE = os.path.dirname(__file__)
MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join(BASE, 'outputs', 'readmission_tpe', 'best_readmission_xgb_tpe.pkl'))
FEATURES_PATH = os.environ.get('FEATURES_PATH', os.path.join(BASE, 'outputs', 'readmission_tpe', 'features.json'))
RESULTS_PATH = os.environ.get('RESULTS_PATH', os.path.join(BASE, 'outputs', 'readmission_tpe', 'readmission_tpe_results.json'))

model = None
features = None
threshold = None

# CORS
ALLOWED_ORIGINS = [o for o in os.environ.get('CORS_ORIGINS', 'http://localhost:3000').split(',') if o]
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_credentials=True, allow_methods=['*'], allow_headers=['*'])


@app.on_event('startup')
def startup():
    global model, features, threshold
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f'Model not found at {MODEL_PATH}')
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    features = _load_json(FEATURES_PATH)
    threshold = _load_threshold(RESULTS_PATH)


@app.get('/health')
def get_health():
    return {'status': 'ok', 'model_loaded': model is not None}


@app.get('/features')
def get_features():
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
        # ensure columns exist
        missing = [f for f in features if f not in df.columns]
        for m in missing:
            df[m] = np.nan
        df = df[features].copy()

    df_prep = coerce_and_prep(df)
    # predictions
    try:
        proba = model.predict_proba(df_prep)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Prediction failed: {e}')
    labels = (proba >= threshold).astype(int)
    df_json = df.replace({np.nan: None}).to_dict(orient='records')
    resp = []
    for i, row in enumerate(df_json):
        resp.append({'index': i, 'input': row, 'pred_proba': float(proba[i]), 'pred_label': int(labels[i])})

    if req.explain:
        try:
            import shap
        except Exception:
            return {'predictions': resp, 'threshold': threshold, 'explain_error': 'shap_not_installed'}
        shap_info = None
        try:
            # If pipeline-like object exists, try using preprocessor + clf
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
                # Explainer fallback logic
                try:
                    explainer = shap.TreeExplainer(clf)
                    shap_vals = explainer.shap_values(X_trans)
                except Exception:
                    explainer = shap.Explainer(clf.predict_proba, X_trans)
                    shap_res = explainer(X_trans)
                    shap_vals = shap_res.values
                shap_arr = np.array(shap_vals)
                # normalize to (n_samples, n_features)
                if shap_arr.ndim == 3:
                    if shap_arr.shape[1] == 2:
                        shap_arr = shap_arr[:, 1, :]
                    elif shap_arr.shape[2] == 2:
                        shap_arr = shap_arr[:, :, 1]
                # fallback if zeros
                try:
                    if np.allclose(shap_arr, 0.0):
                        explainer = shap.Explainer(clf.predict_proba, X_trans)
                        shap_res = explainer(X_trans)
                        shap_arr = np.array(shap_res.values)
                        if shap_arr.ndim == 3:
                            if shap_arr.shape[1] == 2:
                                shap_arr = shap_arr[:, 1, :]
                            elif shap_arr.shape[2] == 2:
                                shap_arr = shap_arr[:, :, 1]
                except Exception:
                    pass
                # aggregate by base name
                shap_info = []
                for i_row, svals in enumerate(shap_arr):
                    row_vals = resp[i_row]['input'] if i_row < len(resp) else None
                    agg = {}
                    s = np.asarray(svals)
                    if s.ndim == 1:
                        iter_vals = zip(feat_names, s)
                    else:
                        iter_vals = zip(feat_names, s.reshape(s.shape[-1],))
                    for fname, sval in iter_vals:
                        base_name = str(fname).split('__')[-1] if '__' in str(fname) else str(fname)
                        val = row_vals.get(base_name) if row_vals and base_name in row_vals else None
                        try:
                            contribution = float(sval)
                        except Exception:
                            sval_a = np.asarray(sval)
                            contribution = float(sval_a.sum()) if sval_a.size > 0 else 0.0
                        if base_name not in agg:
                            agg[base_name] = {'feature': base_name, 'value': val, 'contribution': 0.0}
                        agg[base_name]['contribution'] += contribution
                    factors = []
                    for k, v in agg.items():
                        factors.append({'feature': v['feature'], 'value': v['value'], 'contribution': float(v['contribution']), 'direction': 'increase' if v['contribution'] > 0 else 'decrease'})
                    factors = sorted(factors, key=lambda x: abs(x['contribution']), reverse=True)[:50]
                    shap_info.append(factors)
            else:
                # fallback directly on raw features
                try:
                    explainer = shap.Explainer(model.predict_proba, df_prep)
                    shap_res = explainer(df_prep)
                    shap_arr = np.array(shap_res.values)
                    if shap_arr.ndim == 3:
                        if shap_arr.shape[1] == 2:
                            shap_arr = shap_arr[:, 1, :]
                        elif shap_arr.shape[2] == 2:
                            shap_arr = shap_arr[:, :, 1]
                    feat_names = list(df_prep.columns)
                    shap_info = []
                    for i_row, svals in enumerate(shap_arr):
                        row_vals = resp[i_row]['input'] if i_row < len(resp) else None
                        factors = []
                        for fname, sval in zip(feat_names, svals):
                            try:
                                contribution = float(sval)
                            except Exception:
                                valarr = np.asarray(sval)
                                contribution = float(valarr.sum()) if valarr.size > 0 else 0.0
                            val = row_vals.get(fname) if row_vals and fname in row_vals else None
                            factors.append({'feature': fname, 'value': val, 'contribution': contribution, 'direction': 'increase' if contribution > 0 else 'decrease'})
                        factors = sorted(factors, key=lambda x: abs(x['contribution']), reverse=True)[:50]
                        shap_info.append(factors)
                except Exception:
                    shap_info = None
        except Exception as e:
            return {'predictions': resp, 'threshold': threshold, 'explain_error': str(e)}
        if shap_info is not None and len(shap_info) == len(resp):
            for i_row in range(len(resp)):
                resp[i_row]['shapFactors'] = shap_info[i_row]

    return {'predictions': resp, 'threshold': threshold}


if __name__ == '__main__':
    uvicorn.run('Train_model.serve:app', host='0.0.0.0', port=8000, reload=True)
"""
Simple FastAPI backend to serve the XGBoost model built by `xgboost_tpe.py`.
The server expects `MODEL_PATH` to point to a sklearn Pipeline pkl containing preprocessing + classifier, or a classifier object.

Endpoints:
  GET /health - returns 200 if model loaded
  POST /predict - accepts JSON body containing either a single sample or a list of samples

Input format:
  { "data": { "feature1": 1, "feature2": 0, ... } }
  or
  { "data": [ {..}, {..} ] }

Output:
  returns probabilities and labels per sample and optional `shapFactors` when request asks for explain.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Union, Dict, Any
import uvicorn
import os
import pickle
import json
import site
import sys
import pandas as pd
import numpy as np


class PredictRequest(BaseModel):
    data: Union[Dict[str, Any], List[Dict[str, Any]]]
    explain: bool = False


def load_features(features_path: str):
    if os.path.exists(features_path):
        with open(features_path) as f:
            return json.load(f)
    return None


def load_threshold(results_path: str):
    if os.path.exists(results_path):
        try:
            with open(results_path) as f:
                res = json.load(f)
            return float(res.get('best_threshold', 0.5))
        except Exception:
            return 0.5
    return 0.5


def coerce_and_prep(df: pd.DataFrame):
    # Preprocessing is handled by the saved pipeline. Avoid altering input features here.
    # Normalize some fields that may differ between UI and training data.
    # drop Y if present
    if 'Y' in df.columns:
        df = df.drop(columns=['Y'])
    # Normalize gender variations (e.g., 'male'/'female' to 'M'/'F') if provided
    if 'GENDER' in df.columns:
        df['GENDER'] = df['GENDER'].apply(lambda v: 'M' if str(v).lower() in ['male', 'm'] else ('F' if str(v).lower() in ['female', 'f'] else v))
    return df


app = FastAPI(title='XGBoost Readmission Model Server')

MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join(os.path.dirname(__file__), 'outputs', 'readmission_tpe', 'best_readmission_xgb_tpe.pkl'))
FEATURES_PATH = os.environ.get('FEATURES_PATH', os.path.join(os.path.dirname(__file__), 'outputs', 'readmission_tpe', 'features.json'))
RESULTS_PATH = os.environ.get('RESULTS_PATH', os.path.join(os.path.dirname(__file__), 'outputs', 'readmission_tpe', 'readmission_tpe_results.json'))

model = None
expected_features = None
threshold = None

# CORS: allow origins in env var CORS_ORIGINS (comma-separated). Default includes localhost:3000 for frontend dev
ALLOWED_ORIGINS = [o for o in os.environ.get('CORS_ORIGINS', 'http://localhost:3000').split(',') if o]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event('startup')
def startup_event():
    global model, expected_features, threshold
    # ensure installed xgboost is loaded to avoid local module name shadowing
    site_dirs = []
    try:
        site_dirs.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        site_dirs.append(site.getusersitepackages())
    except Exception:
        pass
    site_dirs = [p for p in site_dirs if p]
    if site_dirs:
        found_path = None
        for sd in site_dirs:
            candidate = os.path.join(sd, 'xgboost')
            init_py = os.path.join(candidate, '__init__.py')
            if os.path.exists(init_py):
                found_path = init_py
                break
        if found_path:
            loader = __import__('importlib.machinery').machinery.SourceFileLoader('xgboost', found_path)
            spec = __import__('importlib.util').util.spec_from_loader(loader.name, loader)
            mod = __import__('importlib.util').util.module_from_spec(spec)
            loader.exec_module(mod)
            sys.modules['xgboost'] = mod

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f'Model not found at {MODEL_PATH} — ensure it is present')
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f'Failed to load model: {e}')
    expected_features = load_features(FEATURES_PATH)
    threshold = load_threshold(RESULTS_PATH)


@app.get('/features')
def get_features():
    global expected_features
    if expected_features is None:
        return {'features': [], 'count': 0}
    return {'features': expected_features, 'count': len(expected_features)}


@app.get('/health')
def health():
    return {'status': 'ok', 'model_loaded': model is not None, 'features_loaded': expected_features is not None}


@app.post('/predict')
def predict(req: PredictRequest):
    global model, expected_features, threshold
    if model is None:
        raise HTTPException(status_code=503, detail='Model not loaded')

    data = req.data
    if isinstance(data, dict):
        data = [data]
    df = pd.DataFrame(data)

    # If expected_features defined, ensure columns exist on DataFrame; do not reject if missing but pass NaNs
    if expected_features is not None:
        missing = [f for f in expected_features if f not in df.columns]
        if missing:
            # add missing columns with NaN values so pipeline can still process
            for m in missing:
                df[m] = np.nan
        # reorder and extra columns are allowed but will be ignored
        df = df[expected_features].copy()

    # preprocess the DataFrame same as training
    df_prep = coerce_and_prep(df)

    # perform prediction
    try:
        proba = model.predict_proba(df_prep)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Prediction failed: {e}')

    labels = (proba >= threshold).astype(int)
    resp = []
    # Convert NaNs to None for JSON serializable output while preserving df for prediction
    df_json = df.replace({np.nan: None})
    for i, row in enumerate(df_json.to_dict(orient='records')):
        prob_val = proba[i]
        if isinstance(prob_val, (float,)) and np.isnan(prob_val):
            pred_proba = None
        else:
            pred_proba = float(prob_val)
        resp.append({'index': i, 'input': row, 'pred_proba': pred_proba, 'pred_label': int(labels[i])})

    if req.explain:
        try:
            import shap
        except Exception:
            return {'predictions': resp, 'threshold': threshold, 'explain_error': 'shap_not_installed'}

        shap_info = []
        try:
            # if our model is a pipeline with 'preprocessor' and 'clf', use the preprocessor transform for SHAP
            if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps and 'clf' in model.named_steps:
                pre = model.named_steps['preprocessor']
                clf = model.named_steps['clf']
                try:
                    print('DEBUG clf type', type(clf), getattr(clf, '__class__', None))
                except Exception:
                    pass
                try:
                    X_trans = pre.transform(df_prep)
                except Exception:
                    X_trans = pre.fit_transform(df_prep)
                try:
                    feature_names = list(pre.get_feature_names_out(df_prep.columns))
                except Exception:
                    feature_names = list(df_prep.columns)
                # compute SHAP using TreeExplainer if possible, fallback to Explainer
                try:
                    explainer = shap.TreeExplainer(clf)
                    shap_values = explainer.shap_values(X_trans)
                except Exception:
                    explainer = shap.Explainer(clf.predict_proba, X_trans)
                    shap_res = explainer(X_trans)
                    shap_values = shap_res.values
                if isinstance(shap_values, (list, tuple)):
                    shap_values = np.array(shap_values)
                shap_arr = np.array(shap_values)
                # handle either layout of class axis
                if shap_arr.ndim == 3:
                    if shap_arr.shape[1] == 2:
                        shap_arr = shap_arr[:, 1, :]
                    elif shap_arr.shape[2] == 2:
                        shap_arr = shap_arr[:, :, 1]
                # if all zeros, try fallback Explainer with predict_proba
                try:
                    if np.allclose(shap_arr, 0.0):
                        explainer = shap.Explainer(clf.predict_proba, X_trans)
                        shap_res = explainer(X_trans)
                        shap_arr = np.array(shap_res.values)
                        if shap_arr.ndim == 3:
                            if shap_arr.shape[1] == 2:
                                shap_arr = shap_arr[:, 1, :]
                            elif shap_arr.shape[2] == 2:
                                shap_arr = shap_arr[:, :, 1]
                except Exception:
                    pass
                # map transformed features back to base input names and aggregate contributions
                try:
                    for i_row, svals in enumerate(shap_arr):
                        row_vals = resp[i_row]['input'] if i_row < len(resp) else None
                        agg = {}
                        svals_arr = np.asarray(svals)
                        if svals_arr.ndim == 0:
                            svals_arr = svals_arr.reshape((1,))
                        for fname, sval in zip(feature_names, svals_arr):
                            base_name = str(fname).split('__')[-1] if '__' in str(fname) else str(fname)
                            fval = row_vals.get(base_name) if row_vals and base_name in row_vals else None
                            try:
                                contribution = float(sval)
                            except Exception:
                                sval_arr = np.asarray(sval)
                                if sval_arr.size == 1:
                                    contribution = float(sval_arr.item())
                                else:
                                    contribution = float(np.sum(sval_arr))
                            if base_name not in agg:
                                agg[base_name] = {'feature': base_name, 'value': fval, 'contribution': 0.0}
                            agg[base_name]['contribution'] += float(contribution)
                        factors = []
                        for k, v in agg.items():
                            direction = 'increase' if float(v['contribution']) > 0 else 'decrease'
                            factors.append({'feature': v['feature'], 'value': v['value'], 'contribution': float(v['contribution']), 'direction': direction})
                        factors = sorted(factors, key=lambda x: abs(x['contribution']), reverse=True)[:50]
                        shap_info.append(factors)
                except Exception as e:
                    # graceful fallback
                    shap_info = None
            else:
                # fallback to raw df features
                try:
                    explainer = shap.Explainer(model.predict_proba, df_prep)
                    shap_res = explainer(df_prep)
                    shap_arr = np.array(shap_res.values)
                    if shap_arr.ndim == 3:
                        if shap_arr.shape[1] == 2:
                            shap_arr = shap_arr[:, 1, :]
                        elif shap_arr.shape[2] == 2:
                            shap_arr = shap_arr[:, :, 1]
                    feature_names = list(df_prep.columns)
                    for i_row, svals in enumerate(shap_arr):
                        row_vals = resp[i_row]['input'] if i_row < len(resp) else None
                        factors = []
                        for fname, sval in zip(feature_names, svals):
                            fval = None
                            if row_vals and fname in row_vals:
                                fval = row_vals.get(fname)
                            factors.append({'feature': fname, 'value': fval, 'contribution': float(sval), 'direction': 'increase' if float(sval) > 0 else 'decrease'})
                        factors = sorted(factors, key=lambda x: abs(x['contribution']), reverse=True)[:50]
                        shap_info.append(factors)
                except Exception:
                    shap_info = None

            if shap_info is not None and len(shap_info) == len(resp):
                for i_row in range(len(resp)):
                    resp[i_row]['shapFactors'] = shap_info[i_row]
        except Exception as e:
            return {'predictions': resp, 'threshold': threshold, 'explain_error': str(e)}

    return {'predictions': resp, 'threshold': threshold}


if __name__ == '__main__':
    # Useful for local testing: uvicorn Train_model.serve:app --reload
    uvicorn.run('Train_model.serve:app', host='0.0.0.0', port=8000, reload=True)
"""
Simple FastAPI backend to serve the XGBoost model built by `xgboost_tpe.py`.
The server expects `MODEL_PATH` to point to a sklearn Pipeline pkl containing preprocessing + classifier, or a classifier object.

Endpoints:
  GET /health - returns 200 if model loaded
  POST /predict - accepts JSON body containing either a single sample or a list of samples

Input format:
  { "data": { "feature1": 1, "feature2": 0, ... } }
  or
  { "data": [ {..}, {..} ] }

Output:
  returns probabilities and labels per sample
"""
from fastapi import FastAPI, HTTPException
    if req.explain:
        try:
            import shap
        except Exception:
            # shap not installed; signal in response
            return {'predictions': resp, 'threshold': threshold, 'explain_error': 'shap_not_installed'}

        shap_info = []
        try:
            # attempt to extract preprocessor and estimator from pipeline
            if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps and 'clf' in model.named_steps:
                pre = model.named_steps['preprocessor']
                clf = model.named_steps['clf']
                try:
                    print('DEBUG clf type', type(clf), getattr(clf, '__class__', None))
                except Exception:
                    pass
                # apply preprocessing transform
                try:
                    X_trans = pre.transform(df_prep)
                except Exception:
                    X_trans = pre.fit_transform(df_prep)
                # get feature names after preprocessing (if supported)
                try:
                    feature_names = list(pre.get_feature_names_out(df_prep.columns))
                except Exception:
                    feature_names = list(df_prep.columns)
                # compute SHAP values using TreeExplainer for XGBoost
                try:
                    explainer = shap.TreeExplainer(clf)
                    shap_values = explainer.shap_values(X_trans)
                except Exception:
                    # fallback to shap.Explainer
                    explainer = shap.Explainer(clf.predict_proba, X_trans)
                    shap_res = explainer(X_trans)
                    shap_values = shap_res.values
                # shap_values can have shape (n, n_features) or (n, 2, n_features) for binary
                if isinstance(shap_values, (list, tuple)):
                    # older shap versions
                    shap_values = np.array(shap_values)
                shap_arr = np.array(shap_values)
                # Handle shap arrays for binary classifiers that can have class dimension in axis 1 or axis 2
                if shap_arr.ndim == 3:
                    if shap_arr.shape[1] == 2:
                        # shape (n_samples, n_classes, n_features)
                        shap_arr = shap_arr[:, 1, :]
                    elif shap_arr.shape[2] == 2:
                        # shape (n_samples, n_features, n_classes) -> pick class 1
                        shap_arr = shap_arr[:, :, 1]
                # If shap returns all zeros (some shap/tree combos), try a different explainer
                try:
                    if np.allclose(shap_arr, 0.0):
                        explainer = shap.Explainer(clf.predict_proba, X_trans)
                        shap_res = explainer(X_trans)
                        shap_arr = np.array(shap_res.values)
                        if shap_arr.ndim == 3 and shap_arr.shape[1] == 2:
                            shap_arr = shap_arr[:, 1, :]
                except Exception:
                    # ignore fallback errors, keep the existing shap_arr
                    pass
                # Debugging: log feature names and shap_arr shape/sample
                try:
                    print('SHAP feature_names len=', len(feature_names))
                    print('SHAP arr shape=', shap_arr.shape)
                    if shap_arr.shape[0] > 0 and shap_arr.shape[1] >= 10:
                        print('SHAP sample first 10:', shap_arr[0, :10])
                except Exception:
                    pass
                # Build shap factors per sample
                for i_row, svals in enumerate(shap_arr):
                    row_vals = resp[i_row]['input'] if i_row < len(resp) else None
                    # We'll aggregate contributions across transformed features that map back to the same input feature
                    agg = {}
                    # Ensure svals is a 1D numpy array
                    svals_arr = np.asarray(svals)
                    if svals_arr.ndim == 0:
                        svals_arr = svals_arr.reshape((1,))
                    for fname, sval in zip(feature_names, svals_arr):
                        # Map transformed feature name back to base input name (e.g., 'num__AGE' -> 'AGE')
                        base_name = str(fname)
                        if '__' in base_name:
                            base_name = base_name.split('__')[-1]
                        # preserve original input value for feature if available
                        fval = None
                        if row_vals and base_name in row_vals:
                            fval = row_vals.get(base_name)
                        # Coerce the shap-value into a scalar safely
                        try:
                            contribution = float(sval)
                        except Exception:
                            sval_arr = np.asarray(sval)
                            if sval_arr.size == 1:
                                contribution = float(sval_arr.item())
                            else:
                                contribution = float(np.sum(sval_arr))
                        # Aggregate contributions by base_name
                        if base_name not in agg:
                            agg[base_name] = {'feature': base_name, 'value': fval, 'contribution': 0.0}
                        agg[base_name]['contribution'] += float(contribution)
                    # convert agg into sorted list of factors
                    factors = []
                    for k, v in agg.items():
                        direction = 'increase' if float(v['contribution']) > 0 else 'decrease'
                        factors.append({'feature': v['feature'], 'value': v['value'], 'contribution': float(v['contribution']), 'direction': direction})
                    # sort by absolute contribution and truncate to top 50
                    factors = sorted(factors, key=lambda x: abs(x['contribution']), reverse=True)[:50]
                    shap_info.append(factors)
            else:
                # pipeline shape unknown; attempt to run shap.Explainer directly with model
                try:
                    explainer = shap.Explainer(model.predict_proba, df_prep)
                    shap_res = explainer(df_prep)
                    shap_arr = np.array(shap_res.values)
                    if shap_arr.ndim == 3 and shap_arr.shape[1] == 2:
                        shap_arr = shap_arr[:, 1, :]
                    feature_names = list(df_prep.columns)
                    for i_row, svals in enumerate(shap_arr):
                        row_vals = resp[i_row]['input'] if i_row < len(resp) else None
                        factors = []
                        for fname, sval in zip(feature_names, svals):
                            fval = None
                            if row_vals and fname in row_vals:
                                fval = row_vals.get(fname)
                            factors.append({'feature': fname, 'value': fval, 'contribution': float(sval), 'direction': 'increase' if float(sval) > 0 else 'decrease'})
                        factors = sorted(factors, key=lambda x: abs(x['contribution']), reverse=True)[:50]
                        shap_info.append(factors)
                except Exception as e:
                    shap_info = None
            # attach shap_info into response per prediction
            if shap_info is not None and len(shap_info) == len(resp):
                for i_row in range(len(resp)):
                    resp[i_row]['shapFactors'] = shap_info[i_row]
        except Exception as e:
            # if something fails, include an explain_error message to help debugging
            return {'predictions': resp, 'threshold': threshold, 'explain_error': str(e)}

    # If expected_features defined, ensure columns exist on DataFrame; do not reject if missing but pass NaNs
    if expected_features is not None:
        missing = [f for f in expected_features if f not in df.columns]
        if missing:
            # add missing columns with NaN values so pipeline can still process
            for m in missing:
                df[m] = np.nan
        # reorder and extra columns are allowed but will be ignored
        df = df[expected_features].copy()

    # preprocess the DataFrame same as training
    df_prep = coerce_and_prep(df)

    # perform prediction
    try:
        proba = model.predict_proba(df_prep)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Prediction failed: {e}')

    labels = (proba >= threshold).astype(int)
    resp = []
    # Convert NaNs to None for JSON serializable output while preserving df for prediction
    df_json = df.replace({np.nan: None})
    for i, row in enumerate(df_json.to_dict(orient='records')):
        prob_val = proba[i]
        if isinstance(prob_val, (float,)) and np.isnan(prob_val):
            pred_proba = None
        else:
            pred_proba = float(prob_val)
        resp.append({'index': i, 'input': row, 'pred_proba': pred_proba, 'pred_label': int(labels[i])})
    # If the request asks for explainability information, try to compute SHAP values per sample
    if req.explain:
        try:
            import shap
        except Exception:
            # shap not installed; signal in response
            return {'predictions': resp, 'threshold': threshold, 'explain_error': 'shap_not_installed'}

        shap_info = []
            # attempt to extract preprocessor and estimator from pipeline
            if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps and 'clf' in model.named_steps:
                pre = model.named_steps['preprocessor']
                clf = model.named_steps['clf']
                        # debug
                try:
                    print('DEBUG clf type', type(clf), getattr(clf,'__class__',None))
                except Exception:
                    pass
                # apply preprocessing transform
                try:
                    X_trans = pre.transform(df_prep)
                except Exception:
                    X_trans = pre.fit_transform(df_prep)
                # get feature names after preprocessing (if supported)
                try:
                    feature_names = list(pre.get_feature_names_out(df_prep.columns))
                except Exception:
                    feature_names = list(df_prep.columns)
                # compute SHAP values using TreeExplainer for XGBoost
                try:
                    explainer = shap.TreeExplainer(clf)
                    shap_values = explainer.shap_values(X_trans)
                except Exception:
                    # fallback to shap.Explainer
                    explainer = shap.Explainer(clf.predict_proba, X_trans)
                    shap_res = explainer(X_trans)
                    shap_values = shap_res.values
                # shap_values can have shape (n, n_features) or (n, 2, n_features) for binary
                if isinstance(shap_values, (list, tuple)):
                    # older shap versions
                    shap_values = np.array(shap_values)
                shap_arr = np.array(shap_values)
                # If shap returns (n, 2, m) for binary classifier, select class 1
                # Handle shap arrays for binary classifiers that can have class dimension in axis 1 or axis 2
                if shap_arr.ndim == 3:
                    if shap_arr.shape[1] == 2:
                        # shape (n_samples, n_classes, n_features)
                        shap_arr = shap_arr[:, 1, :]
                    elif shap_arr.shape[2] == 2:
                        # shape (n_samples, n_features, n_classes) -> pick class 1
                        shap_arr = shap_arr[:, :, 1]
                # If shap returns all zeros (some shap/tree combos), try a different explainer
                try:
                    if np.allclose(shap_arr, 0.0):
                        explainer = shap.Explainer(clf.predict_proba, X_trans)
                        shap_res = explainer(X_trans)
                        shap_arr = np.array(shap_res.values)
                        if shap_arr.ndim == 3 and shap_arr.shape[1] == 2:
                            shap_arr = shap_arr[:, 1, :]
                except Exception:
                    # ignore fallback errors, keep the existing shap_arr
                    pass
                # Debugging: log feature names and shap_arr shape/sample
                try:
                    print('SHAP feature_names len=', len(feature_names))
                    print('SHAP arr shape=', shap_arr.shape)
                    if shap_arr.shape[0] > 0 and shap_arr.shape[1] >= 10:
                        print('SHAP sample first 10:', shap_arr[0, :10])
                except Exception:
                    pass
                # Build shap factors per sample
                for i_row, svals in enumerate(shap_arr):
                    row_vals = resp[i_row]['input'] if i_row < len(resp) else None
                    # We'll aggregate contributions across transformed features that map back to the same input feature
                    agg = {}
                    # Ensure svals is a 1D numpy array
                    svals_arr = np.asarray(svals)
                    if svals_arr.ndim == 0:
                        svals_arr = svals_arr.reshape((1,))
                    for fname, sval in zip(feature_names, svals_arr):
                        # Map transformed feature name back to base input name (e.g., 'num__AGE' -> 'AGE')
                        base_name = str(fname)
                        if '__' in base_name:
                            base_name = base_name.split('__')[-1]
                        # preserve original input value for feature if available
                        fval = None
                        if row_vals and base_name in row_vals:
                            fval = row_vals.get(base_name)
                        # Coerce the shap-value into a scalar safely
                        try:
                            contribution = float(sval)
                        except Exception:
                            sval_arr = np.asarray(sval)
                            if sval_arr.size == 1:
                                contribution = float(sval_arr.item())
                            else:
                                shap_info = []
                                # attempt to extract preprocessor and estimator from pipeline
                                if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps and 'clf' in model.named_steps:
                                    pre = model.named_steps['preprocessor']
                                    clf = model.named_steps['clf']
                                    try:
                                        print('DEBUG clf type', type(clf), getattr(clf, '__class__', None))
                                    except Exception:
                                        pass
                                    # apply preprocessing transform
                                    try:
                                        X_trans = pre.transform(df_prep)
                                    except Exception:
                                        X_trans = pre.fit_transform(df_prep)
                                    # get feature names after preprocessing (if supported)
                                    try:
                                        feature_names = list(pre.get_feature_names_out(df_prep.columns))
                                    except Exception:
                                        feature_names = list(df_prep.columns)
                                    # compute SHAP values using TreeExplainer for XGBoost
                                    try:
                                        explainer = shap.TreeExplainer(clf)
                                        shap_values = explainer.shap_values(X_trans)
                                    except Exception:
                                        # fallback to shap.Explainer
                                        explainer = shap.Explainer(clf.predict_proba, X_trans)
                                        shap_res = explainer(X_trans)
                                        shap_values = shap_res.values
                                    # shap_values can have shape (n, n_features) or (n, 2, n_features) for binary
                                    if isinstance(shap_values, (list, tuple)):
                                        # older shap versions
                                        shap_values = np.array(shap_values)
                                    shap_arr = np.array(shap_values)
                                    # Handle shap arrays for binary classifiers that can have class dimension in axis 1 or axis 2
                                    if shap_arr.ndim == 3:
                                        if shap_arr.shape[1] == 2:
                                            # shape (n_samples, n_classes, n_features)
                                            shap_arr = shap_arr[:, 1, :]
                                        elif shap_arr.shape[2] == 2:
                                            # shape (n_samples, n_features, n_classes) -> pick class 1
                                            shap_arr = shap_arr[:, :, 1]
                                    # If shap returns all zeros (some shap/tree combos), try a different explainer
                                    try:
                                        if np.allclose(shap_arr, 0.0):
                                            explainer = shap.Explainer(clf.predict_proba, X_trans)
                                            shap_res = explainer(X_trans)
                                            shap_arr = np.array(shap_res.values)
                                            if shap_arr.ndim == 3 and shap_arr.shape[1] == 2:
                                                shap_arr = shap_arr[:, 1, :]
                                    except Exception:
                                        # ignore fallback errors, keep the existing shap_arr
                                        pass
                                    # Debugging: log feature names and shap_arr shape/sample
                                    try:
                                        print('SHAP feature_names len=', len(feature_names))
                                        print('SHAP arr shape=', shap_arr.shape)
                                        if shap_arr.shape[0] > 0 and shap_arr.shape[1] >= 10:
                                            print('SHAP sample first 10:', shap_arr[0, :10])
                                    except Exception:
                                        pass
                                    # Build shap factors per sample
                                    for i_row, svals in enumerate(shap_arr):
                                        row_vals = resp[i_row]['input'] if i_row < len(resp) else None
                                        # We'll aggregate contributions across transformed features that map back to the same input feature
                                        agg = {}
                                        # Ensure svals is a 1D numpy array
                                        svals_arr = np.asarray(svals)
                                        if svals_arr.ndim == 0:
                                            svals_arr = svals_arr.reshape((1,))
                                        for fname, sval in zip(feature_names, svals_arr):
                                            # Map transformed feature name back to base input name (e.g., 'num__AGE' -> 'AGE')
                                            base_name = str(fname)
                                            if '__' in base_name:
                                                base_name = base_name.split('__')[-1]
                                            # preserve original input value for feature if available
                                            fval = None
                                            if row_vals and base_name in row_vals:
                                                fval = row_vals.get(base_name)
                                            # Coerce the shap-value into a scalar safely
                                            try:
                                                contribution = float(sval)
                                            except Exception:
                                                sval_arr = np.asarray(sval)
                                                if sval_arr.size == 1:
                                                    contribution = float(sval_arr.item())
                                                else:
                                                    contribution = float(np.sum(sval_arr))
                                            # Aggregate contributions by base_name
                                            if base_name not in agg:
                                                agg[base_name] = {'feature': base_name, 'value': fval, 'contribution': 0.0}
                                            agg[base_name]['contribution'] += float(contribution)
                                        # convert agg into sorted list of factors
                                        factors = []
                                        for k, v in agg.items():
                                            direction = 'increase' if float(v['contribution']) > 0 else 'decrease'
                                            factors.append({'feature': v['feature'], 'value': v['value'], 'contribution': float(v['contribution']), 'direction': direction})
                                        # sort by absolute contribution and truncate to top 50
                                        factors = sorted(factors, key=lambda x: abs(x['contribution']), reverse=True)[:50]
                                        shap_info.append(factors)
