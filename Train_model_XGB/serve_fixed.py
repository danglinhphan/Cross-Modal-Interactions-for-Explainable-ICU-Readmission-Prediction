"""
Alternative server to quickly test SHAP explainability while keeping original `serve.py` intact.
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


app = FastAPI(title='Serve Fixed')

BASE = os.path.dirname(__file__)
MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join(BASE, 'outputs', 'readmission_tpe', 'best_readmission_xgb_tpe.pkl'))
FEATURES_PATH = os.environ.get('FEATURES_PATH', os.path.join(BASE, 'outputs', 'readmission_tpe', 'features.json'))
RESULTS_PATH = os.environ.get('RESULTS_PATH', os.path.join(BASE, 'outputs', 'readmission_tpe', 'readmission_tpe_results.json'))

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


def coerce(df: pd.DataFrame):
    if 'Y' in df.columns:
        df = df.drop(columns=['Y'])
    if 'GENDER' in df.columns:
        df['GENDER'] = df['GENDER'].apply(lambda v: 'M' if str(v).lower() in ['male', 'm'] else ('F' if str(v).lower() in ['female', 'f'] else v))
    return df


model = None
features = None
threshold = None


@app.on_event('startup')
def startup():
    global model, features, threshold
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    features = _load_json(FEATURES_PATH)
    threshold = _load_threshold(RESULTS_PATH)


@app.post('/predict')
def predict(req: PredictRequest):
    global model, features, threshold
    if model is None:
        raise HTTPException(status_code=503, detail='Model not loaded')
    data = req.data
    if isinstance(data, dict): data = [data]
    df = pd.DataFrame(data)
    if features is not None:
        missing = [f for f in features if f not in df.columns]
        for m in missing: df[m] = np.nan
        df = df[features].copy()
    X = coerce(df)
    proba = model.predict_proba(X)[:,1]
    label = (proba >= threshold).astype(int)
    res = [{'index':i,'input':row,'pred_proba':float(proba[i]), 'pred_label':int(label[i])} for i,row in enumerate(df.replace({np.nan:None}).to_dict(orient='records'))]

    if req.explain:
        try:
            import shap
        except Exception:
            return {'predictions': res, 'threshold': threshold, 'explain_error': 'shap_not_installed'}
        shap_info = []
        try:
            if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps and 'clf' in model.named_steps:
                pre = model.named_steps['preprocessor']
                clf = model.named_steps['clf']
                try: X_t = pre.transform(X)
                except: X_t = pre.fit_transform(X)
                try: feat_names = list(pre.get_feature_names_out(X.columns))
                except: feat_names = list(X.columns)
                try:
                    expl = shap.TreeExplainer(clf)
                    shap_vals = expl.shap_values(X_t)
                except Exception:
                    expl = shap.Explainer(clf.predict_proba, X_t)
                    shap_res = expl(X_t)
                    shap_vals = shap_res.values
                arr = np.array(shap_vals)
                if arr.ndim == 3:
                    if arr.shape[1] == 2: arr = arr[:,1,:]
                    elif arr.shape[2] == 2: arr = arr[:,:,1]
                for i_row, svals in enumerate(arr):
                    agg={}
                    for fname, sval in zip(feat_names, svals):
                        base = str(fname).split('__')[-1] if '__' in str(fname) else str(fname)
                        try: contribution = float(sval)
                        except: contribution = float(np.sum(np.asarray(sval)))
                        agg.setdefault(base, {'feature':base,'value':res[i_row]['input'].get(base) if base in res[i_row]['input'] else None, 'contribution':0.0})
                        agg[base]['contribution'] += contribution
                    factors = [{'feature':v['feature'],'value':v['value'],'contribution':float(v['contribution']),'direction':'increase' if v['contribution']>0 else 'decrease'} for v in agg.values()]
                    factors = sorted(factors, key=lambda x: abs(x['contribution']), reverse=True)[:50]
                    shap_info.append(factors)
            else:
                expl = shap.Explainer(model.predict_proba, X)
                shap_res = expl(X)
                arr = np.array(shap_res.values)
                if arr.ndim == 3:
                    if arr.shape[1] == 2: arr = arr[:,1,:]
                    elif arr.shape[2] == 2: arr = arr[:,:,1]
                feat_names = list(X.columns)
                for i_row, svals in enumerate(arr):
                    factors=[]
                    for fname, sval in zip(feat_names, svals):
                        try: contribution = float(sval)
                        except: contribution = float(np.sum(np.asarray(sval)))
                        val = res[i_row]['input'].get(fname) if fname in res[i_row]['input'] else None
                        factors.append({'feature':fname,'value':val,'contribution':contribution,'direction':'increase' if contribution>0 else 'decrease'})
                    factors = sorted(factors, key=lambda x: abs(x['contribution']), reverse=True)[:50]
                    shap_info.append(factors)
        except Exception as e:
            return {'predictions': res, 'threshold': threshold, 'explain_error': str(e)}

        if shap_info is not None and len(shap_info) == len(res):
            for i_row in range(len(res)):
                res[i_row]['shapFactors'] = shap_info[i_row]

    return {'predictions': res, 'threshold': threshold}


if __name__ == '__main__':
    uvicorn.run('Train_model.serve_fixed:app', host='0.0.0.0', port=8030, reload=False)
