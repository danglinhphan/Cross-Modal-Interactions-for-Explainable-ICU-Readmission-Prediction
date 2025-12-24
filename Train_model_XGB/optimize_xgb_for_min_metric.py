"""
Search hyperparameters, resampling, and class weights for an XGBoost model to maximize the minimum of (precision, recall, f1, accuracy)
"""
import argparse
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE
from sklearn.utils import compute_sample_weight
import xgboost as xgb
import joblib


def build_pipeline(clf, X):
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]
    num_tr = SimpleImputer(strategy='median')
    cat_tr = Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='MISSING')),
                       ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])
    transformers = []
    if numeric_features:
        transformers.append(('num', num_tr, numeric_features))
    if categorical_features:
        transformers.append(('cat', cat_tr, categorical_features))
    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    pipe = Pipeline([('pre', preprocessor), ('clf', clf)])
    return pipe


def best_threshold_min_metric(y_true, y_prob, thresholds=np.linspace(0.01, 0.99, 99)):
    best = (0.5, 0.0, 0.0, 0.0, 0.0)
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)
        f = f1_score(y_true, pred, zero_division=0)
        a = accuracy_score(y_true, pred)
        mm = min(p, r, f, a)
        if mm > min(best[1], best[2], best[3], best[4]):
            best = (t, p, r, f, a)
    return best


def resample(X_tr, y_tr, method, random_state=42):
    if method == 'none':
        return X_tr.copy(), y_tr.copy()
    X_enc = X_tr.copy()
    cat_cols = X_enc.select_dtypes(include=['object']).columns.tolist()
    enc = None
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_enc[cat_cols] = enc.fit_transform(X_enc[cat_cols].astype(str))
    if method == 'oversample':
        ros = RandomOverSampler(random_state=random_state)
        Xr, yr = ros.fit_resample(X_enc.values, y_tr.values)
    elif method == 'smote':
        sm = SMOTE(random_state=random_state)
        Xr, yr = sm.fit_resample(X_enc.values, y_tr.values)
    elif method == 'adasyn':
        ad = ADASYN(random_state=random_state)
        Xr, yr = ad.fit_resample(X_enc.values, y_tr.values)
    elif method == 'bl_smote':
        bl = BorderlineSMOTE(random_state=random_state)
        Xr, yr = bl.fit_resample(X_enc.values, y_tr.values)
    else:
        raise ValueError('Unknown resample')
    Xr = pd.DataFrame(Xr, columns=X_enc.columns)
    if enc is not None and cat_cols:
        try:
            Xr[cat_cols] = enc.inverse_transform(Xr[cat_cols])
        except Exception:
            Xr[cat_cols] = Xr[cat_cols].astype(str)
    return Xr, pd.Series(yr)


def evaluate_config(X, y, config, cv=3, random_state=42):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    fold_results = []
    for tr_idx, val_idx in skf.split(X, y):
        X_tr = X.iloc[tr_idx].reset_index(drop=True)
        y_tr = y.iloc[tr_idx].reset_index(drop=True)
        X_val = X.iloc[val_idx].reset_index(drop=True)
        y_val = y.iloc[val_idx].reset_index(drop=True)
        # resample
        # pre-impute numeric and fill categorical before resampling to avoid NaNs for SMOTE/ADASYN
        numeric_cols = X_tr.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in X_tr.columns if c not in numeric_cols]
        if numeric_cols:
            ni = SimpleImputer(strategy='median')
            X_tr_num = pd.DataFrame(ni.fit_transform(X_tr[numeric_cols]), columns=numeric_cols, index=X_tr.index)
        else:
            X_tr_num = pd.DataFrame(index=X_tr.index)
        X_tr_cat = X_tr[cat_cols].fillna('MISSING') if cat_cols else pd.DataFrame(index=X_tr.index)
        X_tr_pre = pd.concat([X_tr_num.reset_index(drop=True), X_tr_cat.reset_index(drop=True)], axis=1)
        X_res, y_res = resample(X_tr_pre, y_tr, config['resample'], random_state=random_state)
        # build pipeline
        clf = xgb.XGBClassifier(use_label_encoder=False, objective='binary:logistic', tree_method='hist', **config['params'])
        pipe = build_pipeline(clf, X_res)
        pipe.fit(X_res, y_res)
        proba = pipe.predict_proba(X_val)[:, 1]
        t, p, r, f, a = best_threshold_min_metric(y_val, proba)
        fold_results.append({'t': t, 'p': p, 'r': r, 'f': f, 'a': a})
    df = pd.DataFrame(fold_results)
    mean_p, mean_r, mean_f, mean_a = df[['p', 'r', 'f', 'a']].mean()
    mean_min = min(mean_p, mean_r, mean_f, mean_a)
    return {'mean_p': mean_p, 'mean_r': mean_r, 'mean_f': mean_f, 'mean_a': mean_a, 'mean_min': mean_min}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='cohort/new_cohort_icu_readmission.csv')
    parser.add_argument('--n-samples', type=int, default=20)
    parser.add_argument('--target', type=float, default=0.7)
    parser.add_argument('--cv', type=int, default=3)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--outdir', default='Train_model_XGB/outputs/opt_min_metric')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.input)
    X = df.drop(columns=['Y'])
    y = df['Y'].astype(int)

    # grid of params
    resamples = ['none', 'oversample', 'smote', 'adasyn', 'bl_smote']
    params_grid = [
        {'max_depth': 6, 'learning_rate': 0.05, 'n_estimators': 300, 'subsample': 0.7, 'colsample_bytree': 0.8, 'scale_pos_weight': 1.0},
        {'max_depth': 8, 'learning_rate': 0.1, 'n_estimators': 300, 'subsample': 0.6, 'colsample_bytree': 0.6, 'scale_pos_weight': 1.5},
        {'max_depth': 6, 'learning_rate': 0.2, 'n_estimators': 200, 'subsample': 0.5, 'colsample_bytree': 0.6, 'scale_pos_weight': 2.0},
        {'max_depth': 8, 'learning_rate': 0.15, 'n_estimators': 500, 'subsample': 0.5, 'colsample_bytree': 0.8, 'scale_pos_weight': 1.0}
    ]

    configs = []
    for r in resamples:
        for p in params_grid:
            configs.append({'resample': r, 'params': p})

    results = []
    for cfg in configs:
        try:
            res = evaluate_config(X, y, cfg, cv=args.cv, random_state=args.random_state)
            print('cfg', cfg['resample'], cfg['params']['max_depth'], cfg['params']['learning_rate'], '-> mean_min', res['mean_min'])
            res_record = {**cfg, **res}
            results.append(res_record)
            pd.DataFrame(results).to_csv(os.path.join(args.outdir, 'results.csv'), index=False)
            if res['mean_min'] >= args.target:
                # train final model on all data
                print('Found config meeting target:', cfg)
                clf = xgb.XGBClassifier(use_label_encoder=False, objective='binary:logistic', tree_method='hist', **cfg['params'])
                # resample full training
                X_res, y_res = resample(X, y, cfg['resample'], random_state=args.random_state)
                pipe = build_pipeline(clf, X_res)
                pipe.fit(X_res, y_res)
                # compute threshold
                proba = pipe.predict_proba(X)[:, 1]
                t, p, r, f, a = best_threshold_min_metric(y, proba)
                joblib.dump({'pipe': pipe, 'threshold': t, 'metrics': {'p': p, 'r': r, 'f': f, 'a': a}}, os.path.join(args.outdir, 'xgb_opt_best.pkl'))
                print('Saved best model to', os.path.join(args.outdir, 'xgb_opt_best.pkl'))
                break
        except Exception as e:
            print('Failed cfg', cfg, 'error', e)

    pd.DataFrame(results).to_csv(os.path.join(args.outdir, 'results.csv'), index=False)
    print('Done')


if __name__ == '__main__':
    main()
