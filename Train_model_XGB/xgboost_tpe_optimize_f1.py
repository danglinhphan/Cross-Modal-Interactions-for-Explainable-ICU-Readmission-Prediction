"""Train XGBoost for ICU readmission using Optuna with TPESampler; optimize for F1."""
import os
import json
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, precision_recall_curve, classification_report, roc_auc_score, average_precision_score

import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder


def load_and_prep(path):
    df = pd.read_csv(path)
    if 'Y' not in df.columns:
        raise RuntimeError('Target Y not found in input')
    drop_cols = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'INTIME', 'OUTTIME']
    cols = [c for c in df.columns if c not in drop_cols]
    df = df[cols].copy()
    return df


def build_pipeline(params, X, n_jobs=1):
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]
    num_transformer = SimpleImputer(strategy='median')
    cat_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])
    transformers = []
    if numeric_features:
        transformers.append(('num', num_transformer, numeric_features))
    if categorical_features:
        transformers.append(('cat', cat_transformer, categorical_features))
    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop', sparse_threshold=0)
    clf = xgb.XGBClassifier(**params, n_jobs=n_jobs)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('clf', clf)])
    return pipeline


def objective(trial, X, y, cv):
    param = {
        'verbosity': 0,
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'use_label_encoder': False,
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.2, log=True),
        'n_estimators': trial.suggest_categorical('n_estimators', [100, 200, 300, 500]),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
    base_spw = int((len(y) - y.sum()) / (y.sum() + 1e-9))
    spw_factor = trial.suggest_float('spw_factor', 0.5, 3.0)
    param['scale_pos_weight'] = max(1.0, base_spw * spw_factor)
    scores = []
    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        pipe = build_pipeline(param, X_tr, n_jobs=1)
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_val)[:, 1]
        preds = (proba >= 0.5).astype(int)
        scores.append(f1_score(y_val, preds, zero_division=0))
    return float(np.mean(scores))


def main(args):
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    df = load_and_prep(args.input)
    X = df.drop(columns=['Y'])
    y = df['Y'].astype(int)
    print('Data rows:', len(df))
    print('Class balance (Y=1):', int(y.sum()))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=args.random_state)
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.random_state)
    sampler = TPESampler(seed=args.random_state)
    study = optuna.create_study(direction='maximize', sampler=sampler, study_name='xgb_readmit_tpe_f1')
    def _obj(trial):
        return objective(trial, X_train, y_train, cv)
    print('Starting Optuna TPE F1 optimization: n_trials=', args.n_trials)
    study.optimize(_obj, n_trials=args.n_trials, n_jobs=args.n_jobs)
    print('Best trial:')
    print(' value: ', study.best_value)
    print(' params: ')
    for k, v in study.best_params.items():
        print(f'  {k}: {v}')
    best_params = study.best_params.copy()
    best_params.update({'verbosity': 0, 'objective': 'binary:logistic', 'use_label_encoder': False, 'tree_method': 'hist', 'random_state': args.random_state})
    final_pipe = build_pipeline(best_params, X_train, n_jobs=args.n_jobs)
    final_pipe.fit(X_train, y_train)
    proba = final_pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)
    precisions, recalls, thresholds = precision_recall_curve(y_test, proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-12)
    best_idx = np.nanargmax(f1_scores)
    best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    y_pred = (proba >= best_threshold).astype(int)
    cls_report = classification_report(y_test, y_pred, output_dict=True)
    out = {'best_cv_f1': float(study.best_value), 'test_auc': float(auc), 'test_ap': float(ap), 'best_threshold': best_threshold, 'classification_report': cls_report, 'best_params': study.best_params, 'n_trials': args.n_trials}
    with open(os.path.join(outdir, 'readmission_tpe_results_f1.json'), 'w') as f:
        json.dump(out, f, indent=2)
    try:
        features_path = os.path.join(outdir, 'features.json')
        with open(features_path, 'w') as ff:
            json.dump(list(X_train.columns), ff, indent=2)
        print('Saved features list to', features_path)
    except Exception as e:
        print('Failed to save features.json:', e)
    try:
        import joblib
        joblib.dump(final_pipe, os.path.join(outdir, f'{args.model_name}.joblib'))
    except Exception:
        try:
            est = None
            try:
                est = final_pipe.named_steps.get('clf')
            except Exception:
                est = None
            if est is not None:
                est.get_booster().save_model(os.path.join(outdir, f'{args.model_name}.json'))
            else:
                final_pipe.get_booster().save_model(os.path.join(outdir, f'{args.model_name}.json'))
        except Exception:
            print('Failed to save model via joblib or booster.save_model')
    if getattr(args, 'save_pkl', True):
        try:
            import pickle
            pkl_path = os.path.join(outdir, f'{args.model_name}.pkl')
            with open(pkl_path, 'wb') as pf:
                pickle.dump(final_pipe, pf)
            print('Saved model pickle to', pkl_path)
        except Exception as e:
            print('Failed to save model as pickle:', e)
    try:
        import matplotlib.pyplot as plt
        est = None
        try:
            est = final_pipe.named_steps.get('clf')
        except Exception:
            est = None
        if est is None:
            raise ValueError('No clf step found in final pipeline')
        fi = est.feature_importances_
        idx = np.argsort(fi)[-30:]
        try:
            feature_names = final_pipe.named_steps['preprocessor'].get_feature_names_out(X_train.columns)
        except Exception:
            feature_names = X_train.columns
        plt.figure(figsize=(8, max(4, len(idx) * 0.2)))
        plt.barh(feature_names[idx], fi[idx])
        plt.title('Top features (TPE optimized model - F1)')
        plt.tight_layout()
        out_fig = os.path.join(outdir, 'feature_importance_tpe_f1.png')
        plt.savefig(out_fig)
        print('Saved feature importance to', out_fig)
    except Exception as e:
        print('Skipping feature importance plot (error):', e)
    try:
        trials = []
        for t in study.trials:
            trials.append({'number': t.number, 'value': t.value, 'params': t.params})
        with open(os.path.join(outdir, 'optuna_trials_f1.json'), 'w') as f:
            json.dump(trials, f, indent=2)
    except Exception:
        pass
    print('\nFinished. Results saved under', outdir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='cohort/new_cohort_icu_readmission.csv')
    parser.add_argument('--outdir', default=os.path.join(os.path.dirname(__file__), 'outputs', 'readmission_tpe_f1'))
    parser.add_argument('--model-name', default='best_readmission_xgb_tpe_f1')
    parser.add_argument('--test-size', default=0.2, type=float)
    parser.add_argument('--random-state', default=42, type=int)
    parser.add_argument('--cv', default=3, type=int)
    parser.add_argument('--n-trials', default=50, type=int)
    parser.add_argument('--n-jobs', default=1, type=int)
    parser.add_argument('--save-pkl', dest='save_pkl', action='store_true')
    parser.add_argument('--no-save-pkl', dest='save_pkl', action='store_false')
    parser.set_defaults(save_pkl=True)
    args = parser.parse_args()
    main(args)
