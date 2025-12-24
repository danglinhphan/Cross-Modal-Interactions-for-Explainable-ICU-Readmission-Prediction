"""
Simple inference helper for web app testing: loads a .pkl model (sklearn Pipeline with preprocessing and classifier), applies minimal validation and writes predictions.
Usage:
  python Train_model/inference.py --model Train_model/outputs/readmission_tpe/best_readmission_xgb_tpe.pkl --input cohort/new_cohort_icu_readmission.csv --output predictions.csv
"""
import argparse
import os
import pickle
import pandas as pd


def load_and_prep(path):
    df = pd.read_csv(path)
    # Keep original columns, but drop identifiers if present
    if 'SUBJECT_ID' in df.columns:
        df = df.copy()
        drop_cols = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'INTIME', 'OUTTIME']
        cols = [c for c in df.columns if c not in drop_cols]
        df = df[cols]
    # encode gender consistent with training
    # Normalize gender values to 'M'/'F' to match training data.
    if 'GENDER' in df.columns:
        df['GENDER'] = df['GENDER'].apply(lambda v: 'M' if str(v).lower() in ['male', 'm'] else ('F' if str(v).lower() in ['female', 'f'] else v))
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to .pkl model')
    parser.add_argument('--input', required=True, help='Input CSV to run predictions on')
    parser.add_argument('--output', required=False, default='predictions.csv', help='CSV to write predictions')
    parser.add_argument('--threshold', type=float, default=None, help='Probability threshold to convert into labels; if none, attempts to read from results JSON')
    parser.add_argument('--results', default=os.path.join(os.path.dirname(__file__), 'outputs', 'readmission_tpe', 'readmission_tpe_results.json'), help='Path to results json for threshold')
    args = parser.parse_args()

    # load model
    with open(args.model, 'rb') as f:
        model = pickle.load(f)

    # load data
    df = load_and_prep(args.input)
    if 'Y' in df.columns:
        X = df.drop(columns=['Y']).copy()
    else:
        X = df.copy()

    # determine threshold
    if args.threshold is None and os.path.exists(args.results):
        try:
            import json
            with open(args.results) as rf:
                res = json.load(rf)
            args.threshold = float(res.get('best_threshold', 0.5))
        except Exception:
            args.threshold = 0.5
    elif args.threshold is None:
        args.threshold = 0.5

    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= args.threshold).astype(int)

    out_df = df.copy()
    out_df['pred_proba'] = proba
    out_df['pred_label'] = pred
    out_df.to_csv(args.output, index=False)
    print('Wrote predictions to', args.output)


if __name__ == '__main__':
    main()
