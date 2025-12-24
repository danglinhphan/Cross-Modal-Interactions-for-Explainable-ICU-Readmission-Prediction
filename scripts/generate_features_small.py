"""Generate a small features CSV with updated feature_engineering code for inspection.

Usage:
  PYTHONPATH=. .venv/bin/python3 scripts/generate_features_small.py --n 200 --out cohort/features_updated_small.csv
"""

import argparse
import pandas as pd
from pathlib import Path
import sqlite3

# make sure local package import works
import sys
pkg_path = str(Path(__file__).resolve().parents[1] / 'data_preprocessing')
if pkg_path not in sys.path:
    sys.path.insert(0, pkg_path)

import feature_engineering as fe

def run_small(n=200, out='cohort/features_updated_small.csv'):
    # load cohort and restrict
    cohort_path = fe.COHORT_PATH if hasattr(fe, 'COHORT_PATH') else 'cohort/filtered_cohort.csv'
    df = pd.read_csv(cohort_path)
    small = df.head(n).copy()
    # ensure numeric IDs
    small['HADM_ID'] = pd.to_numeric(small['HADM_ID'], errors='coerce')
    small['ICUSTAY_ID'] = pd.to_numeric(small['ICUSTAY_ID'], errors='coerce')
    conn = fe.get_db_connection()

    demo_df = fe.get_demographics(conn, small)
    labs_df = fe.get_lab_events(conn, small)
    vitals_df = fe.get_chart_events(conn, small)
    urine_df = fe.get_urine_output(conn, small)
    pao2_df = fe.get_pao2_fio2_ratio(conn, small)
    comorb_df = fe.get_comorbidities(conn, small)
    vent_df = fe.get_ventilation(conn, small)

    final_df = demo_df.merge(labs_df, on='HADM_ID', how='left')
    final_df = final_df.merge(vitals_df, on='ICUSTAY_ID', how='left')
    final_df = final_df.merge(urine_df, on='ICUSTAY_ID', how='left')
    final_df = final_df.merge(pao2_df, on='ICUSTAY_ID', how='left')
    final_df = final_df.merge(comorb_df, on='HADM_ID', how='left')
    final_df = final_df.merge(vent_df, on='HADM_ID', how='left')

    # add range/pct change
    def add_range_pct(df, root):
        min_col = f"{root}_Min"
        max_col = f"{root}_Max"
        avg_col = f"{root}_Avg"
        range_col = f"{root}_Range"
        pct_col = f"{root}_PctChange"
        if min_col in df.columns and max_col in df.columns:
            df[range_col] = df[max_col] - df[min_col]
            if avg_col in df.columns:
                df[pct_col] = df[range_col] / (df[avg_col].abs() + 1e-9)
            else:
                df[pct_col] = df[range_col]
        return df

    lab_roots = ['BUN', 'Creatinine', 'Glucose', 'HMG', 'WBC', 'Platelet', 'Anion_Gap', 'PTT', 'Lactate', 'Albumin']
    vital_roots = ['SBP', 'DBP', 'HeartRate', 'RespRate', 'SpO2']
    for root in lab_roots + vital_roots:
        final_df = add_range_pct(final_df, root)

    final_df.to_csv(out, index=False)
    print('Wrote small features to', out, 'shape=', final_df.shape)
    conn.close()

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=200)
    parser.add_argument('--out', type=str, default='cohort/features_updated_small.csv')
    args = parser.parse_args()
    run_small(n=args.n, out=args.out)
