"""Create a new cohort for ICU readmission prediction.

This standalone script creates a fresh cohort CSV (not overwriting other cohort files)
and a labels file. Filters and rules implemented:

- Keep only patients aged >= 18 at time of admission
- Keep only patient's *first* ICU stay (first by INTIME)
- Remove patients who died during their first ICU stay
- Remove patients that had a post-discharge readmission (any subsequent hospital admission
  with ADMITTIME > first admission's DISCHTIME) — these are excluded from the cohort
- Remove patients that are missing more than 1/3 of the chosen important clinical variables
- Choose clinical variables that are numeric and present (measured) in at least 80% of
  the cohort (measured flag columns like BUN_measured)
- Produce per-variable aggregated statistics across the first ICU stay: Avg, Std, Min, Max
- Label: Y=1 if patient had a later ICU stay within the same hospital admission
  (same HADM_ID) after the first ICU OUTTIME. Otherwise Y=0.

Usage:
  python data_preprocessing/create_cohort_icu_readmission.py

Outputs written to `cohort/new_cohort_icu_readmission.csv` and
`cohort/new_cohort_icu_readmission_labels.csv` in the repo.
"""

import json
import os
from pathlib import Path
import sqlite3
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / 'dataset' / 'MIMIC_III.db'
COHORT_DIR = ROOT / 'cohort'

# reuse feature extraction helpers implemented elsewhere
import sys
# make data_preprocessing importable when running script directly
pkg_path = str(Path(__file__).resolve().parents[0])
if pkg_path not in sys.path:
    sys.path.insert(0, pkg_path)
import feature_engineering as fe


def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}")

    conn = sqlite3.connect(str(DB_PATH))

    # Query first ICU stays per subject and apply filters directly in SQL to avoid datetime edge-cases
    query = '''
    WITH first_icu AS (
        SELECT SUBJECT_ID, HADM_ID, ICUSTAY_ID, INTIME, OUTTIME,
               ROW_NUMBER() OVER (PARTITION BY SUBJECT_ID ORDER BY INTIME) AS rn
        FROM ICUSTAYS
    )
    SELECT p.SUBJECT_ID, fi.HADM_ID, fi.ICUSTAY_ID, a.ADMITTIME, a.DISCHTIME, a.DEATHTIME, p.DOB, fi.INTIME, fi.OUTTIME,
           ROUND((julianday(a.ADMITTIME) - julianday(p.DOB)) / 365.25, 2) AS AGE
    FROM PATIENTS p
    JOIN first_icu fi ON p.SUBJECT_ID = fi.SUBJECT_ID AND fi.rn = 1
    JOIN ADMISSIONS a ON a.HADM_ID = fi.HADM_ID
    WHERE ROUND((julianday(a.ADMITTIME) - julianday(p.DOB)) / 365.25, 0) >= 18
      AND (a.DEATHTIME IS NULL OR a.DEATHTIME NOT BETWEEN fi.INTIME AND fi.OUTTIME)
      AND NOT EXISTS (
          SELECT 1 FROM ADMISSIONS a2 WHERE a2.SUBJECT_ID = p.SUBJECT_ID AND a2.ADMITTIME > a.DISCHTIME
      )
    '''

    pp = pd.read_sql_query(query, conn, parse_dates=['INTIME', 'OUTTIME', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'DOB'])

    # at this stage pp rows are our candidate cohort (first ICU stays, age>=18, not died in first ICU, no post-discharge readmit)
    print('Candidate cohort size:', len(pp))

    # build aggregated features from the database using existing helpers
    print('Extracting features for candidate cohort from database (demographics, labs, vitals, urine, PaO2/FiO2)')
    # cast ID types for compatibility with feature functions
    pp['HADM_ID'] = pd.to_numeric(pp['HADM_ID'], errors='coerce')
    pp['ICUSTAY_ID'] = pd.to_numeric(pp['ICUSTAY_ID'], errors='coerce')
    pp['SUBJECT_ID'] = pd.to_numeric(pp['SUBJECT_ID'], errors='coerce')

    # remove ADMIT/DISCH/DEATHTIME/DOB from pp to let feature_engineering fetch admission/patient
    # records itself and avoid column name collisions during merges
    pp = pp[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'INTIME', 'OUTTIME', 'AGE']]

    demo_df = fe.get_demographics(conn, pp)
    lab_df = fe.get_lab_events(conn, pp)
    chart_df = fe.get_chart_events(conn, pp)
    urine_df = fe.get_urine_output(conn, pp)
    pao2_df = fe.get_pao2_fio2_ratio(conn, pp)

    # Merge all the feature extraction results similar to cohort/features.csv layout
    feats = demo_df.merge(lab_df, on='HADM_ID', how='left')
    feats = feats.merge(chart_df, on='ICUSTAY_ID', how='left')
    feats = feats.merge(urine_df, on='ICUSTAY_ID', how='left')
    feats = feats.merge(pao2_df, on='ICUSTAY_ID', how='left')

    # read feature metadata to determine measured columns and numeric candidates
    meta_path = COHORT_DIR / 'feature_metadata.json'
    measured_cols = []
    numeric_candidates = []
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            measured_cols = meta.get('measured_cols', [])
            numeric_candidates = meta.get('numeric_cols', [])
    else:
        # fallback: detect measured flags as anything ending with '_measured'
        measured_cols = [c for c in feats.columns if c.endswith('_measured')]

    # compute measured flags from aggregated features in feats (if not present)
    for m in measured_cols:
        var = m.replace('_measured', '')
        present_cols = [c for c in feats.columns if c.lower().startswith(var.lower())]
        if present_cols:
            feats[m] = feats[present_cols].notna().any(axis=1).astype(int)
        else:
            feats[m] = 0

    # combine with patient/admission/icu meta in pp
    merged = pp.merge(feats, on='ICUSTAY_ID', how='left', suffixes=('', '_feat'))

    # make sure measured flags are numeric 0/1 in merged as well
    for m in measured_cols:
        if m in merged.columns:
            merged[m] = pd.to_numeric(merged[m], errors='coerce').fillna(0).astype(int)

    # identify which clinical variables meet the 80% measured threshold
    print('Selecting clinical variables present in >= 80% of candidate cohort (measured flags)...')
    selected_vars = []
    for m in measured_cols:
        # canonical variable name (strip _measured)
        var = m.replace('_measured', '')
        if m not in merged.columns:
            continue
        percent = merged[m].sum() / len(merged)
        if percent >= 0.8:
            selected_vars.append(var)
    print(f'Variables selected ({len(selected_vars)}): {selected_vars}')

    # prepare to collect aggregated columns for each selected var
    agg_suffixes = ['_Avg', '_Std', '_Min', '_Max']
    keep_cols = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'AGE', 'GENDER', 'INTIME', 'OUTTIME']
    # ensure AGE numeric
    merged['AGE'] = pd.to_numeric(merged['AGE'], errors='coerce')

    for v in selected_vars:
        for s in agg_suffixes:
            col = f"{v}{s}"
            if col in merged.columns:
                keep_cols.append(col)

    # ensure gender remains present (it may be string values 'M'/'F' but the user wants gender left as non-numeric)
    if 'GENDER' not in merged.columns and 'Gender' in merged.columns:
        merged['GENDER'] = merged['Gender']

    # subset to keep_cols
    cohort_df = merged[keep_cols].copy()

    # convert aggregated numeric columns to numeric types
    num_cols = [c for c in cohort_df.columns if c not in ('GENDER', 'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'INTIME', 'OUTTIME')]
    for c in num_cols:
        cohort_df[c] = pd.to_numeric(cohort_df[c], errors='coerce')

    # remove patients missing > 1/3 of selected important variables
    print('Removing patients missing more than 1/3 of selected important variables...')
    if selected_vars:
        # treat a variable as missing for a row if all aggregated columns for that variable are NaN
        def var_missing_for_row(row, var):
            cols = [f"{var}{s}" for s in agg_suffixes if f"{var}{s}" in cohort_df.columns]
            if not cols:
                return True
            return row[cols].isna().all()

        miss_counts = cohort_df.apply(lambda r: sum(var_missing_for_row(r, v) for v in selected_vars), axis=1)
        cohort_df['_MISS_CNT'] = miss_counts
        cohort_df['_N_VARS'] = len(selected_vars)
        remove_mask = cohort_df['_MISS_CNT'] > (cohort_df['_N_VARS'] / 3.0)
        print('Removing', int(remove_mask.sum()), 'rows for missing >1/3 important vars')
        cohort_df = cohort_df[~remove_mask].drop(columns=['_MISS_CNT', '_N_VARS'])

    # Build labels: check for subsequent ICU stays within same HADM_ID after OUTTIME
    print('Generating labels: Y=1 when patient has a later ICU stay during the same HADM_ID after the OUTTIME...')
    # load all icustays from DB to find any later icu rows with same HADM_ID
    all_icus = pd.read_sql_query('SELECT ICUSTAY_ID, HADM_ID, INTIME, OUTTIME FROM ICUSTAYS', conn, parse_dates=['INTIME','OUTTIME'])
    all_icus['ICUSTAY_ID'] = pd.to_numeric(all_icus['ICUSTAY_ID'], errors='coerce')
    all_icus['HADM_ID'] = pd.to_numeric(all_icus['HADM_ID'], errors='coerce')

    def label_for_row(row):
        hadm = row['HADM_ID']
        outt = row['OUTTIME']
        icu_id = row['ICUSTAY_ID']
        if pd.isna(hadm) or pd.isna(outt):
            return 0
        hadm = int(hadm)
        later = all_icus[(all_icus['HADM_ID'].astype(float).astype('Int64') == hadm) & (all_icus['INTIME'] > outt)]
        # if another icu exists within same hadm and its ICUSTAY_ID != current, label 1
        if not later.empty:
            # ensure it's not just the same icustay
            later_ids = set(later['ICUSTAY_ID'].astype(int).tolist())
            try:
                current = int(row['ICUSTAY_ID'])
            except Exception:
                current = None
            if current is None:
                return 0
            if any(ii != current for ii in later_ids):
                return 1
        return 0

    cohort_df['Y'] = cohort_df.apply(label_for_row, axis=1).astype(int)

    # map raw GENDER (M/F) from PATIENTS table into cohort_df where possible
    try:
        genders = pd.read_sql_query('SELECT SUBJECT_ID, GENDER FROM PATIENTS', conn)
        genders['SUBJECT_ID'] = pd.to_numeric(genders['SUBJECT_ID'], errors='coerce')
        gmap = dict(zip(genders['SUBJECT_ID'].astype(int), genders['GENDER']))
        cohort_df['GENDER'] = cohort_df['SUBJECT_ID'].astype(int).map(gmap).fillna(cohort_df.get('GENDER'))
    except Exception:
        pass

    # close DB connection
    try:
        conn.close()
    except Exception:
        pass

    # Save outputs
    out_cohort = COHORT_DIR / 'new_cohort_icu_readmission.csv'
    out_labels = COHORT_DIR / 'new_cohort_icu_readmission_labels.csv'

    print('Writing cohort to', out_cohort)
    cohort_df.to_csv(out_cohort, index=False)

    print('Writing labels to', out_labels)
    cohort_df[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'Y']].to_csv(out_labels, index=False)

    print('Done. Cohort rows:', len(cohort_df))


if __name__ == '__main__':
    main()
