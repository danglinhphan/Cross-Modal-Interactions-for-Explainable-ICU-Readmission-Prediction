"""Filter or create an NLP features file scoped to the HADM_IDs in new_cohort_icu_readmission.csv

This script reads `cohort/new_cohort_icu_readmission.csv` for the HADM_ID list,
and `cohort/nlp_features.csv` for the cleaned text mapping, and writes an
output that contains one row per HADM in the new cohort with CLEAN_TEXT empty
when text is not available.

Outputs:
- cohort/nlp_features_for_new_cohort.csv
"""
from __future__ import annotations

import os
import argparse
import pandas as pd


def filter_nlp(nlp_csv: str, cohort_csv: str, out_csv: str, fill_missing_with_empty: bool = True):
    print(f"Reading NLP map from: {nlp_csv}")
    nlp = pd.read_csv(nlp_csv, dtype={"HADM_ID": object}, keep_default_na=False)
    print(f"NLP map rows: {len(nlp)}")

    print(f"Reading cohort from: {cohort_csv}")
    cohort = pd.read_csv(cohort_csv, dtype={"HADM_ID": object})
    # Ensure HADM_ID column exists
    if 'HADM_ID' not in cohort.columns:
        # If HADM is second column (older files) try HADM ID as column 2
        if 'HADM' in cohort.columns:
            cohort['HADM_ID'] = cohort['HADM']
        else:
            raise ValueError('cohort CSV must contain HADM_ID column')

    # Keep unique hadm list in cohort
    cohort_hadm = cohort[['HADM_ID']].drop_duplicates().reset_index(drop=True)
    print(f"Unique HADM in cohort: {len(cohort_hadm)}")

    # Merge, keeping all rows from cohort
    merged = cohort_hadm.merge(nlp[['HADM_ID', 'CLEAN_TEXT']], on='HADM_ID', how='left')
    missing_count = merged['CLEAN_TEXT'].isnull().sum()
    print(f"HADM with no CLEAN_TEXT in NLP map: {missing_count}")
    if fill_missing_with_empty:
        merged['CLEAN_TEXT'] = merged['CLEAN_TEXT'].fillna('')

    # Save to out_csv
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv} ({len(merged)} rows, missing CLEAN_TEXT: {missing_count})")
    return merged, missing_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nlp', default='cohort/nlp_features.csv')
    parser.add_argument('--cohort', default='cohort/new_cohort_icu_readmission.csv')
    parser.add_argument('--out', default='cohort/nlp_features_for_new_cohort.csv')
    parser.add_argument('--fill-missing-empty', action='store_true', default=True,
                        help='Fill missing CLEAN_TEXT with empty string (default)')
    args = parser.parse_args()

    filter_nlp(args.nlp, args.cohort, args.out, fill_missing_with_empty=args.fill_missing_empty)


if __name__ == '__main__':
    main()
