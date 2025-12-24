"""Extract fixed-length time-series sequences for each admission up to DISCHTIME.

Creates hourly bins for the last N hours before DISCHTIME and aggregates vital/lab measurements
per bin (mean). Outputs a compressed .npz with array 'X' -> shape (n_samples, timesteps, features)
and 'y' for the labels, plus a CSV index mapping HADM_ID/ICUSTAY_ID.

This script supports fast testing with --limit to restrict the number of admissions processed.
"""
import argparse
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import timedelta


DB = 'dataset/MIMIC_III.db'
COHORT = 'cohort/filtered_cohort.csv'
OUT_DIR = 'cohort/time_series'

# Choose numerical ITEMIDs for common vitals/labs (a compact set)
CHART_ITEMS = {
    'SBP': [51, 442, 455, 6701, 220179, 220050],
    'DBP': [8368, 8440, 8441, 8555, 220180, 220051],
    'HR': [211, 220045],
    'RR': [618, 615, 220210, 224690],
    'SpO2': [646, 220277]
}

LAB_ITEMS = {
    'BUN': [51006],
    'Creatinine': [50912],
    'Glucose': [50931, 50809],
    'Lactate': [50813],
    'Albumin': [50862]
}


def get_admissions(limit=None):
    df = pd.read_csv(COHORT, usecols=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'OUTTIME'])
    df['OUTTIME'] = pd.to_datetime(df['OUTTIME'])
    if limit:
        return df.head(limit).copy()
    return df.copy()


def query_lab_events(conn, hadm_ids, window_start, window_end):
    ids = [i for lst in LAB_ITEMS.values() for i in lst]
    ids_str = ','.join(map(str, ids))
    hadm_tuple = tuple(map(int, hadm_ids)) if len(hadm_ids) > 1 else (int(hadm_ids[0]),)
    q = f"""
    SELECT HADM_ID, ITEMID, VALUENUM, CHARTTIME
    FROM LABEVENTS
    WHERE HADM_ID IN {hadm_tuple}
    AND ITEMID IN ({ids_str})
    AND VALUENUM IS NOT NULL
    AND CHARTTIME >= '{window_start}' AND CHARTTIME <= '{window_end}'
    """
    return pd.read_sql_query(q, conn)


def query_chart_events(conn, icu_ids, window_start, window_end):
    ids = [i for lst in CHART_ITEMS.values() for i in lst]
    ids_str = ','.join(map(str, ids))
    icu_tuple = tuple(map(int, icu_ids)) if len(icu_ids) > 1 else (int(icu_ids[0]),)
    q = f"""
    SELECT ICUSTAY_ID, ITEMID, VALUENUM, CHARTTIME
    FROM CHARTEVENTS
    WHERE ICUSTAY_ID IN {icu_tuple}
    AND ITEMID IN ({ids_str})
    AND VALUENUM IS NOT NULL
    AND CHARTTIME >= '{window_start}' AND CHARTTIME <= '{window_end}'
    """
    return pd.read_sql_query(q, conn)


def build_time_series(adms, conn, window_hours=48, bin_mins=60):
    # timesteps
    timesteps = int(window_hours * 60 / bin_mins)
    # We include several aggregations per variable per bin: mean, min, max, std, median
    AGGS = ['mean', 'min', 'max', 'std', 'median']
    feature_names = []
    for feat in list(CHART_ITEMS.keys()) + list(LAB_ITEMS.keys()):
        for a in AGGS:
            feature_names.append(f"{feat}_{a}")
    n_features = len(feature_names)

    X = []
    idx = []

    for _, row in adms.iterrows():
        hadm = int(row['HADM_ID'])
        icu = int(row['ICUSTAY_ID']) if not pd.isna(row['ICUSTAY_ID']) else None
        dischtime = row['OUTTIME']
        window_start = (dischtime - pd.Timedelta(hours=window_hours)).strftime('%Y-%m-%d %H:%M:%S')
        window_end = dischtime.strftime('%Y-%m-%d %H:%M:%S')

        # Query events limited to this admission's window
        lab_q = f"SELECT HADM_ID, ITEMID, VALUENUM, CHARTTIME FROM LABEVENTS WHERE HADM_ID = {hadm} AND VALUENUM IS NOT NULL AND CHARTTIME >= '{window_start}' AND CHARTTIME <= '{window_end}'"
        labs = pd.read_sql_query(lab_q, conn)
        labs['CHARTTIME'] = pd.to_datetime(labs['CHARTTIME'])

        # If ICUSTAY_ID is missing, fall back to HADM_ID for CHARTEVENTS so we still capture floor charting
        if icu is None:
            chart_q = f"SELECT HADM_ID as ICUSTAY_ID, ITEMID, VALUENUM, CHARTTIME FROM CHARTEVENTS WHERE HADM_ID = {hadm} AND VALUENUM IS NOT NULL AND CHARTTIME >= '{window_start}' AND CHARTTIME <= '{window_end}'"
        else:
            chart_q = f"SELECT ICUSTAY_ID, ITEMID, VALUENUM, CHARTTIME FROM CHARTEVENTS WHERE ICUSTAY_ID = {icu} AND VALUENUM IS NOT NULL AND CHARTTIME >= '{window_start}' AND CHARTTIME <= '{window_end}'"
        charts = pd.read_sql_query(chart_q, conn)
        charts['CHARTTIME'] = pd.to_datetime(charts['CHARTTIME'])

        # create bins
        edges = pd.date_range(end=dischtime, periods=timesteps+1, freq=f'{bin_mins}min')
        bins = []
        arr = np.full((timesteps, n_features), np.nan, dtype=np.float32)

        # aggregate per feature per bin
        # charts -> chart_items
        for i, (feat, ids) in enumerate(CHART_ITEMS.items()):
            if charts.empty:
                continue
            sub = charts[charts['ITEMID'].isin(ids)].copy()
            if sub.empty:
                continue
            # find bin index for each row
            sub['bin'] = np.searchsorted(edges, sub['CHARTTIME'], side='right') - 1
            valid = sub[(sub['bin'] >= 0) & (sub['bin'] < timesteps)]
            if not valid.empty:
                ag = valid.groupby('bin')['VALUENUM'].agg(['mean', 'min', 'max', 'std', 'median'])
                for bin_i, rowvals in ag.iterrows():
                    for s_i, s_name in enumerate(AGGS):
                        col_idx = i * len(AGGS) + s_i
                        arr[int(bin_i), col_idx] = rowvals[s_name]

        # labs -> lab items (map to hadm)
        for j, (feat, ids) in enumerate(LAB_ITEMS.items()):
            idx_feat = len(CHART_ITEMS) + j
            if labs.empty:
                continue
            sub = labs[labs['ITEMID'].isin(ids)].copy()
            if sub.empty:
                continue
            sub['bin'] = np.searchsorted(edges, sub['CHARTTIME'], side='right') - 1
            valid = sub[(sub['bin'] >= 0) & (sub['bin'] < timesteps)]
            if not valid.empty:
                ag = valid.groupby('bin')['VALUENUM'].agg(['mean', 'min', 'max', 'std', 'median'])
                for bin_i, rowvals in ag.iterrows():
                    for s_i, s_name in enumerate(AGGS):
                        col_idx = (idx_feat) * len(AGGS) + s_i
                        arr[int(bin_i), col_idx] = rowvals[s_name]

        X.append(arr)
        idx.append({'HADM_ID': hadm, 'ICUSTAY_ID': icu})

    # pack into array
    X = np.stack(X, axis=0) if X else np.zeros((0, timesteps, n_features), dtype=np.float32)

    # LOCF per sample per feature
    if X.size > 0:
        # Forward-fill (LOCF) along axis 1 (timesteps) for each sample and feature
        # We'll create a copy to avoid mutating original in unexpected ways
        X_f = X.copy()
        n_samples = X_f.shape[0]
        for s in range(n_samples):
            for f in range(n_features):
                col = X_f[s, :, f]
                # forward fill
                last = np.nan
                for t in range(col.shape[0]):
                    v = col[t]
                    if np.isnan(v):
                        if not np.isnan(last):
                            col[t] = last
                    else:
                        last = col[t]
                X_f[s, :, f] = col

        # compute global median per feature across all samples and timesteps ignoring NaN
        flat = X_f.reshape(-1, n_features)
        feature_medians = np.nanmedian(flat, axis=0)
        # Replace any remaining NaNs (leading NaNs where no measurement exists) with population median
        for s in range(n_samples):
            for f in range(n_features):
                col = X_f[s, :, f]
                mask = np.isnan(col)
                if mask.any():
                    # If entire column is NaN and median is NaN, leave as NaN
                    if np.isnan(feature_medians[f]):
                        continue
                    col[mask] = feature_medians[f]
                    X_f[s, :, f] = col
        X = X_f
    idx_df = pd.DataFrame(idx)
    return X, idx_df, feature_names


def main(limit=200, window_hours=48):
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    adms = get_admissions(limit=limit)
    conn = sqlite3.connect(DB)
    print('Processing', len(adms), 'admissions — window hrs', window_hours)
    X, idx_df, features = build_time_series(adms, conn, window_hours=window_hours, bin_mins=60)
    conn.close()

    out_npz = Path(OUT_DIR) / f'timeseries_last{window_hours}h_limit{limit}.npz'
    np.savez_compressed(out_npz, X=X)
    idx_df.to_csv(Path(OUT_DIR) / f'index_last{window_hours}h_limit{limit}.csv', index=False)
    print('Wrote', out_npz, 'shape=', X.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=200, help='number of admissions to process (fast test)')
    parser.add_argument('--window-hours', type=int, default=48, help='how many hours before discharge to include')
    args = parser.parse_args()
    main(limit=args.limit, window_hours=args.window_hours)
