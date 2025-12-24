"""Extract discharge-note text for each admission and compute sentence-transformer embeddings.

Saves .npz with 'X' (n_samples x embedding_dim) and index CSV mapping HADM_ID/ICUSTAY_ID.
"""
import argparse
from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np


DB = 'dataset/MIMIC_III.db'
COHORT = 'cohort/filtered_cohort.csv'
OUT_DIR = 'cohort/note_embeddings'


def get_admissions(limit=None):
    df = pd.read_csv(COHORT, usecols=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'OUTTIME'])
    df['OUTTIME'] = pd.to_datetime(df['OUTTIME'])
    if limit:
        return df.head(limit).copy()
    return df.copy()


def collect_discharge_notes(conn, hadm_id, outtime):
    q = f"""
    SELECT HADM_ID, CATEGORY, CHARTTIME, TEXT
    FROM NOTEEVENTS
    WHERE HADM_ID = {hadm_id}
      AND CATEGORY LIKE '%Discharge%'
      AND CHARTTIME <= '{outtime.strftime('%Y-%m-%d %H:%M:%S')}'
    ORDER BY CHARTTIME
    """
    df = pd.read_sql_query(q, conn)
    if df.empty:
        return None
    # concatenate sequential discharge notes into a single string
    text = '\n\n'.join(df['TEXT'].astype(str).tolist())
    return text


def main(limit=200):
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    adms = get_admissions(limit=limit)
    conn = sqlite3.connect(DB)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')

    texts = []
    idx_rows = []

    for _, row in adms.iterrows():
        hadm = int(row['HADM_ID'])
        icu = int(row['ICUSTAY_ID']) if not pd.isna(row['ICUSTAY_ID']) else None
        outtime = row['OUTTIME']
        txt = collect_discharge_notes(conn, hadm, outtime)
        if txt is None:
            # fallback to empty string so dimensionally consistent
            texts.append('')
        else:
            texts.append(txt)
        idx_rows.append({'HADM_ID': hadm, 'ICUSTAY_ID': icu})

    conn.close()

    # compute embeddings (may require downloading model the first time)
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    idx_df = pd.DataFrame(idx_rows)
    out_npz = Path(OUT_DIR) / f'discharge_embeddings_limit{limit}.npz'
    np.savez_compressed(out_npz, X=embeddings)
    idx_df.to_csv(Path(OUT_DIR) / f'index_embeddings_limit{limit}.csv', index=False)
    print('Wrote', out_npz, 'shape=', embeddings.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=200, help='how many admissions to process')
    args = parser.parse_args()
    main(limit=args.limit)
