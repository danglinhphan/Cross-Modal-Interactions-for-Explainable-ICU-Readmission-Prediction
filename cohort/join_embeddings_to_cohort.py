"""
Join note embeddings (npz + index CSV) into cohort CSV (HADM_ID) and save output. 
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def main(cohort_in, emb_npz, emb_index_csv, out_csv):
    cohort = pd.read_csv(cohort_in, dtype={ 'HADM_ID': object })
    idxdf = pd.read_csv(emb_index_csv, dtype={ 'HADM_ID': object })
    npz = np.load(emb_npz)
    X = npz['X']
    # X is n_samples x dim - ensure shapes match
    if X.shape[0] != len(idxdf):
        raise ValueError(f"Embedding rows {X.shape[0]} != index length {len(idxdf)}")
    emb_df = pd.DataFrame(X, columns=[f'note_emb_{i}' for i in range(X.shape[1])])
    emb_df['HADM_ID'] = idxdf['HADM_ID'].astype(object).values

    # merge
    merged = cohort.merge(emb_df, on='HADM_ID', how='left')
    # fill missing embedding with zeros
    emb_cols = [c for c in merged.columns if c.startswith('note_emb_')]
    merged[emb_cols] = merged[emb_cols].fillna(0.0)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    print('Wrote merged cohort with embeddings to', out_csv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort-in', default='cohort/new_cohort_icu_readmission.csv')
    parser.add_argument('--emb-npz', default='cohort/note_embeddings/discharge_embeddings_limit200.npz')
    parser.add_argument('--emb-index', default='cohort/note_embeddings/index_embeddings_limit200.csv')
    parser.add_argument('--out', default='cohort/new_cohort_with_emb.csv')
    args = parser.parse_args()
    main(args.cohort_in, args.emb_npz, args.emb_index, args.out)
