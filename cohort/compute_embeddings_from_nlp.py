"""
Compute embeddings from `cohort/nlp_features.csv` using sentence-transformers and save to npz + index CSV.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def main(nlp_csv, out_npz, out_index, model_name='all-MiniLM-L6-v2', limit=None):
    df = pd.read_csv(nlp_csv, dtype={'HADM_ID': object})
    if 'CLEAN_TEXT' not in df.columns or 'HADM_ID' not in df.columns:
        raise RuntimeError('nlp_csv must contain HADM_ID and CLEAN_TEXT columns')
    if limit:
        df = df.head(limit).copy()
    texts = df['CLEAN_TEXT'].fillna('').astype(str).tolist()
    hadms = df['HADM_ID'].astype(object).tolist()

    # Compute embeddings
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    arr = np.asarray(embeddings, dtype=np.float32)

    # Save
    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, X=arr)
    idx_df = pd.DataFrame({'HADM_ID': hadms})
    idx_df.to_csv(out_index, index=False)
    print('Wrote', out_npz, 'shape=', arr.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nlp', default='cohort/nlp_features.csv')
    parser.add_argument('--out-npz', default='cohort/note_embeddings/discharge_embeddings_full.npz')
    parser.add_argument('--out-index', default='cohort/note_embeddings/index_embeddings_full.csv')
    parser.add_argument('--model', default='all-MiniLM-L6-v2')
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    main(args.nlp, args.out_npz, args.out_index, model_name=args.model, limit=args.limit)
