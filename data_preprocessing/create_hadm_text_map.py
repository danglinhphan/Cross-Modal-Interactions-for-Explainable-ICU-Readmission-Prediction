"""Build HADM_ID -> DISCHARGE_SUMMARY_TEXT mapping and filter a cohort CSV.

Usage:
    python create_hadm_text_map.py --cohort cohort/new_cohort_icu_readmission.csv \
        --note-files "cohort/filtered_cohort_with_discharge_summary_test.csv,cohort/filtered_cohort_with_discharge_summary_test_batch.csv" \
        --out-map cohort/hadm_to_text.pkl --out-cohort cohort/new_cohort_icu_readmission_filtered.csv

Features:
- Reads note CSVs in chunks for memory efficiency.
- Ensures HADM_ID and DISCHARGE_SUMMARY_TEXT columns exist and handles quoted multi-line text.
- Handles duplicates: keeps the longest text for a HADM_ID (configurable) or concatenates.
- Filters input cohort to only HADM_IDs found in the map and optionally ensures unique HADM_ID rows.
- Optionally persist the map to a pickle file for reuse.
"""
import argparse
import os
import pickle
from pathlib import Path
import pandas as pd
import gc
from typing import Dict, Iterable, Optional
import numpy as np


def build_map(note_files: Iterable[str], usecols=('HADM_ID', 'DISCHARGE_SUMMARY_TEXT'), chunksize: int = 5000,
              dedup_strategy: str = 'keep_longest') -> Dict[int, str]:
    """Build a map of HADM_ID -> DISCHARGE_SUMMARY_TEXT from one or more CSV files.

    dedup_strategy: 'keep_longest' | 'concat' | 'keep_first'
    """
    id_to_text: Dict[int, str] = {}
    for f in note_files:
        p = Path(f)
        if not p.exists():
            print(f"Warning: note file {f} does not exist, skipping.")
            continue
        print(f"Processing note file: {f}")
        for chunk in pd.read_csv(f, usecols=list(usecols), chunksize=chunksize):
            # Drop rows where either HADM_ID or TEXT is missing
            chunk = chunk.dropna(subset=['HADM_ID', 'DISCHARGE_SUMMARY_TEXT'])
            # Ensure integer HADM_ID
            try:
                chunk['HADM_ID'] = chunk['HADM_ID'].astype(int)
            except Exception:
                # attempt coercion
                chunk['HADM_ID'] = pd.to_numeric(chunk['HADM_ID'], errors='coerce').astype('Int64')
                chunk = chunk.dropna(subset=['HADM_ID'])
                chunk['HADM_ID'] = chunk['HADM_ID'].astype(int)

            for hadm_id, text in zip(chunk['HADM_ID'], chunk['DISCHARGE_SUMMARY_TEXT'].astype(str)):
                text = text.strip()
                if text == '':
                    continue
                if hadm_id not in id_to_text:
                    id_to_text[hadm_id] = text
                else:
                    if dedup_strategy == 'keep_longest':
                        if len(text) > len(id_to_text[hadm_id]):
                            id_to_text[hadm_id] = text
                    elif dedup_strategy == 'concat':
                        if text not in id_to_text[hadm_id]:
                            id_to_text[hadm_id] = id_to_text[hadm_id] + '\n\n' + text
                    # keep_first does nothing
            gc.collect()
        print(f"Done processing {f}")
    return id_to_text


def filter_cohort(cohort_csv: str, id_to_text: Dict[int, str], out_csv: Optional[str] = None,
                  ensure_unique_hadm: bool = True) -> pd.DataFrame:
    """Filter cohort entries to only those HADM_ID present in id_to_text.

    If ensure_unique_hadm is True, drop duplicates on HADM_ID keeping the first.
    """
    print(f"Loading cohort from {cohort_csv}")
    cohort_df = pd.read_csv(cohort_csv)
    print(f"Original cohort size: {len(cohort_df)}")
    hadm_set = set(id_to_text.keys())
    cohort_filtered = cohort_df[cohort_df['HADM_ID'].isin(hadm_set)].reset_index(drop=True)
    print(f"Filtered cohort size (HADM present in map): {len(cohort_filtered)}")
    if ensure_unique_hadm:
        if cohort_filtered['HADM_ID'].duplicated().any():
            print("Duplicates found for HADM_ID in filtered cohort. Dropping duplicate rows (keeping first).")
            cohort_filtered = cohort_filtered.drop_duplicates(subset=['HADM_ID'], keep='first').reset_index(drop=True)
        else:
            print("All HADM_IDs are unique in filtered cohort.")

    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        cohort_filtered.to_csv(out_csv, index=False)
        print(f"Wrote filtered cohort to {out_csv}")
    return cohort_filtered


def map_to_nlp_features(id_to_text: Dict[int, str], model_name: str = 'all-MiniLM-L6-v2',
                        batch_size: int = 64, out_csv: Optional[str] = None,
                        cohort_hadm_ids: Optional[Iterable[int]] = None, fill_missing: bool = False) -> pd.DataFrame:
    """Convert a HADM_ID->text map into numeric NLP features.

    If the `sentence_transformers` package is available, generate dense sentence embeddings
    (one vector per HADM_ID). Otherwise, fall back to simple numeric textual features:
    length, word count, unique token count.
    """
    # If cohort_hadm_ids is provided and fill_missing True, ensure all those IDs exist in output
    if cohort_hadm_ids is not None and fill_missing:
        hadm_ids = list(cohort_hadm_ids)
        texts = [id_to_text.get(h, '') for h in hadm_ids]
    else:
        hadm_ids = list(id_to_text.keys())
        texts = [id_to_text[h] for h in hadm_ids]

    # Try to use sentence-transformers for embeddings
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        print(f"Computing embeddings for {len(texts)} texts with {model_name} (batch_size={batch_size})")
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        n_dim = embeddings.shape[1]
        col_names = [f'nlp_{i}' for i in range(n_dim)]
        emb_df = pd.DataFrame(embeddings, columns=col_names)
        emb_df.insert(0, 'HADM_ID', hadm_ids)
        if out_csv:
            Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
            emb_df.to_csv(out_csv, index=False)
            print(f"Wrote NLP features to {out_csv}")
        return emb_df
    except Exception as e:
        print('sentence_transformers model not available or failed. Falling back to simple text features. Error:', e)
        # fallback features: text length, number of words, number of unique tokens
        # Use the project's preprocessing utilities to normalize and tokenize text (without stopword removal)
        from data_preprocessing import preprocess
        feats = []
        for t in texts:
            length = len(t)
            words = preprocess(t, remove_stopwords_flag=False, return_tokens=True)
            word_count = len(words)
            unique_tokens = len(set(words))
            feats.append([length, word_count, unique_tokens])
        feats = np.asarray(feats)
        col_names = ['nlp_len', 'nlp_word_count', 'nlp_unique_tokens']
        feat_df = pd.DataFrame(feats, columns=col_names)
        feat_df.insert(0, 'HADM_ID', hadm_ids)
        if out_csv:
            Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
            feat_df.to_csv(out_csv, index=False)
            print(f"Wrote simple NLP features to {out_csv}")
        return feat_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', type=str, default='cohort/new_cohort_icu_readmission.csv')
    parser.add_argument('--note-files', type=str,
                        default='cohort/filtered_cohort_with_discharge_summary_test.csv,cohort/filtered_cohort_with_discharge_summary_test_batch.csv')
    parser.add_argument('--chunksize', type=int, default=5000)
    parser.add_argument('--dedup', type=str, default='keep_longest', choices=['keep_longest', 'concat', 'keep_first'])
    parser.add_argument('--out-map', type=str, default='cohort/hadm_to_text.pkl')
    parser.add_argument('--out-cohort', type=str, default='cohort/new_cohort_icu_readmission_filtered.csv')
    parser.add_argument('--save-map', action='store_true', help='Persist hadm->text map to disk')
    parser.add_argument('--to-nlp-csv', action='store_true', help='Convert HADM_ID->text map to NLP features CSV (embeddings or fallback)')
    parser.add_argument('--nlp-out', type=str, default=None, help='Output path for generated NLP feature CSV')
    parser.add_argument('--embed-model', type=str, default='all-MiniLM-L6-v2', help='SentenceTransformer model name to use for embeddings')
    parser.add_argument('--embed-batch', type=int, default=64, help='Batch size for embedding computation')
    parser.add_argument('--fill-missing', action='store_true', help='Include all cohort HADM_IDs in NLP features, filling missing text with empty features')
    parser.add_argument('--cohort-for-nlp', type=str, default=None, help='Cohort CSV to use for listing all HADM_IDs to include in NLP CSV (default uses provided --cohort)')
    args = parser.parse_args()

    note_files = [s.strip() for s in args.note_files.split(',') if s.strip()]
    id_to_text = build_map(note_files, chunksize=args.chunksize, dedup_strategy=args.dedup)
    print(f"Total HADM_IDs with text: {len(id_to_text)}")

    # Optional persist
    if args.save_map:
        out_map = Path(args.out_map)
        out_map.parent.mkdir(parents=True, exist_ok=True)
        with open(out_map, 'wb') as fh:
            pickle.dump(id_to_text, fh)
        print(f"Saved id->text map to {out_map}")

    # Filter cohort and (optionally) persist
    filtered = filter_cohort(args.cohort, id_to_text, out_csv=args.out_cohort)
    print(f"Filtered cohort rows: {len(filtered)}")

    # Optional: generate NLP features CSV from map
    if args.to_nlp_csv:
        nlp_out = args.nlp_out or os.path.join(Path(args.out_cohort).parent, 'nlp_features.csv')
        print(f'Converting map to NLP features and writing to {nlp_out}')
        # Determine cohort HADM list if requested
        cohort_for_nlp = args.cohort_for_nlp or args.cohort
        cohort_hadm_ids = None
        if args.fill_missing:
            cohort_df = pd.read_csv(cohort_for_nlp, usecols=['HADM_ID'])
            cohort_hadm_ids = [int(x) for x in cohort_df['HADM_ID'].tolist()]
            print(f'Including {len(cohort_hadm_ids)} cohort HADM_IDs in NLP CSV (fill_missing=True)')
        map_to_nlp_features(id_to_text, model_name=args.embed_model, batch_size=args.embed_batch, out_csv=nlp_out, cohort_hadm_ids=cohort_hadm_ids, fill_missing=args.fill_missing)


if __name__ == '__main__':
    main()
