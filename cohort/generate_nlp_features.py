"""Generate a cleaned nlp_features.csv mapping HADM_ID -> CLEAN_TEXT

This script reads a cohort CSV with DISCHARGE_SUMMARY_TEXT and creates a
`nlp_features.csv` with columns: HADM_ID, CLEAN_TEXT. Cleaning steps:
 - Remove de-identification tokens like [** ... **]
 - Normalize whitespace (replace newlines and multiple spaces)
 - Remove common labels like Admission Date and Discharge Date
 - Collapse repeated punctuation
 - Join multiple rows for the same HADM_ID
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from typing import Optional

try:
    import pandas as pd
except Exception:
    pd = None


DEID_RE = re.compile(r"\[\*\*.*?\*\*\]", flags=re.DOTALL)
MULTI_WS_RE = re.compile(r"\s+")
REPEAT_PUNCT_RE = re.compile(r"([.?!,;:\-])\1{1,}")
HTML_TAG_RE = re.compile(r"<[^>]+>")


def clean_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    text = str(text)
    # Remove de-identification tokens like [** ... **]
    text = DEID_RE.sub(" ", text)
    # Remove any stray HTML or tags
    text = HTML_TAG_RE.sub(" ", text)
    # Remove common labels that carry dates/timestamps
    text = re.sub(r"Admission Date:\s*[^\n]*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"Discharge Date:\s*[^\n]*", " ", text, flags=re.IGNORECASE)
    # Remove repeated punctuation (e.g. "....." -> "...") then compress to single if necessary
    text = REPEAT_PUNCT_RE.sub(lambda m: m.group(1), text)
    # Replace newline and tabs with spaces
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    # Normalize whitespace
    text = MULTI_WS_RE.sub(" ", text)
    # Trim
    text = text.strip()
    # Final defensive cutoff: if text is only short garbage, return empty
    if len(text) <= 2:
        return ""
    return text





def generate_nlp_features(src_csv: list[str], out_csv: str):
    # src_csv can be a list of file paths
    if pd is not None:
        # Read with pandas to correctly parse quoted multiline fields
        # Read and concatenate multiple files if provided
        dfs = []
        for f in src_csv:
            if not os.path.exists(f):
                print(f"Warning: source file {f} does not exist, skipping")
                continue
            dfs.append(pd.read_csv(f, dtype={"HADM_ID": object}, keep_default_na=False))
        if not dfs:
            raise ValueError("No valid source CSV files provided for input")
        df = pd.concat(dfs, ignore_index=True)
        if "HADM_ID" not in df.columns or "DISCHARGE_SUMMARY_TEXT" not in df.columns:
            raise ValueError(f"Required columns not found in {src_csv}. Found: {df.columns.tolist()}")
        # Keep only rows where DISCHARGE_SUMMARY_TEXT is present (non-empty)
        df["DISCHARGE_SUMMARY_TEXT"] = df["DISCHARGE_SUMMARY_TEXT"].astype(str)
        df = df[df["DISCHARGE_SUMMARY_TEXT"].str.strip().astype(bool)].copy()
        # Clean: choose method from flags
        # default will be using existing clean_text unless callers pass in other method
        clean_method = generate_nlp_features.clean_method if hasattr(generate_nlp_features, 'clean_method') else 'default'
        df["CLEAN_TEXT"] = df["DISCHARGE_SUMMARY_TEXT"].apply(clean_text)
        # Drop empty CLEAN_TEXT rows
        df = df[df["CLEAN_TEXT"].str.strip().astype(bool)].copy()
        # Dedup by HADM_ID: default to keep longest (as requested by user)
        dedup = generate_nlp_features.dedup if hasattr(generate_nlp_features, 'dedup') else 'keep_longest'
        if dedup == 'keep_longest':
            # compute text length, sort descending and dedup keeping longest
            df['text_len'] = df['CLEAN_TEXT'].str.len()
            df = df.sort_values('text_len', ascending=False)
            dedup_df = df.drop_duplicates(subset=['HADM_ID'], keep='first')
            grouped = dedup_df.groupby('HADM_ID')['CLEAN_TEXT'].first().reset_index()
        elif dedup == 'keep_first':
            grouped = df.groupby('HADM_ID')['CLEAN_TEXT'].first().reset_index()
        else:  # concat
            grouped = df.groupby('HADM_ID')['CLEAN_TEXT'].agg(lambda series: ' '.join(dict.fromkeys(series))).reset_index()
        out_df = grouped
        # Save to CSV with HADM_ID as-is and CLEAN_TEXT
        # Ensure exactly two columns HADM_ID and CLEAN_TEXT
        out_df = out_df[['HADM_ID', 'CLEAN_TEXT']]
        # Backup existing out file if present
        if os.path.exists(out_csv):
            bak = out_csv + ".bak"
            print(f"Backing up existing {out_csv} to {bak}")
            try:
                os.replace(out_csv, bak)
            except Exception:
                pass
        out_df.to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    else:
        # Fallback to csv module if pandas not available: careful about quoted multiline fields
        # For csv fallback: iterate multiple files
        mapping = {}
        for fpath in src_csv:
            if not os.path.exists(fpath):
                print(f"Warning: source file {fpath} does not exist, skipping")
                continue
            with open(fpath, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if "HADM_ID" not in reader.fieldnames or "DISCHARGE_SUMMARY_TEXT" not in reader.fieldnames:
                    raise ValueError(f"Required columns not found in {fpath}. Found: {reader.fieldnames}")
                for row in reader:
                    hadm = row.get("HADM_ID")
                    text = row.get("DISCHARGE_SUMMARY_TEXT") or ""
                    if not text.strip():
                        continue
                    # choose cleaning method
                    clean_method = generate_nlp_features.clean_method if hasattr(generate_nlp_features, 'clean_method') else 'default'
                    cleaned = clean_text(text)
                    if not cleaned:
                        continue
                    mapping.setdefault(hadm, [])
                    mapping[hadm].append(cleaned)
        # At the end of reading all files, mapping holds candidate texts per HADM
            # loop continues to next src file
        # Combine unique texts per HADM
        # dedup for csv-stack fallback
        out_rows = []
        dedup = generate_nlp_features.dedup if hasattr(generate_nlp_features, 'dedup') else 'keep_longest'
        for hadm, texts in mapping.items():
            if dedup == 'keep_longest':
                longest = max(texts, key=lambda t: len(t))
                out_rows.append((hadm, longest))
            elif dedup == 'keep_first':
                out_rows.append((hadm, texts[0]))
            else:
                out_rows.append((hadm, ' '.join(dict.fromkeys(texts))))
        # Backup existing out file if present
        if os.path.exists(out_csv):
            bak = out_csv + ".bak"
            print(f"Backing up existing {out_csv} to {bak}")
            try:
                os.replace(out_csv, bak)
            except Exception:
                pass
        with open(out_csv, "w", newline="", encoding="utf-8") as fw:
            writer = csv.writer(fw)
            writer.writerow(["HADM_ID", "CLEAN_TEXT"]) 
            for hadm, text in out_rows:
                writer.writerow([hadm, text])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help="Source cohort CSV(s) (comma separated) with DISCHARGE_SUMMARY_TEXT",
                        default="cohort/filtered_cohort_with_discharge_summary_test_latest.csv")
    parser.add_argument("--out", help="Output CSV path",
                        default="cohort/nlp_features.csv")
    parser.add_argument('--clean-method', choices=['default'], default='default',
                        help='Cleaning method to use for CLEAN_TEXT (default=default).')
    parser.add_argument('--dedup', choices=['keep_longest', 'keep_first', 'concat'], default='keep_longest',
                        help='Deduplication strategy for multiple notes per HADM_ID')
    args = parser.parse_args()

    # Support comma-separated list of source files
    src_list = [s.strip() for s in args.src.split(',') if s.strip()]
    src = [os.path.abspath(s) for s in src_list]
    out = os.path.abspath(args.out)
    print(f"Reading from {src}")
    # store options on function object for easier use within helper
    generate_nlp_features.clean_method = args.clean_method
    generate_nlp_features.dedup = args.dedup
    generate_nlp_features(src, out)
    print(f"Wrote cleaned nlp features to {out}")


if __name__ == "__main__":
    main()
