"""Extract clinical notes written BEFORE ICU OUTTIME.

This script extracts notes from NOTEEVENTS that were recorded BEFORE the patient's
first ICU OUTTIME, avoiding data leakage from Discharge Summary which is written
AFTER the patient leaves ICU/hospital.

Included note categories:
- Nursing/other (96% written before ICU OUT)
- Nursing (92.3% written before ICU OUT)  
- Physician (92.1% written before ICU OUT)

These notes contain clinical observations and assessments made during the ICU stay,
which can legitimately be used to predict ICU readmission risk.

Usage:
  python data_preprocessing/extract_notes_before_icu_out.py

Output:
  - cohort/notes_before_icu_out.csv: HADM_ID, CLEAN_TEXT (concatenated notes before ICU OUT)
"""

import argparse
import csv
import os
import re
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

# Constants
ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / 'dataset' / 'MIMIC_III.db'
COHORT_DIR = ROOT / 'cohort'

# Note categories to include (high % written before ICU OUTTIME)
SAFE_CATEGORIES = [
    'Nursing/other',  # 96.0% before ICU OUT
    'Nursing',        # 92.3% before ICU OUT
    'Physician ',     # 92.1% before ICU OUT (note the space in original data)
]

# Text cleaning patterns
DEID_RE = re.compile(r"\[\*\*.*?\*\*\]", flags=re.DOTALL)
MULTI_WS_RE = re.compile(r"\s+")
REPEAT_PUNCT_RE = re.compile(r"([.?!,;:\-])\1{1,}")
HTML_TAG_RE = re.compile(r"<[^>]+>")


def clean_text(text: Optional[str]) -> str:
    """Clean clinical note text for NLP processing."""
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
    # Remove repeated punctuation
    text = REPEAT_PUNCT_RE.sub(lambda m: m.group(1), text)
    # Replace newline and tabs with spaces
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    # Normalize whitespace
    text = MULTI_WS_RE.sub(" ", text)
    # Trim
    text = text.strip()
    # Final defensive cutoff
    if len(text) <= 2:
        return ""
    return text


def clean_text_for_bert(text: Optional[str]) -> str:
    """Apply stricter cleaning for BERT-style models."""
    if text is None:
        return ""
    text = str(text)
    # Remove deid tokens
    text = re.sub(r"\[\*\*.*?\*\*\]", " ", text)
    # Keep only letters, numbers and punctuation
    text = re.sub(r"[^a-zA-Z0-9\s\.,;:%\-\(\)]", "", text)
    # Lowercase
    text = text.lower()
    # Normalize whitespace
    text = MULTI_WS_RE.sub(" ", text).strip()
    return text


def get_first_icu_stays(conn: sqlite3.Connection) -> pd.DataFrame:
    """Get first ICU stay for each patient in cohort (same criteria as original cohort)."""
    query = '''
    WITH first_icu AS (
        SELECT SUBJECT_ID, HADM_ID, ICUSTAY_ID, INTIME, OUTTIME,
               ROW_NUMBER() OVER (PARTITION BY SUBJECT_ID ORDER BY INTIME) AS rn
        FROM ICUSTAYS
    )
    SELECT fi.SUBJECT_ID, fi.HADM_ID, fi.ICUSTAY_ID, fi.INTIME, fi.OUTTIME,
           a.ADMITTIME, a.DISCHTIME
    FROM first_icu fi
    JOIN PATIENTS p ON p.SUBJECT_ID = fi.SUBJECT_ID
    JOIN ADMISSIONS a ON a.HADM_ID = fi.HADM_ID
    WHERE fi.rn = 1
      AND ROUND((julianday(a.ADMITTIME) - julianday(p.DOB)) / 365.25, 0) >= 18
      AND (a.DEATHTIME IS NULL OR a.DEATHTIME NOT BETWEEN fi.INTIME AND fi.OUTTIME)
    '''
    return pd.read_sql_query(query, conn)


def extract_notes_before_icu_out(
    conn: sqlite3.Connection,
    icu_stays: pd.DataFrame,
    categories: list = None,
    max_notes_per_hadm: int = 50,
    max_text_length: int = 50000
) -> pd.DataFrame:
    """
    Extract notes written BEFORE ICU OUTTIME for each HADM_ID.
    
    Args:
        conn: SQLite connection
        icu_stays: DataFrame with HADM_ID, OUTTIME columns
        categories: List of note categories to include
        max_notes_per_hadm: Maximum number of notes to include per admission
        max_text_length: Maximum total text length per admission
    
    Returns:
        DataFrame with HADM_ID, CLEAN_TEXT columns
    """
    if categories is None:
        categories = SAFE_CATEGORIES
    
    # Build category filter using LIKE to handle trailing spaces in MIMIC data
    cat_conditions = ' OR '.join(f"TRIM(n.CATEGORY) = '{c.strip()}'" for c in categories)
    
    # Query notes with time filter
    query = f'''
    SELECT n.HADM_ID, TRIM(n.CATEGORY) as CATEGORY, n.CHARTTIME, n.TEXT
    FROM NOTEEVENTS n
    WHERE ({cat_conditions})
      AND n.TEXT IS NOT NULL
      AND n.TEXT != ''
      AND (n.ISERROR IS NULL OR n.ISERROR = '' OR n.ISERROR = '0')
    ORDER BY n.HADM_ID, n.CHARTTIME
    '''
    
    print(f"Querying notes from categories: {categories}")
    notes_df = pd.read_sql_query(query, conn)
    print(f"Total notes retrieved: {len(notes_df)}")
    
    # Convert dates
    notes_df['CHARTTIME'] = pd.to_datetime(notes_df['CHARTTIME'], errors='coerce')
    icu_stays['OUTTIME'] = pd.to_datetime(icu_stays['OUTTIME'], errors='coerce')
    
    # Create HADM -> OUTTIME mapping
    outtime_map = dict(zip(icu_stays['HADM_ID'].astype(str), icu_stays['OUTTIME']))
    
    # Filter notes to only those before ICU OUTTIME
    results = []
    hadm_groups = notes_df.groupby('HADM_ID')
    
    print("Filtering notes to only those before ICU OUTTIME...")
    for hadm_id, group in hadm_groups:
        hadm_str = str(hadm_id)
        if hadm_str not in outtime_map:
            continue
            
        outtime = outtime_map[hadm_str]
        if pd.isna(outtime):
            continue
        
        # Filter notes before ICU OUTTIME
        before_out = group[group['CHARTTIME'] < outtime].copy()
        
        if before_out.empty:
            continue
        
        # Sort by time and limit notes
        before_out = before_out.sort_values('CHARTTIME').head(max_notes_per_hadm)
        
        # Clean and concatenate texts
        texts = []
        total_len = 0
        for _, row in before_out.iterrows():
            cleaned = clean_text(row['TEXT'])
            if cleaned and total_len + len(cleaned) <= max_text_length:
                texts.append(cleaned)
                total_len += len(cleaned)
        
        if texts:
            combined_text = ' '.join(texts)
            results.append({
                'HADM_ID': hadm_id,
                'CLEAN_TEXT': combined_text
            })
    
    result_df = pd.DataFrame(results)
    print(f"Extracted notes for {len(result_df)} HADM_IDs")
    return result_df


def main():
    parser = argparse.ArgumentParser(
        description='Extract clinical notes written BEFORE ICU OUTTIME'
    )
    parser.add_argument(
        '--output', '-o',
        default=str(COHORT_DIR / 'notes_before_icu_out.csv'),
        help='Output CSV path'
    )
    parser.add_argument(
        '--max-notes', type=int, default=50,
        help='Maximum notes per HADM_ID'
    )
    parser.add_argument(
        '--max-text-length', type=int, default=50000,
        help='Maximum total text length per HADM_ID'
    )
    parser.add_argument(
        '--categories', nargs='+', default=SAFE_CATEGORIES,
        help='Note categories to include'
    )
    args = parser.parse_args()
    
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}")
    
    print(f"Connecting to database: {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    
    try:
        # Get first ICU stays
        print("Getting first ICU stays...")
        icu_stays = get_first_icu_stays(conn)
        print(f"Found {len(icu_stays)} first ICU stays")
        
        # Extract notes before ICU OUT
        result_df = extract_notes_before_icu_out(
            conn, icu_stays,
            categories=args.categories,
            max_notes_per_hadm=args.max_notes,
            max_text_length=args.max_text_length
        )
        
        # Save to CSV
        output_path = args.output
        if os.path.exists(output_path):
            backup = output_path + '.bak'
            print(f"Backing up existing file to {backup}")
            os.replace(output_path, backup)
        
        result_df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"Saved {len(result_df)} records to {output_path}")
        
        # Print statistics
        print("\n" + "="*60)
        print("STATISTICS")
        print("="*60)
        print(f"Total HADM_IDs with notes: {len(result_df)}")
        if len(result_df) > 0:
            print(f"Avg text length: {result_df['CLEAN_TEXT'].str.len().mean():.0f} chars")
            print(f"Min text length: {result_df['CLEAN_TEXT'].str.len().min()}")
            print(f"Max text length: {result_df['CLEAN_TEXT'].str.len().max()}")
        else:
            print("No notes extracted!")
        
    finally:
        conn.close()


if __name__ == '__main__':
    main()
