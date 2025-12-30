import sqlite3
import pandas as pd
import numpy as np
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = ' '.join(text.split())
    return text

def main():
    base_dir = '/Users/phandanglinh/Desktop/VRES'
    db_path = os.path.join(base_dir, 'dataset/MIMIC_III.db')
    cohort_path = os.path.join(base_dir, 'cohort/features_phase4_clinical.csv')
    output_path = os.path.join(base_dir, 'cohort/text_tfidf_features.csv')
    
    print(f"Loading cohort from {cohort_path}...")
    df_cohort = pd.read_csv(cohort_path)
    # Standardize column names
    df_cohort.columns = [c.upper() if 'id' in c.lower() else c for c in df_cohort.columns]
    
    if 'HADM_ID' not in df_cohort.columns:
        print("Error: HADM_ID not found.")
        return
        
    # Clean IDs
    hadm_ids = df_cohort['HADM_ID'].dropna().astype(int).unique().tolist()
    print(f"Cohort Size: {len(hadm_ids)} unique admissions.")
    
    conn = sqlite3.connect(db_path)
    
    # 1. Inspect Categories (Debug)
    print("Checking Categories in DB...")
    cats = pd.read_sql_query("SELECT CATEGORY, COUNT(*) as C FROM NOTEEVENTS GROUP BY CATEGORY", conn)
    print(cats)
    
    # 2. Get ICU OUTTIME (Leakage Barrier)
    print("Fetching ICUSTAYS...")
    # Fetch all relevant stays
    id_str = ",".join(map(str, hadm_ids))
    # Chunking query for IDs to avoid limit
    outtime_data = []
    chunk_size = 2000
    for i in range(0, len(hadm_ids), chunk_size):
        chunk = hadm_ids[i:i+chunk_size]
        start_str = ",".join(map(str, chunk))
        q = f"SELECT HADM_ID, OUTTIME FROM ICUSTAYS WHERE HADM_ID IN ({start_str})"
        outtime_data.append(pd.read_sql_query(q, conn))
        
    df_icu = pd.concat(outtime_data, ignore_index=True)
    df_icu['OUTTIME'] = pd.to_datetime(df_icu['OUTTIME'])
    
    # Map HADM_ID -> Max Outtime (Conservative: Filter anything after the *last* ICU discharge of that admission)
    stay_time_map = df_icu.groupby('HADM_ID')['OUTTIME'].max().to_dict()
    print(f"Mapped discharge times for {len(stay_time_map)} admissions.")
    
    # 3. Fetch Notes
    print("Fetching Nursing/Physician notes...")
    notes_data = []
    
    # We use LIKE to catch 'Nursing', 'Nursing/other', 'Physician', 'Physician ' etc.
    # And chunk by ID
    
    for i in range(0, len(hadm_ids), chunk_size):
        if i % 2000 == 0:
            print(f"  Processing note chunk {i}/{len(hadm_ids)}...")
        
        chunk = hadm_ids[i:i+chunk_size]
        chunk_str = ",".join(map(str, chunk))
        
        # Query
        q_notes = f"""
        SELECT HADM_ID, CATEGORY, CHARTTIME, TEXT, ISERROR
        FROM NOTEEVENTS 
        WHERE HADM_ID IN ({chunk_str}) 
          AND (CATEGORY LIKE 'Nursing%' OR CATEGORY LIKE 'Physician%')
        """
        notes_data.append(pd.read_sql_query(q_notes, conn))
        
    conn.close()
    
    if not notes_data:
        print("No notes fetched.")
        return
        
    df_notes = pd.concat(notes_data, ignore_index=True)
    print(f"Fetched {len(df_notes)} raw notes.")
    
    # Filter ISERROR (NULL or Empty)
    # Check if ISERROR column exists and has content
    if 'ISERROR' in df_notes.columns:
        # Keep only where ISERROR is NULL or ''
        mask_err = df_notes['ISERROR'].isna() | (df_notes['ISERROR'] == '')
        df_notes = df_notes[mask_err]
        print(f"Notes after ISERROR filter: {len(df_notes)}")
    
    if df_notes.empty:
        print("DataFrame is empty after concat. Check query.")
        return

    # 4. Filter by Time
    print("Filtering leakage...")
    df_notes['CHARTTIME'] = pd.to_datetime(df_notes['CHARTTIME'])
    df_notes['CUTOFF'] = df_notes['HADM_ID'].map(stay_time_map)
    df_notes['CUTOFF'] = pd.to_datetime(df_notes['CUTOFF'])
    
    # Drop NaT
    df_notes = df_notes.dropna(subset=['CHARTTIME', 'CUTOFF'])
    
    # Compare
    valid_mask = df_notes['CHARTTIME'] < df_notes['CUTOFF']
    df_clean = df_notes[valid_mask].copy()
    
    print(f"Retained {len(df_clean)} / {len(df_notes)} notes (Pre-Discharge).")
    
    if df_clean.empty:
        print("Zero notes passed time filter. Check timestamps!")
        print("Sample Chart:", df_notes['CHARTTIME'].iloc[0])
        print("Sample Cutoff:", df_notes['CUTOFF'].iloc[0])
        return
        
    # 5. Concatenate & TF-IDF
    print("Processing Text...")
    df_clean['TEXT'] = df_clean['TEXT'].astype(str).apply(preprocess_text)
    
    # Aggregation
    df_agg = df_clean.groupby('HADM_ID')['TEXT'].apply(lambda x: ' '.join(x)).reset_index()
    print(f"Aggregated texts for {len(df_agg)} admissions.")
    
    # Save Raw Text for Honest Training (Phase 18)
    raw_output_path = os.path.join(base_dir, 'cohort/phase17_honest_text.csv')
    df_agg.to_csv(raw_output_path, index=False)
    print(f"Saved {len(df_agg)} raw text rows to {raw_output_path}")

    # Vectorize (Legacy Leakage Path - kept for compatibility or reference if needed, but we rely on raw text now)
    print("Vectorizing (Legacy Global Mode)...")
    X = vectorizer.fit_transform(df_agg['TEXT'])
    names = [f"tfidf_{n}" for n in vectorizer.get_feature_names_out()]
    
    df_tfidf = pd.DataFrame(X.toarray(), columns=names)
    df_tfidf['HADM_ID'] = df_agg['HADM_ID']
    
    # Save
    df_tfidf.to_csv(output_path, index=False)
    print(f"Saved {len(df_tfidf)} rows to {output_path}")

if __name__ == "__main__":
    main()
