import pandas as pd
import numpy as np
import sqlite3
import os

def main():
    db_path = 'dataset/MIMIC_III.db'
    output_path = 'cohort/features_phase11_extra.csv'
    cohort_path = 'cohort/new_cohort_icu_readmission_labels.csv'
    
    if not os.path.exists(db_path):
        print(f"DB not found: {db_path}")
        return
        
    print(f"Connecting to {db_path}...")
    conn = sqlite3.connect(db_path)
    
    # 1. Load Cohort to get HADM_IDs and ICUSTAY_IDs
    print("Loading cohort...")
    cohort = pd.read_csv(cohort_path)
    # Ensure ID columns are int
    if 'HADM_ID' in cohort.columns:
        cohort['HADM_ID'] = cohort['HADM_ID'].astype(int)
    
    hadm_ids = cohort['HADM_ID'].unique()
    hadm_str = ",".join(map(str, hadm_ids))
    
    print(f"Cohort size: {len(hadm_ids)} admissions.")
    
    # 2. Get OUTTIME for each ICUSTAY to filter events
    # We need ICU_OUTTIME. Usually in ICUSTAYS table.
    print("Fetching ICU Outtimes...")
    q_stays = f"""
    SELECT HADM_ID, ICUSTAY_ID, OUTTIME
    FROM ICUSTAYS
    WHERE HADM_ID IN ({hadm_str})
    """
    icustays = pd.read_sql_query(q_stays, conn)
    icustays['OUTTIME'] = pd.to_datetime(icustays['OUTTIME'])
    
    # Keep last stay per HADM_ID if multiple? Readmission usually concerns the *last* ICU stay.
    # Sorted by OUTTIME desc
    icustays = icustays.sort_values('OUTTIME', ascending=False).drop_duplicates('HADM_ID')
    
    # Create temp table for easier joining? 
    # Or just iterate. SQL extraction for 3500 patients can be done in batch.
    # Let's use left join logic in pandas if we extract raw events, but raw events are huge.
    # Better to aggregate in SQL if possible, but SQLite date diffs are tricky.
    # Let's extract raw events for these HADM_IDs and filter in Pandas.
    
    # ---------------------------------------------------------
    # 3. URINE OUTPUT
    # ITEMIDs typically: 40055, 43175, 40069, 40094, 40715, 40473, 40085, 40056, 40056, 40405, 40428, 40086, 40096, 40651, 226559, 226560, 226561, 226584, 226563, 226564, 226565, 226567, 227510, 227488, 227489
    # Common: 40055 (Urine Out Foley), 226559 (Foley)
    # Let's select ALL Output items for now or broad range?
    # Better: Select * from OUTPUTEVENTS where HADM_ID in ...
    # warning: might be big.
    # Let's try select ITEMID, CHARTTIME, VALUE
    
    print("Extracting Urine Output events...")
    q_out = f"""
    SELECT HADM_ID, CHARTTIME, VALUE, ITEMID
    FROM OUTPUTEVENTS
    WHERE HADM_ID IN ({hadm_str})
    AND VALUE IS NOT NULL
    """
    # Note: This might fetch a few million rows. 
    # Chunking might be needed.
    outputs = pd.read_sql_query(q_out, conn)
    print(f"Fetched {len(outputs)} output events. Processing...")
    
    outputs['CHARTTIME'] = pd.to_datetime(outputs['CHARTTIME'])
    outputs['VALUE'] = pd.to_numeric(outputs['VALUE'], errors='coerce').fillna(0)
    
    # Ensure HADM_ID types match
    outputs['HADM_ID'] = outputs['HADM_ID'].astype(int)
    icustays['HADM_ID'] = icustays['HADM_ID'].astype(int)
    
    # Merge with OUTTIME
    outputs = outputs.merge(icustays[['HADM_ID', 'OUTTIME']], on='HADM_ID', how='inner')
    
    # Filter 24h before OUTTIME
    # Time delta
    outputs['diff_hours'] = (outputs['OUTTIME'] - outputs['CHARTTIME']).dt.total_seconds() / 3600
    
    # Keep interval [0, 24] hours before discharge
    mask_24h = (outputs['diff_hours'] >= 0) & (outputs['diff_hours'] <= 24)
    # Also considering "Whole Stay" sum is useful? No, last 24h indicates kidney recovery status.
    
    out_24h = outputs[mask_24h]
    
    # Group by HADM_ID
    # Sum of Volume
    urine_feats = out_24h.groupby('HADM_ID')['VALUE'].sum().reset_index()
    urine_feats.columns = ['HADM_ID', 'URINE_OUT_24H']
    
    # ---------------------------------------------------------
    # 4. ANTIBIOTICS (Infection Proxy)
    # Since MICROBIOLOGYEVENTS is missing, we use PRESCRIPTIONS.
    # Look for common antibiotics.
    print("Extracting Antibiotic Prescriptions...")
    
    # Common Anti-infectives
    antibiotics_list = [
        'Vancomycin', 'Piperacillin', 'Cefepime', 'Meropenem', 'Levofloxacin',
        'Ciprofloxacin', 'Ceftriaxone', 'Metronidazole', 'Azithromycin',
        'Gentamicin', 'Tobramycin', 'Ampicillin', 'Nafcillin', 'Oxacillin'
    ]
    # In SQL LIKE '%...%' is safer.
    
    # We select Drug Name and StartDate
    q_pres = f"""
    SELECT HADM_ID, STARTDATE, ENDDATE, DRUG
    FROM PRESCRIPTIONS
    WHERE HADM_ID IN ({hadm_str})
    """
    pres = pd.read_sql_query(q_pres, conn)
    print(f"Fetched {len(pres)} prescriptions.")
    
    pres['STARTDATE'] = pd.to_datetime(pres['STARTDATE'])
    pres['ENDDATE'] = pd.to_datetime(pres['ENDDATE'])
    
    # Filter for Antibiotics
    # Regex filter is faster in pandas
    mask_abx = pres['DRUG'].str.contains('|'.join(antibiotics_list), case=False, na=False)
    pres_abx = pres[mask_abx]
    print(f"  Found {len(pres_abx)} antibiotic orders.")
    
    # Ensure ID match
    pres_abx['HADM_ID'] = pres_abx['HADM_ID'].astype(int)
    pres_abx = pres_abx.merge(icustays[['HADM_ID', 'OUTTIME']], on='HADM_ID', how='inner')
    
    # Active in last 24h?
    # Logic: STARTDATE <= OUTTIME and ENDDATE >= (OUTTIME - 24h)
    # Or just "Order active at discharge?"
    
    # Active around discharge time (OUTTIME)
    # Let's say: Is the patient on antibiotics within the last 24 hours of stay?
    # Overlap logic:
    # Range [START, END] overlaps with [OUTTIME-24h, OUTTIME]
    
    window_start = pres_abx['OUTTIME'] - pd.Timedelta(hours=24)
    window_end = pres_abx['OUTTIME']
    
    # Overlap: Not (End < WindowStart OR Start > WindowEnd)
    # => End >= WindowStart AND Start <= WindowEnd
    
    has_overlap = (pres_abx['ENDDATE'] >= window_start) & (pres_abx['STARTDATE'] <= window_end)
    
    pres_active = pres_abx[has_overlap]
    
    # Feature: Count of Active Antibiotics + Binary "On Antibiotics"
    abx_feats = pres_active.groupby('HADM_ID')['DRUG'].nunique().reset_index(name='ACTIVE_ANTIBIOTICS_COUNT')
    abx_feats['ON_ANTIBIOTICS'] = 1
    
    # 5. Merge Features
    print("Merging new features...")
    features = pd.DataFrame({'HADM_ID': hadm_ids})
    features = features.merge(urine_feats, on='HADM_ID', how='left')
    features = features.merge(abx_feats, on='HADM_ID', how='left')
    
    features = features.fillna(0)
    
    print(f"Saving to {output_path}...")
    features.to_csv(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
