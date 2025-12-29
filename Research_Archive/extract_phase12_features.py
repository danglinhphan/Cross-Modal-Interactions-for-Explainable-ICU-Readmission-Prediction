import pandas as pd
import numpy as np
import os
import gc

def main():
    base_dir = '/Users/phandanglinh/Desktop/VRES'
    extra_dir = os.path.join(base_dir, 'dataset/additional')
    output_path = os.path.join(base_dir, 'cohort/features_phase12_extra.csv')
    cohort_path = os.path.join(base_dir, 'cohort/new_cohort_icu_readmission_labels.csv')
    
    print("Loading cohort...")
    cohort = pd.read_csv(cohort_path)
    if 'HADM_ID' in cohort.columns:
        cohort['HADM_ID'] = cohort['HADM_ID'].astype(int)
    
    target_hadms = set(cohort['HADM_ID'].unique())
    print(f"Target Admissions: {len(target_hadms)}")
    
    # Needs OUTTIME for 24h calculations
    # We can get OUTTIME from ICUSTAYS table in DB or just use 'ADMITTIME'/'DISCHTIME' from somewhere?
    # Phase 11 script queried ICUSTAYS from DB. We can re-use that or query DB again.
    # Assuming DB is still available for ICUSTAYS table (dataset/MIMIC_III.db).
    
    import sqlite3
    db_path = os.path.join(base_dir, 'dataset/MIMIC_III.db')
    conn = sqlite3.connect(db_path)
    print("fetching ICU OUTTIME from DB...")
    
    hadm_str = ",".join(map(str, target_hadms))
    q_stays = f"SELECT HADM_ID, OUTTIME FROM ICUSTAYS WHERE HADM_ID IN ({hadm_str})"
    icustays = pd.read_sql_query(q_stays, conn)
    icustays['OUTTIME'] = pd.to_datetime(icustays['OUTTIME'])
    # Keep last outtime per HADM
    icustays = icustays.sort_values('OUTTIME', ascending=False).drop_duplicates('HADM_ID')
    icustays['HADM_ID'] = icustays['HADM_ID'].astype(int)
    
    # Map for fast lookup
    hadm_outtime = icustays.set_index('HADM_ID')['OUTTIME'].to_dict()
    
    # -------------------------------------------------------------
    # 1. MICROBIOLOGY (Specific Infection)
    # -------------------------------------------------------------
    print("Processing MICROBIOLOGYEVENTS.csv...")
    micro_path = os.path.join(extra_dir, 'MICROBIOLOGYEVENTS.csv')
    
    # Cols: HADM_ID, CHARTTIME, ORG_NAME, SPEC_TYPE_DESC
    # Load all (70MB is fine)
    micro = pd.read_csv(micro_path, usecols=['HADM_ID', 'CHARTTIME', 'ORG_NAME', 'SPEC_TYPE_DESC'])
    micro = micro[micro['HADM_ID'].notna()]
    micro['HADM_ID'] = micro['HADM_ID'].astype(int)
    micro = micro[micro['HADM_ID'].isin(target_hadms)]
    
    # Filter Positive
    micro = micro.dropna(subset=['ORG_NAME'])
    
    # Count total positives
    micro_count = micro.groupby('HADM_ID').size().reset_index(name='MICRO_TOTAL_POS')
    
    # Count in last 48h?
    micro['CHARTTIME'] = pd.to_datetime(micro['CHARTTIME'])
    
    # Vectorized check
    # Map outtime
    micro['OUTTIME'] = micro['HADM_ID'].map(hadm_outtime)
    micro = micro.dropna(subset=['OUTTIME']) # Logic check
    
    micro['diff_hours'] = (micro['OUTTIME'] - micro['CHARTTIME']).dt.total_seconds() / 3600
    mask_48h = (micro['diff_hours'] >= 0) & (micro['diff_hours'] <= 48)
    
    micro_48h_grp = micro[mask_48h].groupby('HADM_ID').size().reset_index(name='MICRO_POS_48H')
    
    print(f"  Found {len(micro)} positive cultures for cohort.")
    
    # -------------------------------------------------------------
    # 2. TRANSFERS (Ward Stay)
    # -------------------------------------------------------------
    print("Processing TRANSFERS.csv...")
    trans_path = os.path.join(extra_dir, 'TRANSFERS.csv')
    # Cols: HADM_ID, EVENTTYPE, PREV_CAREUNIT, CURR_CAREUNIT, INTIME, OUTTIME
    trans = pd.read_csv(trans_path, usecols=['HADM_ID', 'EVENTTYPE', 'CURR_CAREUNIT', 'INTIME', 'OUTTIME'])
    trans = trans[trans['HADM_ID'].notna()]
    trans['HADM_ID'] = trans['HADM_ID'].astype(int)
    trans = trans[trans['HADM_ID'].isin(target_hadms)]
    
    trans['INTIME'] = pd.to_datetime(trans['INTIME'])
    trans['OUTTIME'] = pd.to_datetime(trans['OUTTIME'])
    
    # Logic: Look at events AFTER the ICU Outtime.
    # But wait, ICU_OUTTIME comes from ICUSTAYS which comes from TRANSFERS aggregated.
    # Simple logic: Calculate total time spent in 'Unspecified' or 'Ward' interactions?
    # Better: Last event before 'discharge'.
    # If CURR_CAREUNIT is NaN or 'Discharge Lounge' etc.
    
    # Let's calculate: Total Length of Stay (Hospital) - ICU Length of Stay.
    # This approximates Ward time.
    # We have ICU_LOS (from ICUSTAYS, maybe not extracted yet, but Phase 4 features has LOS).
    # Let's do a direct calculation from TRANSFERS.
    # Sum durations where CURR_CAREUNIT is NOT an ICU.
    
    icu_units = ['MICU', 'SICU', 'CCU', 'CSRU', 'TSICU']
    
    # Helper to detect ICU
    def is_icu(unit):
        if pd.isna(unit): return False
        return any(u in str(unit) for u in icu_units)
    
    trans['IS_ICU'] = trans['CURR_CAREUNIT'].apply(is_icu)
    trans['DURATION_HRS'] = (trans['OUTTIME'] - trans['INTIME']).dt.total_seconds() / 3600
    
    # Sum Non-ICU Duration
    ward_time = trans[~trans['IS_ICU']].groupby('HADM_ID')['DURATION_HRS'].sum().reset_index(name='WARD_LOS_HRS')
    
    # Also Count of Transfers (Instability?)
    transfer_count = trans.groupby('HADM_ID').size().reset_index(name='TRANSFER_COUNT')
    
    print("  Transfers processed.")
    
    # -------------------------------------------------------------
    # 3. INPUTEVENTS (Fluid Input) - The Giant
    # -------------------------------------------------------------
    print("Processing INPUTEVENTS (CV & MV)...")
    
    input_cols = ['HADM_ID', 'CHARTTIME', 'AMOUNT']
    
    input_feats = {} # HADM_ID -> Total Input
    
    for suffix in ['CV', 'MV']:
        path = os.path.join(extra_dir, f'INPUTEVENTS_{suffix}.csv')
        if not os.path.exists(path):
            print(f"  Skipping {path} (not found)")
            continue
            
        print(f"  Reading {path} (Chunks)...")
        chunk_size = 1000000
        # Check cols in CSV first line? Assume logic holds.
        # MV has 'STARTTIME'/'ENDTIME', CV has 'CHARTTIME'.
        # Both have HADM_ID, AMOUNT.
        # Handling MV dates: usually spread over interval. simple approach: use STARTTIME as timestamp.
        
        # We need to sniff columns to pick time col
        header = pd.read_csv(path, nrows=0).columns.tolist()
        time_col = 'STARTTIME' if 'STARTTIME' in header else 'CHARTTIME'
        use_cols = ['HADM_ID', 'AMOUNT', time_col]
        
        for chunk in pd.read_csv(path, usecols=use_cols, chunksize=chunk_size):
            # Filter HADM
            chunk = chunk[chunk['HADM_ID'].isin(target_hadms)].copy()
            if chunk.empty: continue
            
            chunk['HADM_ID'] = chunk['HADM_ID'].astype(int)
            chunk[time_col] = pd.to_datetime(chunk[time_col])
            chunk['AMOUNT'] = pd.to_numeric(chunk['AMOUNT'], errors='coerce').fillna(0)
            
            # Map Outtime
            chunk['OUTTIME'] = chunk['HADM_ID'].map(hadm_outtime)
            chunk = chunk.dropna(subset=['OUTTIME'])
            
            # 24h Filter
            chunk['diff'] = (chunk['OUTTIME'] - chunk[time_col]).dt.total_seconds() / 3600
            mask_24 = (chunk['diff'] >= 0) & (chunk['diff'] <= 24)
            
            # Aggregate
            aggs = chunk[mask_24].groupby('HADM_ID')['AMOUNT'].sum()
            
            for hid, val in aggs.items():
                input_feats[hid] = input_feats.get(hid, 0) + val
                
    input_df = pd.DataFrame(list(input_feats.items()), columns=['HADM_ID', 'FLUID_INPUT_24H'])
    print(f"  Inputs processed for {len(input_df)} admissions.")
    
    # -------------------------------------------------------------
    # 4. MERGE
    # -------------------------------------------------------------
    print("Merging all features...")
    features = pd.DataFrame({'HADM_ID': list(target_hadms)})
    features = features.merge(micro_count, on='HADM_ID', how='left')
    features = features.merge(micro_48h_grp, on='HADM_ID', how='left')
    features = features.merge(ward_time, on='HADM_ID', how='left')
    features = features.merge(transfer_count, on='HADM_ID', how='left')
    features = features.merge(input_df, on='HADM_ID', how='left')
    
    features = features.fillna(0)
    
    print(f"Saving to {output_path}...")
    features.to_csv(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
