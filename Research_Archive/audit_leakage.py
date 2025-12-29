import pandas as pd
import numpy as np
import sqlite3
import os

def main():
    base_dir = '/Users/phandanglinh/Desktop/VRES'
    db_path = os.path.join(base_dir, 'dataset/MIMIC_III.db')
    
    conn = sqlite3.connect(db_path)
    
    # New implementation
    # 1. Load ICU OUTTIMES from DB
    q_icu = "SELECT HADM_ID, OUTTIME as ICU_OUTTIME FROM ICUSTAYS LIMIT 5"
    icu_df = pd.read_sql_query(q_icu, conn)
    
    target_ids = [int(x) for x in icu_df['HADM_ID'].tolist()]
    print(f"Target IDs from DB: {target_ids}")
    
    # 2. Load TRANSFERS from CSV (filtered)
    t_path = os.path.join(base_dir, 'dataset/additional/TRANSFERS.csv')
    print("Loading TRANSFERS.csv...")
    trans_full = pd.read_csv(t_path)
    
    print("CSV Columns:", trans_full.columns.tolist())
    print("First 5 CSV Rows:\n", trans_full.head())
    
    # Ensure HADM_ID is numeric matching DB
    trans_full = trans_full[pd.to_numeric(trans_full['HADM_ID'], errors='coerce').notna()]
    trans_full['HADM_ID'] = trans_full['HADM_ID'].astype(int)
    
    csv_ids = set(trans_full['HADM_ID'].unique())
    intersection = set(target_ids).intersection(csv_ids)
    print(f"IDs found in CSV: {len(intersection)} / {len(target_ids)}")
    trans_full['INTIME'] = pd.to_datetime(trans_full['INTIME'])
    
    for idx, row in icu_df.iterrows():
        hid = int(row['HADM_ID'])
        out = pd.to_datetime(row['ICU_OUTTIME'])
        
        print(f"\n--- Checking HADM {hid} (ICU Out: {out}) ---")
        
        print(f"Filtering for ID: {hid} (Type: {type(hid)})")
        mask = trans_full['HADM_ID'] == hid
        print(f"Matches found: {mask.sum()}")
        trans = trans_full[mask].sort_values('INTIME')
        print("Transfers found:\n", trans[['INTIME', 'OUTTIME', 'CURR_CAREUNIT']])
        trans['INTIME'] = pd.to_datetime(trans['INTIME'])
        
        # Check for Ward stays STARTING after or equal to ICU_OUTTIME
        # (Allow tiny buffer for simultaneous timestamp)
        future_stays = trans[trans['INTIME'] >= out]
        
        if not future_stays.empty:
            print("  [LEAKAGE CONFIRMED] Found transfers occurring AFTER ICU Discharge:")
            print(future_stays[['INTIME', 'CURR_CAREUNIT']])
            
            # Did our Phase 12 script count this?
            # Phase 12 logic was:
            # ward_time = trans[~trans['IS_ICU']].groupby('HADM_ID')['DURATION_HRS'].sum()
            # It did NOT filter by time < ICU_OUTTIME.
            # So it included these future stays.
        else:
            print("  No future transfers found (or discharged directly).")

if __name__ == "__main__":
    main()
