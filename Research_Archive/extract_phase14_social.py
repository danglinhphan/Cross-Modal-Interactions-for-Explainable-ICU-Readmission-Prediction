import pandas as pd
import numpy as np
import sqlite3
import os

def main():
    base_dir = '/Users/phandanglinh/Desktop/VRES'
    db_path = os.path.join(base_dir, 'dataset/MIMIC_III.db')
    output_path = os.path.join(base_dir, 'cohort/features_phase14_social.csv')
    cohort_path = os.path.join(base_dir, 'cohort/new_cohort_icu_readmission_labels.csv')
    
    print("Loading cohort...")
    cohort = pd.read_csv(cohort_path)
    if 'HADM_ID' in cohort.columns:
        cohort['HADM_ID'] = cohort['HADM_ID'].astype(int)
    
    target_hadms = cohort['HADM_ID'].unique()
    hadm_str = ",".join(map(str, target_hadms))
    
    print(f"Connecting to {db_path}...")
    conn = sqlite3.connect(db_path)
    
    # Query ADMISSIONS table
    # Columns: INSURANCE, LANGUAGE, RELIGION, MARITAL_STATUS, ETHNICITY, DISCHARGE_LOCATION
    print("Querying ADMISSIONS table...")
    q = f"""
    SELECT HADM_ID, INSURANCE, MARITAL_STATUS, ETHNICITY, DISCHARGE_LOCATION
    FROM ADMISSIONS
    WHERE HADM_ID IN ({hadm_str})
    """
    
    social = pd.read_sql_query(q, conn)
    social['HADM_ID'] = social['HADM_ID'].astype(int)
    
    print(f"Fetched {len(social)} records.")
    
    # Preprocessing / Cleaning
    # 1. DISCHARGE_LOCATION
    # Group rare categories?
    # HOME, SNF, REHAB, HOME HEALTH CARE, SHORT TERM HOSPITAL, etc.
    # Leave as raw categorical? EBM handles it.
    
    # 2. MARITAL_STATUS
    # MARRIED, SINGLE, WIDOWED, DIVORCED, UNKNOWN
    
    # 3. ETHNICITY
    # High cardinality. Group into WHITE, BLACK, ASIAN, HISPANIC, OTHER.
    def clean_ethnicity(x):
        x = str(x).upper()
        if 'WHITE' in x: return 'WHITE'
        if 'BLACK' in x: return 'BLACK'
        if 'ASIAN' in x: return 'ASIAN'
        if 'HISPANIC' in x or 'LATINO' in x: return 'HISPANIC'
        return 'OTHER'
    
    social['ETHNICITY_GROUP'] = social['ETHNICITY'].apply(clean_ethnicity)
    
    # 4. INSURANCE
    # Medicare, Private, Medicaid, Government, Self Pay
    
    # Select final columns
    final_df = social[['HADM_ID', 'INSURANCE', 'MARITAL_STATUS', 'ETHNICITY_GROUP', 'DISCHARGE_LOCATION']]
    
    # Fill NA
    final_df = final_df.fillna('UNKNOWN')
    
    # Save
    print(f"Saving to {output_path}...")
    final_df.to_csv(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
