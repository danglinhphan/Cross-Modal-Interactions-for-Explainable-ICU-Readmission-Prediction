import sqlite3
import pandas as pd
import numpy as np
import os

def main():
    base_dir = '/Users/phandanglinh/Desktop/VRES'
    db_path = os.path.join(base_dir, 'dataset/MIMIC_III.db')
    cohort_path = os.path.join(base_dir, 'cohort/features_phase4_clinical.csv')
    output_path = os.path.join(base_dir, 'cohort/new_pathology_features.csv')
    
    # 1. Load Cohort IDs
    print(f"Loading cohort from {cohort_path}...")
    df_cohort = pd.read_csv(cohort_path)
    # Standardize column names
    df_cohort.columns = [c.upper() if 'id' in c.lower() else c for c in df_cohort.columns]
    
    if 'HADM_ID' not in df_cohort.columns:
        print("Error: HADM_ID not found in cohort file.")
        return
        
    hadm_ids = df_cohort['HADM_ID'].unique().tolist()
    print(f"Found {len(hadm_ids)} unique admissions.")
    
    # 2. Define Features to Extract
    # Dictionary mapping Concept Name -> list of ITEMIDs
    lab_concepts = {
        'CRP': [50889],
        'TroponinI': [51002],
        'TroponinT': [51003],
        'NTproBNP': [50963],
        'D_Dimer': [50915, 51196],
        'Fibrinogen': [51214],
        'ALT': [50861],
        'Bilirubin_Total': [50885, 51464], 
        'pH': [50820, 50831, 51094, 51491], # Include blood/body fluid/urine for coverage
        'pCO2': [50818, 50830],
        'pO2': [50821, 50832],
        'Lactate': [50813, 52442], # Adding Lactate explicitly if not fully covered before, though likely is.
        'Albumin': [50862, 51542]  # Explicit Albumin
    }
    
    # Flatten IDs for query
    all_item_ids = []
    for ids in lab_concepts.values():
        all_item_ids.extend(ids)
        
    # 3. Query Database
    print(f"Querying MIMIC-III database at {db_path}...")
    conn = sqlite3.connect(db_path)
    
    # Chunking the query to avoid "SQL variable limit" if list is too long
    # SQLite limit is often 999. We have ~17k admissions.
    # We will fetch ALL labs for these items and then filter by inner join in pandas or temp table.
    # But fetching ALL labs for huge DB might be slow globally.
    # Better strategy: Filter by ITEMID in SQL, fetch filtering by HADM_ID? 
    # Or create temporary table with HADM_IDs.
    
    # Let's try simple chunking of IN clause for HADM_IDs
    chunk_size = 500
    all_lab_data = []
    
    item_id_str = ",".join(map(str, all_item_ids))
    
    print("Fetching lab events...")
    
    # Create a temp string of HADM_IDs for WHERE clause? No, too long.
    # We loop over chunks of HADM_IDs.
    for i in range(0, len(hadm_ids), chunk_size):
        if i % 2000 == 0:
            print(f"  Processing admission chunk {i}/{len(hadm_ids)}...")
            
        chunk_ids = hadm_ids[i:i+chunk_size]
        id_str = ",".join(map(str, chunk_ids))
        
        query = f"""
        SELECT HADM_ID, ITEMID, VALUENUM, CHARTTIME
        FROM LABEVENTS
        WHERE ITEMID IN ({item_id_str})
          AND HADM_ID IN ({id_str})
          AND VALUENUM IS NOT NULL
        """
        
        chunk_df = pd.read_sql_query(query, conn)
        all_lab_data.append(chunk_df)
        
    conn.close()
    
    if not all_lab_data:
        print("No lab data found for these features.")
        return
        
    df_labs = pd.concat(all_lab_data, ignore_index=True)
    print(f"fetched {len(df_labs)} lab measurements.")
    
    # Map ITEMID back to Concept Name
    item_to_concept = {}
    for concept, ids in lab_concepts.items():
        for iid in ids:
            item_to_concept[iid] = concept

    # DEBUG
    print("Sample Item IDs from DB:", df_labs['ITEMID'].unique()[:5])
    print("Sample Map Keys:", list(item_to_concept.keys())[:5])
    
    # Fix Type Mismatch
    df_labs['ITEMID'] = df_labs['ITEMID'].astype(int)
    # Fix Value Type (force numeric)
    df_labs['VALUENUM'] = pd.to_numeric(df_labs['VALUENUM'], errors='coerce')
    
    df_labs['CONCEPT'] = df_labs['ITEMID'].map(item_to_concept)
    
    print("Mapped Concepts sample:", df_labs['CONCEPT'].unique())
    print("Rows with NaN Concept:", df_labs['CONCEPT'].isna().sum())
    
    # Group by HADM_ID and CONCEPT
    grouped = df_labs.groupby(['HADM_ID', 'CONCEPT'])['VALUENUM']
    
    # Calculating stats
    df_min = grouped.min().unstack(fill_value=np.nan).add_suffix('_Min')
    df_max = grouped.max().unstack(fill_value=np.nan).add_suffix('_Max')
    df_avg = grouped.mean().unstack(fill_value=np.nan).add_suffix('_Avg')
    
    # Last value requires sorting
    df_labs_sorted = df_labs.sort_values(['HADM_ID', 'CHARTTIME'])
    df_last = df_labs_sorted.groupby(['HADM_ID', 'CONCEPT'])['VALUENUM'].last().unstack(fill_value=np.nan).add_suffix('_Last')
    
    # Count (Frequency - to replace "Monitoring Intensity" properly)
    df_count = grouped.count().unstack(fill_value=0).add_suffix('_Count')
    
    # Merge all
    df_features = pd.concat([df_min, df_max, df_avg, df_last, df_count], axis=1)
    
    # Add _Measured binary flag
    for concept in lab_concepts.keys():
        col_name = f"{concept}_Measured"
        # If Count > 0, then Measured = 1
        count_col = f"{concept}_Count"
        if count_col in df_features.columns:
            df_features[col_name] = (df_features[count_col] > 0).astype(int)
        else:
             df_features[col_name] = 0

    # Reset index to make HADM_ID a column
    df_features = df_features.reset_index()
    
    # Fill NaN with 0? 
    # For EBM, 0 is often treated as a value. 
    # For Lab values like pH, 0 is bad.
    # Better to leave NaN if possible? EBM handles clean data.
    # Impute missing values with -1 or 0 and let tree split? 
    # Standard practice: Impute with Normal Mean OR 0 if 0 is not in range.
    # For simplicity and EBM ability: Fill NaN with -1.
    df_features = df_features.fillna(-1)
    
    print(f"Extracted {df_features.shape[1]} new features for {len(df_features)} admissions.")
    
    # 5. Save
    df_features.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
