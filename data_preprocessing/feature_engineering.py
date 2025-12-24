import pandas as pd
import sqlite3
import numpy as np
import os
import json
import argparse
from pathlib import Path

# Configuration
DB_PATH = '/Users/phandanglinh/Desktop/VRES/dataset/MIMIC_III.db'
COHORT_PATH = '/Users/phandanglinh/Desktop/VRES/cohort/filtered_cohort.csv'
OUTPUT_PATH = '/Users/phandanglinh/Desktop/VRES/cohort/features.csv'

def get_db_connection():
    return sqlite3.connect(DB_PATH)


def apply_locf_imputation(df, id_col, time_col, value_col, resample_freq='1h'):
    """
    Apply Last Observation Carried Forward (LOCF) imputation for time series data.
    
    This function:
    1. Resamples time series to regular intervals (default: hourly)
    2. Applies forward fill (LOCF) to fill gaps in the time series
    3. Returns the imputed dataframe
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing time series data
    id_col : str
        Column name for patient/admission ID (e.g., 'HADM_ID', 'ICUSTAY_ID')
    time_col : str
        Column name for timestamp (e.g., 'CHARTTIME')
    value_col : str
        Column name for the measurement value (e.g., 'VALUENUM')
    resample_freq : str
        Frequency for resampling (default: '1H' for hourly)
    
    Returns:
    --------
    pd.DataFrame with LOCF-imputed values
    """
    if df.empty:
        return df
    
    # Ensure proper types
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    
    # Remove rows with invalid timestamps or values
    df = df.dropna(subset=[time_col, value_col, id_col])
    
    if df.empty:
        return df
    
    imputed_dfs = []
    
    for patient_id, group in df.groupby(id_col):
        if len(group) == 0:
            continue
            
        # Sort by time
        group = group.sort_values(time_col)
        
        # Set time as index for resampling
        group = group.set_index(time_col)
        
        # Get the time range for this patient
        start_time = group.index.min()
        end_time = group.index.max()
        
        # If only one measurement or very short time range, keep original
        if len(group) == 1 or (end_time - start_time).total_seconds() < 3600:
            group = group.reset_index()
            imputed_dfs.append(group)
            continue
        
        # Resample to regular intervals and forward fill (LOCF)
        # Use mean if multiple values in same hour
        resampled = group[[value_col]].resample(resample_freq).mean()
        
        # Apply forward fill (LOCF)
        resampled = resampled.ffill()
        
        # Also backward fill the first few values if they start with NaN
        # (this handles cases where first measurement isn't at the start)
        resampled = resampled.bfill()
        
        # Reset index and add patient ID back
        resampled = resampled.reset_index()
        resampled[id_col] = patient_id
        resampled = resampled.rename(columns={'index': time_col})
        
        imputed_dfs.append(resampled)
    
    if not imputed_dfs:
        return df
    
    return pd.concat(imputed_dfs, ignore_index=True)


def compute_stats_with_locf(df, id_col, time_col, value_col, name, apply_locf=True):
    """
    Compute statistics for a variable, optionally applying LOCF imputation first.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw time series data
    id_col : str
        Column name for patient/admission ID
    time_col : str
        Column name for timestamp
    value_col : str
        Column name for measurement value
    name : str
        Variable name (e.g., 'BUN', 'HeartRate')
    apply_locf : bool
        Whether to apply LOCF imputation before computing stats
    
    Returns:
    --------
    tuple: (stats_df, last_vals_df, slope_df)
    """
    if df.empty:
        return None, None, None
    
    # Apply LOCF if requested
    if apply_locf:
        df_imputed = apply_locf_imputation(df, id_col, time_col, value_col)
    else:
        df_imputed = df.copy()
    
    if df_imputed.empty:
        return None, None, None
    
    # Compute aggregate statistics
    stats = df_imputed.groupby(id_col)[value_col].agg(['min', 'max', 'mean', 'median', 'std', 'count']).reset_index()
    stats.columns = [id_col, f'{name}_Min', f'{name}_Max', f'{name}_Avg', f'{name}_Median', f'{name}_Std', f'{name}_Count']
    
    # Get last value (from original data, not resampled - to get actual last measurement)
    df_orig = df.copy()
    df_orig[time_col] = pd.to_datetime(df_orig[time_col], errors='coerce')
    df_orig = df_orig.dropna(subset=[time_col])
    df_orig = df_orig.sort_values([id_col, time_col])
    last_vals = df_orig.groupby(id_col).last().reset_index()[[id_col, value_col, time_col]]
    last_vals.columns = [id_col, f'{name}_Last', f'{name}_Last_ChartTime']
    
    # Compute slope from imputed data (more robust with LOCF)
    slope_rows = []
    df_imputed[time_col] = pd.to_datetime(df_imputed[time_col], errors='coerce')
    df_imputed['__ts'] = df_imputed[time_col].astype('int64') // 10**9
    
    for patient_id, g in df_imputed.groupby(id_col):
        if len(g) < 2:
            slope = np.nan
        else:
            x = g['__ts'].values.astype(np.float64)
            y = g[value_col].values.astype(np.float64)
            # Remove NaN values
            mask = ~(np.isnan(x) | np.isnan(y))
            x, y = x[mask], y[mask]
            if len(x) < 2:
                slope = np.nan
            else:
                # Shift x for numerical stability
                x = x - x.mean()
                try:
                    slope = np.polyfit(x, y, 1)[0]
                except Exception:
                    slope = np.nan
        slope_rows.append({id_col: patient_id, f'{name}_Slope': slope})
    
    slope_df = pd.DataFrame(slope_rows)
    
    return stats, last_vals, slope_df

def load_cohort():
    return pd.read_csv(COHORT_PATH)

def get_demographics(conn, cohort_df):
    print("Extracting Demographics...")
    
    # Get Gender and DOB from PATIENTS
    patients_query = "SELECT SUBJECT_ID, GENDER, DOB FROM PATIENTS"
    patients_df = pd.read_sql_query(patients_query, conn)
    patients_df['SUBJECT_ID'] = pd.to_numeric(patients_df['SUBJECT_ID'], errors='coerce')
    
    # Get Admission details
    adm_query = "SELECT SUBJECT_ID, HADM_ID, ADMITTIME, DISCHTIME, DISCHARGE_LOCATION FROM ADMISSIONS"
    adm_df = pd.read_sql_query(adm_query, conn)
    adm_df['SUBJECT_ID'] = pd.to_numeric(adm_df['SUBJECT_ID'], errors='coerce')
    adm_df['HADM_ID'] = pd.to_numeric(adm_df['HADM_ID'], errors='coerce')
    
    # Get ICU details for LOS
    icu_query = "SELECT ICUSTAY_ID, INTIME, OUTTIME FROM ICUSTAYS"
    icu_df = pd.read_sql_query(icu_query, conn)
    icu_df['ICUSTAY_ID'] = pd.to_numeric(icu_df['ICUSTAY_ID'], errors='coerce')
    
    # Merge
    df = cohort_df.merge(patients_df, on='SUBJECT_ID', how='left')
    df = df.merge(adm_df, on=['SUBJECT_ID', 'HADM_ID'], how='left')
    
    # 1. Gender
    df['Gender'] = df['GENDER'].apply(lambda x: 1 if x == 'M' else 0)
    
    # 2. Age (Already in cohort)
    
    # 3. LOS Hospital
    df['ADMITTIME'] = pd.to_datetime(df['ADMITTIME'])
    df['DISCHTIME'] = pd.to_datetime(df['DISCHTIME'])
    df['LOS_Hospital'] = (df['DISCHTIME'] - df['ADMITTIME']).dt.total_seconds() / (24 * 3600)
    
    # 4. LOS ICU
    if 'ICU_HOURS' in df.columns:
        df['LOS_ICU'] = df['ICU_HOURS'] / 24.0
    else:
        df['INTIME'] = pd.to_datetime(df['INTIME'])
        df['OUTTIME'] = pd.to_datetime(df['OUTTIME'])
        df['LOS_ICU'] = (df['OUTTIME'] - df['INTIME']).dt.total_seconds() / (24 * 3600)

    # 5. Number of Prior Readmits
    adm_df['ADMITTIME'] = pd.to_datetime(adm_df['ADMITTIME'])
    adm_df = adm_df.sort_values(['SUBJECT_ID', 'ADMITTIME'])
    adm_df['seq'] = adm_df.groupby('SUBJECT_ID').cumcount()
    
    df = df.merge(adm_df[['HADM_ID', 'seq']], on='HADM_ID', how='left')
    df.rename(columns={'seq': 'Number_of_Prior_Readmits'}, inplace=True)
    
    # 6. Discharge Disposition
    df['Discharge_Disposition'] = df['DISCHARGE_LOCATION']
    
    return df[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'AGE', 'Gender', 'LOS_Hospital', 'LOS_ICU', 'Number_of_Prior_Readmits', 'Discharge_Disposition']]

def get_lab_events(conn, cohort_df):
    """
    Extract lab events with LOCF (Last Observation Carried Forward) imputation.
    
    Imputation strategy:
    1. Apply LOCF within each patient's time series to fill gaps
    2. Use population median for patients with NO measurements at all
    """
    print("Extracting Lab Events with LOCF imputation...")
    lab_items = {
        'BUN': [51006],
        'Creatinine': [50912],
        'Glucose': [50931, 50809],
        'HMG': [51222, 50811],
        'WBC': [51301, 51300],
        'Platelet': [51265],
        'Lactate': [50813],
        'Albumin': [50862],
        'Anion_Gap': [50868],
        'PTT': [51275]
    }
    
    all_ids = []
    for ids in lab_items.values():
        all_ids.extend(ids)
    
    ids_str = ','.join(map(str, all_ids))
    hadm_ids = list(cohort_df['HADM_ID'].unique())
    
    def chunker(seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))

    dfs = []
    for chunk in chunker(hadm_ids, 5000):
        chunk_str = str(tuple(map(int, chunk))) if len(chunk) > 1 else f"({int(chunk[0])})"
        query = f"""
        SELECT HADM_ID, ITEMID, VALUENUM, CHARTTIME
        FROM LABEVENTS
        WHERE ITEMID IN ({ids_str})
        AND HADM_ID IN {chunk_str}
        AND VALUENUM IS NOT NULL
        """
        try:
            chunk_df = pd.read_sql_query(query, conn)
            chunk_df['HADM_ID'] = pd.to_numeric(chunk_df['HADM_ID'], errors='coerce')
            chunk_df['ITEMID'] = pd.to_numeric(chunk_df['ITEMID'], errors='coerce')
            dfs.append(chunk_df)
        except Exception as e:
            print(f"Error reading lab chunk: {e}")
            
    if not dfs:
        print("  WARNING: No lab data found!")
        return pd.DataFrame({'HADM_ID': cohort_df['HADM_ID'].unique()})
        
    df = pd.concat(dfs, ignore_index=True)
    print(f"  Total lab rows retrieved: {len(df)}")
    df['VALUENUM'] = pd.to_numeric(df['VALUENUM'], errors='coerce')
    df['CHARTTIME'] = pd.to_datetime(df['CHARTTIME'], errors='coerce')
    df['ITEMID'] = pd.to_numeric(df['ITEMID'], errors='coerce')
    
    results = []
    # pull a discharge mapping for last-n-hour counts
    time_col = 'DISCHTIME' if 'DISCHTIME' in cohort_df.columns else 'OUTTIME'
    cohort_times = cohort_df[['HADM_ID', time_col]].drop_duplicates().rename(columns={time_col: 'DISCHTIME'})
    cohort_times['DISCHTIME'] = pd.to_datetime(cohort_times['DISCHTIME'], errors='coerce')

    for name, ids in lab_items.items():
        sub_df = df[df['ITEMID'].isin(ids)].copy()
        if sub_df.empty:
            print(f"  WARNING: No data for {name}")
            continue
        
        # Apply LOCF imputation and compute stats
        print(f"  {name}: Applying LOCF imputation...")
        stats, last_vals, slope_df = compute_stats_with_locf(
            sub_df, 'HADM_ID', 'CHARTTIME', 'VALUENUM', name, apply_locf=True
        )
        
        if stats is not None:
            print(f"  {name}: {len(stats)} admissions (after LOCF)")
            results.append(stats)
        if last_vals is not None:
            results.append(last_vals)
        if slope_df is not None:
            results.append(slope_df)

        # --- counts within last 24/48 hours relative to DISCHTIME (from original data)
        try:
            pre = sub_df.copy()
            pre = pre.sort_values(['HADM_ID', 'CHARTTIME'])
            merged = pre.merge(cohort_times, on='HADM_ID', how='left')
            merged['hours_before_discharge'] = (merged['DISCHTIME'] - merged['CHARTTIME']).dt.total_seconds() / 3600.0
            cnt24 = merged[merged['hours_before_discharge'] <= 24].groupby('HADM_ID').size().reset_index(name=f'{name}_Count_24h')
            cnt48 = merged[merged['hours_before_discharge'] <= 48].groupby('HADM_ID').size().reset_index(name=f'{name}_Count_48h')
        except Exception:
            cnt24 = pd.DataFrame({'HADM_ID': cohort_df['HADM_ID'].unique(), f'{name}_Count_24h': 0})
            cnt48 = pd.DataFrame({'HADM_ID': cohort_df['HADM_ID'].unique(), f'{name}_Count_48h': 0})
        results.append(cnt24)
        results.append(cnt48)
        
    final_df = pd.DataFrame({'HADM_ID': cohort_df['HADM_ID'].unique()})
    for r in results:
        if r is not None and not r.empty:
            final_df = final_df.merge(r, on='HADM_ID', how='left')

    # Population median imputation for admissions with NO measurements
    # This is the second step: for patients who have absolutely no data for a variable
    print("  Applying population median imputation for missing patients...")
    lab_roots = list(lab_items.keys())
    for name in lab_roots:
        count_col = f"{name}_Count"
        stats_cols = [f"{name}_Min", f"{name}_Max", f"{name}_Avg", f"{name}_Median", f"{name}_Std", f"{name}_Last", f"{name}_Slope"]
        
        # Calculate population medians from patients who HAVE data
        medians = {}
        for c in stats_cols:
            if c in final_df.columns:
                med = final_df[c].median(skipna=True)
                medians[c] = med

        # Identify patients with NO measurements (Count == 0 or NaN)
        if count_col in final_df.columns:
            no_data_mask = final_df[count_col].isna() | (final_df[count_col] == 0)
        else:
            last_col = f"{name}_Last"
            no_data_mask = final_df[last_col].isna() if last_col in final_df.columns else pd.Series(False, index=final_df.index)

        # Fill with population median only for patients with NO data
        for c, med in medians.items():
            if pd.isna(med):
                continue
            final_df.loc[no_data_mask & final_df[c].isna(), c] = med
        
        # Log imputation stats
        n_imputed = no_data_mask.sum()
        if n_imputed > 0:
            print(f"    {name}: Imputed {n_imputed} patients with population median")

    return final_df


def get_chart_events(conn, cohort_df):
    """
    Extract vital signs (chart events) with LOCF (Last Observation Carried Forward) imputation.
    
    Imputation strategy:
    1. Apply LOCF within each patient's time series to fill gaps
    2. Use population median for patients with NO measurements at all
    """
    print("Extracting Chart Events (Vitals) with LOCF imputation...")
    chart_items = {
        'GCS': [198, 220739, 223900, 223901],
        'SBP': [51, 442, 455, 6701, 220179, 220050],
        'DBP': [8368, 8440, 8441, 8555, 220180, 220051],
        'HeartRate': [211, 220045],
        'RespRate': [618, 615, 220210, 224690],
        'SpO2': [646, 220277],
    }
    
    all_ids = []
    for ids in chart_items.values():
        all_ids.extend(ids)
        
    ids_str = ','.join(map(str, all_ids))
    icustay_ids = list(cohort_df['ICUSTAY_ID'].unique())
    
    def chunker(seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))
    
    dfs = []
    for chunk in chunker(icustay_ids, 5000):
        chunk_str = str(tuple(map(int, chunk))) if len(chunk) > 1 else f"({int(chunk[0])})"
        query = f"""
        SELECT ICUSTAY_ID, ITEMID, VALUENUM, CHARTTIME
        FROM CHARTEVENTS
        WHERE ITEMID IN ({ids_str})
        AND ICUSTAY_ID IN {chunk_str}
        AND VALUENUM IS NOT NULL
        """
        try:
            chunk_df = pd.read_sql_query(query, conn)
            chunk_df['ICUSTAY_ID'] = pd.to_numeric(chunk_df['ICUSTAY_ID'], errors='coerce')
            chunk_df['ITEMID'] = pd.to_numeric(chunk_df['ITEMID'], errors='coerce')
            dfs.append(chunk_df)
        except Exception as e:
            print(f"Error reading chart chunk: {e}")
            
    if not dfs:
        print("  WARNING: No chart data found!")
        return pd.DataFrame({'ICUSTAY_ID': cohort_df['ICUSTAY_ID'].unique()})
        
    df = pd.concat(dfs, ignore_index=True)
    print(f"  Total chart rows retrieved: {len(df)}")
    df['VALUENUM'] = pd.to_numeric(df['VALUENUM'], errors='coerce')
    if 'CHARTTIME' in df.columns:
        df['CHARTTIME'] = pd.to_datetime(df['CHARTTIME'], errors='coerce')
    df['ITEMID'] = pd.to_numeric(df['ITEMID'], errors='coerce')
    
    results = []
    # attach ICU OUTTIME map to compute hours before ICU discharge
    cohort_times = cohort_df[['ICUSTAY_ID', 'OUTTIME']].drop_duplicates()
    cohort_times['OUTTIME'] = pd.to_datetime(cohort_times['OUTTIME'], errors='coerce')

    for name, ids in chart_items.items():
        sub_df = df[df['ITEMID'].isin(ids)].copy()
        if sub_df.empty:
            print(f"  WARNING: No data for {name}")
            continue
        
        # Apply LOCF imputation and compute stats
        print(f"  {name}: Applying LOCF imputation...")
        stats, last_vals, slope_df = compute_stats_with_locf(
            sub_df, 'ICUSTAY_ID', 'CHARTTIME', 'VALUENUM', name, apply_locf=True
        )
        
        # For GCS, we want specific columns
        if name == 'GCS' and stats is not None:
            # Keep only Median, Min, Max for GCS
            gcs_cols = ['ICUSTAY_ID', f'{name}_Median', f'{name}_Min', f'{name}_Max']
            available_cols = [c for c in gcs_cols if c in stats.columns]
            if available_cols:
                stats = stats[available_cols]
        
        if stats is not None:
            print(f"  {name}: {len(stats)} ICU stays (after LOCF)")
            results.append(stats)
        if last_vals is not None:
            results.append(last_vals)
        if slope_df is not None:
            results.append(slope_df)

        # 24h/48h ICU counts (from original data)
        try:
            sub = sub_df.copy()
            sub = sub.sort_values(['ICUSTAY_ID', 'CHARTTIME'])
            merged = sub.merge(cohort_times, on='ICUSTAY_ID', how='left')
            merged['hours_before_icu_out'] = (merged['OUTTIME'] - merged['CHARTTIME']).dt.total_seconds() / 3600.0
            cnt24 = merged[merged['hours_before_icu_out'] <= 24].groupby('ICUSTAY_ID').size().reset_index(name=f'{name}_Count_24h')
            cnt48 = merged[merged['hours_before_icu_out'] <= 48].groupby('ICUSTAY_ID').size().reset_index(name=f'{name}_Count_48h')
        except Exception:
            cnt24 = pd.DataFrame({'ICUSTAY_ID': cohort_df['ICUSTAY_ID'].unique(), f'{name}_Count_24h': 0})
            cnt48 = pd.DataFrame({'ICUSTAY_ID': cohort_df['ICUSTAY_ID'].unique(), f'{name}_Count_48h': 0})
        results.append(cnt24)
        results.append(cnt48)
        
    final_df = pd.DataFrame({'ICUSTAY_ID': cohort_df['ICUSTAY_ID'].unique()})
    for r in results:
        if r is not None and not r.empty:
            final_df = final_df.merge(r, on='ICUSTAY_ID', how='left')

    # Population median imputation for vitals if NO measurements for the ICU stay
    print("  Applying population median imputation for missing patients...")
    vital_roots = list(chart_items.keys())
    for name in vital_roots:
        count_col = f"{name}_Count"
        # for GCS we use Median instead of Avg
        if name == 'GCS':
            stats_cols = [f"{name}_Median", f"{name}_Min", f"{name}_Max", f"{name}_Last", f"{name}_Slope"]
        else:
            stats_cols = [f"{name}_Min", f"{name}_Max", f"{name}_Avg", f"{name}_Median", f"{name}_Std", f"{name}_Last", f"{name}_Slope"]
        
        # Calculate population medians from patients who HAVE data
        medians = {}
        for c in stats_cols:
            if c in final_df.columns:
                med = final_df[c].median(skipna=True)
                medians[c] = med

        # Identify patients with NO measurements
        if count_col in final_df.columns:
            no_data_mask = final_df[count_col].isna() | (final_df[count_col] == 0)
        else:
            last_col = f"{name}_Last"
            no_data_mask = final_df[last_col].isna() if last_col in final_df.columns else pd.Series(False, index=final_df.index)

        # Fill with population median only for patients with NO data
        for c, med in medians.items():
            if pd.isna(med):
                continue
            final_df.loc[no_data_mask & final_df[c].isna(), c] = med
        
        # Log imputation stats
        n_imputed = no_data_mask.sum()
        if n_imputed > 0:
            print(f"    {name}: Imputed {n_imputed} patients with population median")
        
    return final_df

def get_urine_output(conn, cohort_df):
    print("Extracting Urine Output...")
    urine_ids = [
        40055, 43175, 40069, 40094, 40715, 40473, 40085, 40056, 40405, 40428, 
        40086, 40096, 40651, 226559, 226560, 226561, 226584, 226563, 226564, 
        226565, 226567, 226557, 226558, 227488, 227489
    ]
    ids_str = ','.join(map(str, urine_ids))
    icustay_ids = list(cohort_df['ICUSTAY_ID'].unique())
    
    def chunker(seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))
        
    dfs = []
    for chunk in chunker(icustay_ids, 5000):
        chunk_str = str(tuple(map(int, chunk))) if len(chunk) > 1 else f"({int(chunk[0])})"
        query = f"""
        SELECT ICUSTAY_ID, VALUE
        FROM OUTPUTEVENTS
        WHERE ITEMID IN ({ids_str})
        AND ICUSTAY_ID IN {chunk_str}
        """
        try:
            chunk_df = pd.read_sql_query(query, conn)
            chunk_df['ICUSTAY_ID'] = pd.to_numeric(chunk_df['ICUSTAY_ID'], errors='coerce')
            dfs.append(chunk_df)
        except Exception as e:
            print(f"Error reading urine chunk: {e}")
            
    if not dfs:
        return pd.DataFrame({'ICUSTAY_ID': cohort_df['ICUSTAY_ID'].unique()})
        
    df = pd.concat(dfs, ignore_index=True)
    df['VALUE'] = pd.to_numeric(df['VALUE'], errors='coerce')
    
    stats = df.groupby('ICUSTAY_ID')['VALUE'].agg(['median', 'max', 'mean']).reset_index()
    stats.columns = ['ICUSTAY_ID', 'UrineOutput_Median', 'UrineOutput_Max', 'UrineOutput_Avg']
    
    return stats

def get_pao2_fio2_ratio(conn, cohort_df):
    print("Extracting PaO2/FiO2 Ratio...")
    # PaO2 (Partial Pressure of Oxygen in arterial blood)
    pao2_ids = [50821, 490]
    # FiO2 (Fraction of Inspired Oxygen)
    fio2_ids = [50816, 223835, 3420, 190]
    
    icustay_ids = list(cohort_df['ICUSTAY_ID'].unique())
    
    def chunker(seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))
    
    # Get PaO2 from LABEVENTS
    pao2_dfs = []
    pao2_ids_str = ','.join(map(str, pao2_ids))
    hadm_ids = list(cohort_df['HADM_ID'].unique())
    
    for chunk in chunker(hadm_ids, 5000):
        chunk_str = str(tuple(map(int, chunk))) if len(chunk) > 1 else f"({int(chunk[0])})"
        query = f"""
        SELECT HADM_ID, CHARTTIME, VALUENUM as PaO2
        FROM LABEVENTS
        WHERE ITEMID IN ({pao2_ids_str})
        AND HADM_ID IN {chunk_str}
        AND VALUENUM IS NOT NULL
        """
        try:
            chunk_df = pd.read_sql_query(query, conn)
            chunk_df['HADM_ID'] = pd.to_numeric(chunk_df['HADM_ID'], errors='coerce')
            pao2_dfs.append(chunk_df)
        except Exception as e:
            print(f"Error reading PaO2 chunk: {e}")
    
    # Get FiO2 from CHARTEVENTS
    fio2_dfs = []
    fio2_ids_str = ','.join(map(str, fio2_ids))
    
    for chunk in chunker(icustay_ids, 5000):
        chunk_str = str(tuple(map(int, chunk))) if len(chunk) > 1 else f"({int(chunk[0])})"
        query = f"""
        SELECT ICUSTAY_ID, CHARTTIME, VALUENUM as FiO2
        FROM CHARTEVENTS
        WHERE ITEMID IN ({fio2_ids_str})
        AND ICUSTAY_ID IN {chunk_str}
        AND VALUENUM IS NOT NULL
        """
        try:
            chunk_df = pd.read_sql_query(query, conn)
            chunk_df['ICUSTAY_ID'] = pd.to_numeric(chunk_df['ICUSTAY_ID'], errors='coerce')
            fio2_dfs.append(chunk_df)
        except Exception as e:
            print(f"Error reading FiO2 chunk: {e}")
    
    if not pao2_dfs or not fio2_dfs:
        result_df = pd.DataFrame({'ICUSTAY_ID': cohort_df['ICUSTAY_ID'].unique()})
        result_df['PaO2_FiO2_Ratio_Median'] = np.nan
        return result_df
    
    # Merge cohort with HADM_ID and ICUSTAY_ID mapping
    cohort_map = cohort_df[['HADM_ID', 'ICUSTAY_ID']].drop_duplicates()
    
    pao2_df = pd.concat(pao2_dfs, ignore_index=True)
    fio2_df = pd.concat(fio2_dfs, ignore_index=True)
    
    pao2_df['CHARTTIME'] = pd.to_datetime(pao2_df['CHARTTIME'])
    fio2_df['CHARTTIME'] = pd.to_datetime(fio2_df['CHARTTIME'])
    
    pao2_df['PaO2'] = pd.to_numeric(pao2_df['PaO2'], errors='coerce')
    fio2_df['FiO2'] = pd.to_numeric(fio2_df['FiO2'], errors='coerce')
    
    # Normalize FiO2 to fraction (if given as percentage)
    fio2_df['FiO2'] = fio2_df['FiO2'].apply(lambda x: x/100 if x > 1 else x)
    
    # Merge PaO2 with cohort to get ICUSTAY_ID
    pao2_df = pao2_df.merge(cohort_map, on='HADM_ID', how='inner')
    
    # Merge PaO2 and FiO2 on ICUSTAY_ID and similar time (within 1 hour)
    merged_list = []
    for icustay_id in pao2_df['ICUSTAY_ID'].unique():
        pao2_stay = pao2_df[pao2_df['ICUSTAY_ID'] == icustay_id]
        fio2_stay = fio2_df[fio2_df['ICUSTAY_ID'] == icustay_id]
        
        if fio2_stay.empty:
            continue
            
        for _, pao2_row in pao2_stay.iterrows():
            time_diff = abs(fio2_stay['CHARTTIME'] - pao2_row['CHARTTIME'])
            closest_idx = time_diff.idxmin()
            
            if time_diff[closest_idx].total_seconds() <= 3600:  # Within 1 hour
                fio2_val = fio2_stay.loc[closest_idx, 'FiO2']
                if fio2_val > 0:
                    ratio = pao2_row['PaO2'] / fio2_val
                    merged_list.append({
                        'ICUSTAY_ID': icustay_id,
                        'PaO2_FiO2_Ratio': ratio
                    })
    
    if not merged_list:
        result_df = pd.DataFrame({'ICUSTAY_ID': cohort_df['ICUSTAY_ID'].unique()})
        result_df['PaO2_FiO2_Ratio_Median'] = np.nan
        return result_df
    
    ratio_df = pd.DataFrame(merged_list)
    stats = ratio_df.groupby('ICUSTAY_ID')['PaO2_FiO2_Ratio'].median().reset_index()
    stats.columns = ['ICUSTAY_ID', 'PaO2_FiO2_Ratio_Median']
    
    return stats


def get_nlp_summary_features(emb_npz=None, emb_index=None, n_clusters=8, keywords=None):
    """Return a DataFrame of NLP-derived features (one row per HADM_ID)
    - cluster assignment(s) using embeddings
    - keyword counts normalized by token count
    - note word counts
    """
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans

    if emb_npz is None or emb_index is None:
        return pd.DataFrame({'HADM_ID': []})
    idxdf = pd.read_csv(emb_index, dtype={'HADM_ID': object})
    npz = np.load(emb_npz)
    X = npz['X']
    emb_df = pd.DataFrame(X, columns=[f'note_emb_pca_16_{i}' for i in range(X.shape[1])])
    emb_df['HADM_ID'] = pd.to_numeric(idxdf['HADM_ID'], errors='coerce').astype('Int64').astype('int64')

    # cluster embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cl = kmeans.fit_predict(X)
    emb_df['note_cluster'] = cl

    out = emb_df[['HADM_ID', 'note_cluster'] + [c for c in emb_df.columns if c.startswith('note_emb_pca_16_')]]

    # keyword counts: fallback to simple word counts if raw texts available
    kw_list = keywords if keywords is not None else ['sepsis','infection','pneumonia','intubated','extubated','readmit','rehospitalization','re-admit','dyspnea','respiratory']
    # try to load raw texts if exists
    idx = emb_index.replace('.csv','')
    try:
        # try to find a text CSV in cohort
        txt_path = os.path.join(os.path.dirname(__file__), '..', 'cohort', 'nlp_features.csv')
        txtdf = pd.read_csv(txt_path, dtype={'HADM_ID':object})
        # attempt to find a column with text
        text_col_candidates = [c for c in txtdf.columns if 'text' in c.lower() or 'note' in c.lower()] or [c for c in txtdf.columns if c.lower() not in ('hadm_id','subject_id')]
        text_col = text_col_candidates[0]
        # compute counts
        def count_kw(text):
            if pd.isna(text):
                return 0
            t = str(text).lower()
            wc = len(t.split())
            res = {}
            for kw in kw_list:
                res[f'kw_{kw}'] = t.count(kw) / (wc+1e-9)
            res['note_word_count'] = wc
            res['HADM_ID'] = int(txtdf['HADM_ID']) if 'HADM_ID' in txtdf.columns else None
            return res
        # create df of counts per HADM
        kw_rows = []
        for _, r in txtdf.iterrows():
            h = int(r['HADM_ID']) if 'HADM_ID' in r else None
            if h is None:
                continue
            t = str(r[text_col]) if text_col in r else ''
            wd = len(t.split())
            row = {'HADM_ID': h, 'note_word_count': wd}
            for kw in kw_list:
                row[f'kw_{kw}'] = t.lower().count(kw)
            kw_rows.append(row)
        kw_features = pd.DataFrame(kw_rows).groupby('HADM_ID').agg('sum').reset_index()
        # normalize keyword counts by note_word_count
        for kw in kw_list:
            if 'note_word_count' in kw_features.columns and kw_features['note_word_count'].sum() > 0:
                kw_features[f'kw_{kw}_norm'] = kw_features[f'kw_{kw}'] / (kw_features['note_word_count']+1e-9)
        out = out.merge(kw_features, on='HADM_ID', how='left')
    except Exception as e:
        # cannot find text
        pass

    # fill missing
    out.fillna(0, inplace=True)
    return out

def get_comorbidities(conn, cohort_df):
    print("Extracting Comorbidities...")
    hadm_ids = list(cohort_df['HADM_ID'].unique())
    
    def chunker(seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))
        
    dfs = []
    for chunk in chunker(hadm_ids, 5000):
        chunk_str = str(tuple(map(int, chunk))) if len(chunk) > 1 else f"({int(chunk[0])})"
        query = f"""
        SELECT HADM_ID, ICD9_CODE
        FROM DIAGNOSES_ICD
        WHERE HADM_ID IN {chunk_str}
        """
        try:
            chunk_df = pd.read_sql_query(query, conn)
            chunk_df['HADM_ID'] = pd.to_numeric(chunk_df['HADM_ID'], errors='coerce')
            dfs.append(chunk_df)
        except Exception as e:
            print(f"Error reading diagnoses chunk: {e}")
            
    if not dfs:
        return pd.DataFrame({'HADM_ID': cohort_df['HADM_ID'].unique()})
        
    df = pd.concat(dfs, ignore_index=True)
    df['ICD9_CODE'] = df['ICD9_CODE'].astype(str)
    
    # Charlson Comorbidity Index conditions and their weights
    def calculate_cci(codes):
        score = 0
        codes_set = set(codes)
        
        # 1 point conditions
        # Myocardial infarction: 410.x, 412
        if any(c.startswith('410') or c == '412' for c in codes_set):
            score += 1
        
        # CHF: 428.x
        if any(c.startswith('428') for c in codes_set):
            score += 1
        
        # Peripheral vascular disease: 441.x, 443.9, 785.4, V43.4, procedure codes 38.48, 39.50
        if any(c.startswith('441') or c == '443.9' or c.startswith('7854') or c == 'V43.4' for c in codes_set):
            score += 1
        
        # CVA/TIA: 430-438
        if any(c.startswith(('430', '431', '432', '433', '434', '435', '436', '437', '438')) for c in codes_set):
            score += 1
        
        # Dementia: 290.x
        if any(c.startswith('290') for c in codes_set):
            score += 1
        
        # Chronic pulmonary disease: 490-505, 506.4
        if any(c.startswith(('490', '491', '492', '493', '494', '495', '496', '497', '498', '499', '500', '501', '502', '503', '504', '505')) or c == '506.4' for c in codes_set):
            score += 1
        
        # Connective tissue disease: 710.0, 710.1, 710.4, 714.0-714.2, 714.81, 725.x
        if any(c in ('710.0', '710.1', '710.4', '714.0', '714.1', '714.2', '714.81') or c.startswith('725') for c in codes_set):
            score += 1
        
        # Peptic ulcer disease: 531.x-534.x
        if any(c.startswith(('531', '532', '533', '534')) for c in codes_set):
            score += 1
        
        # Mild liver disease: 571.2, 571.5, 571.6, 571.4-571.49
        mild_liver = any(c in ('571.2', '571.5', '571.6') or c.startswith('5714') for c in codes_set)
        # Moderate to severe liver disease: 572.2-572.8, 456.0-456.21
        severe_liver = any(c.startswith(('5722', '5723', '5724', '5725', '5726', '5727', '5728', '4560', '4561', '4562')) for c in codes_set)
        
        if severe_liver:
            score += 3
        elif mild_liver:
            score += 1
        
        # Diabetes without complications: 250.0-250.3, 250.7
        diabetes_uncomp = any(c.startswith(('2500', '2501', '2502', '2503', '2507')) for c in codes_set)
        # Diabetes with end organ damage: 250.4-250.6
        diabetes_comp = any(c.startswith(('2504', '2505', '2506')) for c in codes_set)
        
        if diabetes_comp:
            score += 2
        elif diabetes_uncomp:
            score += 1
        
        # 2 point conditions
        # Hemiplegia: 342.x, 344.1
        if any(c.startswith('342') or c == '344.1' for c in codes_set):
            score += 2
        
        # Moderate to severe CKD: 582.x, 583.0-583.7, 585.x, 586.x, V42.0, V45.1, V56.x
        if any(c.startswith(('582', '585', '586')) or c.startswith('583') and c[3] in '0123456' or c in ('V42.0', 'V45.1') or c.startswith('V56') for c in codes_set):
            score += 2
        
        # Solid tumor (non-metastatic): 140.x-172.x, 174.x-195.8
        solid_tumor = any(c.startswith(tuple(str(i) for i in range(140, 173))) or 
                         c.startswith(tuple(str(i) for i in range(174, 196))) for c in codes_set)
        # Metastatic solid tumor: 196.x-199.x
        metastatic = any(c.startswith(('196', '197', '198', '199')) for c in codes_set)
        
        if metastatic:
            score += 6
        elif solid_tumor:
            score += 2
        
        # Leukemia: 204.x-208.x
        if any(c.startswith(('204', '205', '206', '207', '208')) for c in codes_set):
            score += 2
        
        # Lymphoma: 200.x-203.x, 238.6
        if any(c.startswith(('200', '201', '202', '203')) or c == '238.6' for c in codes_set):
            score += 2
        
        # AIDS: 042.x-044.x
        if any(c.startswith(('042', '043', '044')) for c in codes_set):
            score += 6
        
        return score
    
    # Elixhauser Comorbidity Index - counts number of comorbidities present
    def calculate_elixhauser(codes):
        count = 0
        codes_set = set(codes)
        
        # 1. CHF: 398.91, 402.01, 402.11, 402.91, 404.01, 404.03, 404.11, 404.13, 404.91, 404.93, 425.x, 428.x
        if any(c in ('398.91', '402.01', '402.11', '402.91', '404.01', '404.03', '404.11', '404.13', '404.91', '404.93') or 
               c.startswith(('425', '428')) for c in codes_set):
            count += 1
        
        # 2. Cardiac arrhythmias: 426.x, 427.x
        if any(c.startswith(('426', '427')) for c in codes_set):
            count += 1
        
        # 3. Valvular disease: 093.2, 394.x-397.x, 424.x, 746.x, V42.2, V43.3
        if any(c == '093.2' or c.startswith(('394', '395', '396', '397', '424', '746')) or 
               c in ('V42.2', 'V43.3') for c in codes_set):
            count += 1
        
        # 4. Pulmonary circulation disorders: 415.x, 416.x, 417.x
        if any(c.startswith(('415', '416', '417')) for c in codes_set):
            count += 1
        
        # 5. Peripheral vascular disorders: 440.x, 441.x, 442.x, 443.x, 447.1, 557.1, 557.9, V43.4
        if any(c.startswith(('440', '441', '442', '443')) or c in ('447.1', '557.1', '557.9', 'V43.4') for c in codes_set):
            count += 1
        
        # 6. Hypertension (uncomplicated): 401.x
        hypertension_uncomp = any(c.startswith('401') for c in codes_set)
        # Hypertension (complicated): 402.x-405.x
        hypertension_comp = any(c.startswith(('402', '403', '404', '405')) for c in codes_set)
        if hypertension_comp:
            count += 1
        elif hypertension_uncomp:
            count += 1
        
        # 7. Paralysis: 342.x, 344.x
        if any(c.startswith(('342', '344')) for c in codes_set):
            count += 1
        
        # 8. Other neurological disorders: 330.x-337.x, 340.x, 341.x, 345.x, 348.1, 348.3, 780.3, 784.3
        if any(c.startswith(('330', '331', '332', '333', '334', '335', '336', '337', '340', '341', '345')) or
               c in ('348.1', '348.3', '780.3', '784.3') for c in codes_set):
            count += 1
        
        # 9. Chronic pulmonary disease: 490.x-505.x, 506.4
        if any(c.startswith(tuple(str(i) for i in range(490, 506))) or c == '506.4' for c in codes_set):
            count += 1
        
        # 10. Diabetes (uncomplicated): 250.0-250.3
        diabetes_uncomp_elix = any(c.startswith(('2500', '2501', '2502', '2503')) for c in codes_set)
        # Diabetes (complicated): 250.4-250.9
        diabetes_comp_elix = any(c.startswith(('2504', '2505', '2506', '2507', '2508', '2509')) for c in codes_set)
        if diabetes_comp_elix:
            count += 1
        elif diabetes_uncomp_elix:
            count += 1
        
        # 11. Hypothyroidism: 243.x, 244.x
        if any(c.startswith(('243', '244')) for c in codes_set):
            count += 1
        
        # 12. Renal failure: 585.x, 586.x, V42.0, V45.1, V56.x
        if any(c.startswith(('585', '586', 'V56')) or c in ('V42.0', 'V45.1') for c in codes_set):
            count += 1
        
        # 13. Liver disease: 070.x, 570.x-573.x, V42.7
        if any(c.startswith(('070', '570', '571', '572', '573')) or c == 'V42.7' for c in codes_set):
            count += 1
        
        # 14. Peptic ulcer disease (excluding bleeding): 531.x-534.x
        if any(c.startswith(('531', '532', '533', '534')) for c in codes_set):
            count += 1
        
        # 15. AIDS/HIV: 042.x-044.x
        if any(c.startswith(('042', '043', '044')) for c in codes_set):
            count += 1
        
        # 16. Lymphoma: 200.x-202.x, 203.0, 238.6
        if any(c.startswith(('200', '201', '202')) or c in ('203.0', '238.6') for c in codes_set):
            count += 1
        
        # 17. Metastatic cancer: 196.x-199.x
        if any(c.startswith(('196', '197', '198', '199')) for c in codes_set):
            count += 1
        
        # 18. Solid tumor without metastasis: 140.x-172.x, 174.x-195.x
        if any(c.startswith(tuple(str(i) for i in range(140, 173))) or 
               c.startswith(tuple(str(i) for i in range(174, 196))) for c in codes_set):
            count += 1
        
        # 19. Rheumatoid arthritis/collagen vascular diseases: 701.0, 710.x, 714.x, 720.x, 725.x
        if any(c == '701.0' or c.startswith(('710', '714', '720', '725')) for c in codes_set):
            count += 1
        
        # 20. Coagulopathy: 286.x, 287.1, 287.3-287.5
        if any(c.startswith('286') or c in ('287.1', '287.3', '287.4', '287.5') for c in codes_set):
            count += 1
        
        # 21. Obesity: 278.0x
        if any(c.startswith('2780') for c in codes_set):
            count += 1
        
        # 22. Weight loss: 260.x-263.x, 783.2x
        if any(c.startswith(('260', '261', '262', '263', '7832')) for c in codes_set):
            count += 1
        
        # 23. Fluid and electrolyte disorders: 276.x
        if any(c.startswith('276') for c in codes_set):
            count += 1
        
        # 24. Blood loss anemia: 280.0
        if any(c == '280.0' for c in codes_set):
            count += 1
        
        # 25. Deficiency anemia: 280.x-281.x
        if any(c.startswith(('280', '281')) for c in codes_set):
            count += 1
        
        # 26. Alcohol abuse: 291.x, 303.x, 305.0x
        if any(c.startswith(('291', '303', '3050')) for c in codes_set):
            count += 1
        
        # 27. Drug abuse: 292.x, 304.x, 305.2-305.9
        if any(c.startswith(('292', '304')) or 
               c.startswith('305') and len(c) > 3 and c[3] in '23456789' for c in codes_set):
            count += 1
        
        # 28. Psychoses: 295.x-298.x, 299.1x
        if any(c.startswith(('295', '296', '297', '298')) or c.startswith('2991') for c in codes_set):
            count += 1
        
        # 29. Depression: 300.4, 301.12, 309.x, 311.x
        if any(c in ('300.4', '301.12') or c.startswith(('309', '311')) for c in codes_set):
            count += 1
        
        return count
    
    # Binary comorbidity flags
    def is_hypertension(code):
        return code.startswith(('401', '402', '403', '404', '405'))
        
    def is_cvd(code):
        return code.startswith(('410', '411', '412', '413', '414', '428'))
    
    df['Hypertension'] = df['ICD9_CODE'].apply(is_hypertension).astype(int)
    df['Cardiovascular'] = df['ICD9_CODE'].apply(is_cvd).astype(int)
    
    # Calculate CCI for each admission
    cci_scores = df.groupby('HADM_ID')['ICD9_CODE'].apply(lambda x: calculate_cci(x.tolist())).reset_index()
    cci_scores.columns = ['HADM_ID', 'CCI_Score']
    
    # Calculate Elixhauser for each admission
    elix_scores = df.groupby('HADM_ID')['ICD9_CODE'].apply(lambda x: calculate_elixhauser(x.tolist())).reset_index()
    elix_scores.columns = ['HADM_ID', 'Elixhauser_Index']
    
    # Get binary flags
    comorb = df.groupby('HADM_ID')[['Hypertension', 'Cardiovascular']].max().reset_index()
    
    # Merge CCI and Elixhauser scores with comorbidities
    comorb = comorb.merge(cci_scores, on='HADM_ID', how='left')
    comorb = comorb.merge(elix_scores, on='HADM_ID', how='left')
    comorb['CCI_Score'] = comorb['CCI_Score'].fillna(0)
    comorb['Elixhauser_Index'] = comorb['Elixhauser_Index'].fillna(0)
    
    return comorb

def get_ventilation(conn, cohort_df):
    print("Extracting Ventilation Usage...")
    hadm_ids = list(cohort_df['HADM_ID'].unique())
    
    def chunker(seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))
        
    dfs = []
    for chunk in chunker(hadm_ids, 5000):
        chunk_str = str(tuple(map(int, chunk))) if len(chunk) > 1 else f"({int(chunk[0])})"
        query = f"""
        SELECT HADM_ID, ICD9_CODE
        FROM PROCEDURES_ICD
        WHERE HADM_ID IN {chunk_str}
        AND ICD9_CODE IN ('9670', '9671', '9672')
        """
        try:
            chunk_df = pd.read_sql_query(query, conn)
            chunk_df['HADM_ID'] = pd.to_numeric(chunk_df['HADM_ID'], errors='coerce')
            dfs.append(chunk_df)
        except Exception as e:
            print(f"Error reading procedures chunk: {e}")
            
    if not dfs:
        return pd.DataFrame({'HADM_ID': cohort_df['HADM_ID'].unique(), 'Ventilation_Usage': 0})
        
    df = pd.concat(dfs, ignore_index=True)
    vent_hadms = df['HADM_ID'].unique()
    
    vent_df = pd.DataFrame({'HADM_ID': cohort_df['HADM_ID'].unique()})
    vent_df['Ventilation_Usage'] = vent_df['HADM_ID'].isin(vent_hadms).astype(int)
    
    return vent_df

def main(emb_npz=None, emb_index=None, output_path=None):
    print("Starting feature extraction...")
    print(f"Database: {DB_PATH}")
    print(f"Cohort: {COHORT_PATH}")
    if output_path is None:
        output_path = OUTPUT_PATH
    print(f"Output: {output_path}")
    
    conn = get_db_connection()
    cohort_df = load_cohort()
    print(f"Loaded cohort: {len(cohort_df)} patients\n")
    
    demo_df = get_demographics(conn, cohort_df)
    print(f"Demographics extracted: {len(demo_df)} rows\n")
    
    labs_df = get_lab_events(conn, cohort_df)
    print(f"Lab events extracted: {len(labs_df)} rows\n")
    
    vitals_df = get_chart_events(conn, cohort_df)
    print(f"Chart events extracted: {len(vitals_df)} rows\n")
    
    urine_df = get_urine_output(conn, cohort_df)
    print(f"Urine output extracted: {len(urine_df)} rows\n")
    
    pao2_fio2_df = get_pao2_fio2_ratio(conn, cohort_df)
    print(f"PaO2/FiO2 ratio extracted: {len(pao2_fio2_df)} rows\n")
    
    comorb_df = get_comorbidities(conn, cohort_df)
    print(f"Comorbidities extracted: {len(comorb_df)} rows\n")
    
    vent_df = get_ventilation(conn, cohort_df)
    print(f"Ventilation extracted: {len(vent_df)} rows\n")
    
    print("Merging all features...")
    final_df = demo_df.merge(labs_df, on='HADM_ID', how='left')
    final_df = final_df.merge(vitals_df, on='ICUSTAY_ID', how='left')
    final_df = final_df.merge(urine_df, on='ICUSTAY_ID', how='left')
    final_df = final_df.merge(pao2_fio2_df, on='ICUSTAY_ID', how='left')
    final_df = final_df.merge(comorb_df, on='HADM_ID', how='left')
    final_df = final_df.merge(vent_df, on='HADM_ID', how='left')

    # New derived features: range and percentage change for each lab/vital
    # For labs we computed BUN_Min/BUN_Max etc., create RANGE and PCT_CHANGE
    def add_range_pct(df, root):
        min_col = f"{root}_Min"
        max_col = f"{root}_Max"
        avg_col = f"{root}_Avg"
        range_col = f"{root}_Range"
        pct_col = f"{root}_PctChange"
        if min_col in df.columns and max_col in df.columns:
            df[range_col] = df[max_col] - df[min_col]
            if avg_col in df.columns:
                df[pct_col] = df[range_col] / (df[avg_col].abs() + 1e-9)
            else:
                df[pct_col] = df[range_col]
        return df

    lab_roots = ['BUN', 'Creatinine', 'Glucose', 'HMG', 'WBC', 'Platelet', 'Anion_Gap', 'PTT', 'Lactate', 'Albumin']
    vital_roots = ['SBP', 'DBP', 'HeartRate', 'RespRate', 'SpO2']
    for root in lab_roots + vital_roots:
        final_df = add_range_pct(final_df, root)

    # --- NEW: add measured indicators for meaningful-absence features ---
    # Prefer measured columns list in feature metadata if available. This keeps measured flag names
    # consistent with `cohort/feature_metadata.json` (e.g. 'Lactate_measured') and avoids guessing.
    metadata_path = os.path.join(os.path.dirname(__file__), '..', 'cohort', 'feature_metadata.json')
    try:
        with open(os.path.abspath(metadata_path), 'r') as mf:
            metadata = json.load(mf)
            measured_cols = metadata.get('measured_cols', [])
    except Exception:
        measured_cols = []

    # If metadata lists measured_cols, use the roots to determine which columns to inspect in final_df
    if measured_cols:
        for mcol in measured_cols:
            # expected format in metadata: '<Root>_measured'
            if not mcol.endswith('_measured'):
                final_df[mcol] = 0
                continue

            root = mcol.replace('_measured', '')
            # candidate check columns - try commonly used aggregation names
            candidates = [f'{root}_Min', f'{root}_Max', f'{root}_Avg', f'{root}_Std', f'{root}_Median']
            existing = [c for c in candidates if c in final_df.columns]
            if existing:
                final_df[mcol] = final_df[existing].notna().any(axis=1).astype(int)
            else:
                # no related measurement columns found -> default to 0
                final_df[mcol] = 0
    else:
        # fallback: previous behavior - derive measured flags by common lab/vital names
        lab_names = ['BUN','Creatinine','Glucose','HMG','WBC','Platelet','Lactate','Albumin','Anion_Gap','PTT']
        for name in lab_names:
            cols = [f'{name}_Min', f'{name}_Max', f'{name}_Avg', f'{name}_Std']
            existing = [c for c in cols if c in final_df.columns]
            if existing:
                final_df[f'{name}_measured'] = final_df[existing].notna().any(axis=1).astype(int)
            else:
                final_df[f'{name}_measured'] = 0

        vital_groups = {
            'GCS': ['GCS_Median', 'GCS_Min', 'GCS_Max'],
            'SBP': ['SBP_Min', 'SBP_Max', 'SBP_Avg', 'SBP_Std'],
            'DBP': ['DBP_Min', 'DBP_Max', 'DBP_Avg', 'DBP_Std'],
            'HeartRate': ['HeartRate_Min', 'HeartRate_Max', 'HeartRate_Avg', 'HeartRate_Std'],
            'RespRate': ['RespRate_Min', 'RespRate_Max', 'RespRate_Avg', 'RespRate_Std'],
            'SpO2': ['SpO2_Min', 'SpO2_Max', 'SpO2_Avg', 'SpO2_Std']
        }
        for v, cols in vital_groups.items():
            existing = [c for c in cols if c in final_df.columns]
            if existing:
                final_df[f'{v}_measured'] = final_df[existing].notna().any(axis=1).astype(int)
            else:
                final_df[f'{v}_measured'] = 0

        # Urine output and PaO2/FiO2 ratio measured indicator (fallback)
        if any(c in final_df.columns for c in ['UrineOutput_Median','UrineOutput_Avg','UrineOutput_Max']):
            existing = [c for c in ['UrineOutput_Median','UrineOutput_Avg','UrineOutput_Max'] if c in final_df.columns]
            final_df['UrineOutput_measured'] = final_df[existing].notna().any(axis=1).astype(int)
        else:
            final_df['UrineOutput_measured'] = 0

        if 'PaO2_FiO2_Ratio_Median' in final_df.columns:
            final_df['PaO2_FiO2_measured'] = final_df['PaO2_FiO2_Ratio_Median'].notna().astype(int)
        else:
            final_df['PaO2_FiO2_measured'] = 0
    
    print(f"Final dataset shape: {final_df.shape}")
    print(f"Columns: {final_df.columns.tolist()}\n")
    
    # Outlier capping is disabled by default (user requested no automatic capping)
    print(f"Saving to {output_path}...")
    # Merge in note embeddings if provided
    if emb_npz and emb_index:
        try:
            idxdf = pd.read_csv(emb_index, dtype={'HADM_ID': object})
            npz = np.load(emb_npz)
            X = npz['X']
            emb_df = pd.DataFrame(X, columns=[f'note_emb_{i}' for i in range(X.shape[1])])
            # Ensure HADM_ID types match for merging
            emb_df['HADM_ID'] = pd.to_numeric(idxdf['HADM_ID'], errors='coerce').astype('Int64').astype('int64')
            final_df['HADM_ID'] = pd.to_numeric(final_df['HADM_ID'], errors='coerce').astype('Int64').astype('int64')
            final_df = final_df.merge(emb_df, on='HADM_ID', how='left')
            # fill NA embeddings with zeros
            emb_cols = [c for c in final_df.columns if c.startswith('note_emb_')]
            final_df[emb_cols] = final_df[emb_cols].fillna(0.0)
            print(f"Merged embeddings (shape {len(emb_df)} x {X.shape[1]})")
            # compute NLP features: clusters & keyword counts
            try:
                nlp_df = get_nlp_summary_features(emb_npz=emb_npz, emb_index=emb_index, n_clusters=8)
                final_df = final_df.merge(nlp_df, on='HADM_ID', how='left')
                # fill missing NLP features
                nlp_cols = [c for c in final_df.columns if c.startswith('note_cluster') or c.startswith('kw_') or 'note_word_count' in c]
                for c in nlp_cols:
                    final_df[c] = final_df[c].fillna(0)
                print(f"Merged NLP summary features (shape {len(nlp_df)} x {nlp_df.shape[1]})")
            except Exception as e:
                print('NLP feature merge failed:', e)
        except Exception as e:
            print('Failed to merge embeddings:', e)
    # ---- Add interactions & abnormal flag features ----
    # interactions: Ventilation * LOS_Hospital, Ventilation * Hypertension
    if 'Ventilation_Usage' in final_df.columns and 'LOS_Hospital' in final_df.columns:
        final_df['Ventilation_x_LOS_Hospital'] = final_df['Ventilation_Usage'] * final_df['LOS_Hospital']
    if 'Ventilation_Usage' in final_df.columns and 'Hypertension' in final_df.columns:
        final_df['Ventilation_x_Hypertension'] = final_df['Ventilation_Usage'] * final_df['Hypertension']

    # abnormal lab flag (last out of cohort-level range using 1.5*IQR)
    lab_roots = ['BUN', 'Creatinine', 'Glucose', 'HMG', 'WBC', 'Platelet', 'Anion_Gap', 'PTT', 'Lactate', 'Albumin']
    for name in lab_roots:
        last_col = f'{name}_Last'
        ab_col = f'{name}_abnormal_last'
        if last_col in final_df.columns:
            arr = final_df[last_col].dropna().values
            if len(arr) > 0:
                q1 = np.percentile(arr, 25)
                q3 = np.percentile(arr, 75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                final_df[ab_col] = ((~final_df[last_col].isna()) & ((final_df[last_col] < lower) | (final_df[last_col] > upper))).astype(int)
            else:
                final_df[ab_col] = 0
        else:
            final_df[ab_col] = 0

    # create count of abnormal labs
    ab_cols = [f'{name}_abnormal_last' for name in lab_roots if f'{name}_abnormal_last' in final_df.columns]
    if ab_cols:
        final_df['num_abnormal_labs'] = final_df[ab_cols].sum(axis=1)

    final_df.to_csv(output_path, index=False)
    print("Done!")
    print(f"\nSummary:")
    print(f"  - Total patients: {len(final_df)}")
    print(f"  - Total features: {len(final_df.columns)}")
    print(f"  - Missing values: {final_df.isnull().sum().sum()}")
    # ---- Add interactions & abnormal flag features ----
    # interactions: Ventilation * LOS_Hospital, Ventilation * Hypertension
    if 'Ventilation_Usage' in final_df.columns and 'LOS_Hospital' in final_df.columns:
        final_df['Ventilation_x_LOS_Hospital'] = final_df['Ventilation_Usage'] * final_df['LOS_Hospital']
    if 'Ventilation_Usage' in final_df.columns and 'Hypertension' in final_df.columns:
        final_df['Ventilation_x_Hypertension'] = final_df['Ventilation_Usage'] * final_df['Hypertension']

    # abnormal lab flag (last out of range): if last exists and min/max exist
    lab_roots = ['BUN', 'Creatinine', 'Glucose', 'HMG', 'WBC', 'Platelet', 'Anion_Gap', 'PTT', 'Lactate', 'Albumin']
    for name in lab_roots:
        last_col = f'{name}_Last'
        min_col = f'{name}_Min'
        max_col = f'{name}_Max'
        ab_col = f'{name}_abnormal_last'
        if last_col in final_df.columns and min_col in final_df.columns and max_col in final_df.columns:
            final_df[ab_col] = (~final_df[[last_col, min_col, max_col]].isnull().any(axis=1)) & ((final_df[last_col] < final_df[min_col]) | (final_df[last_col] > final_df[max_col]))
            final_df[ab_col] = final_df[ab_col].astype(int)
        else:
            final_df[ab_col] = 0

    # create count of abnormal labs
    ab_cols = [f'{name}_abnormal_last' for name in lab_roots if f'{name}_abnormal_last' in final_df.columns]
    if ab_cols:
        final_df['num_abnormal_labs'] = final_df[ab_cols].sum(axis=1)
    
    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb-npz', type=str, default=None, help='Path to npz containing embeddings with key X')
    parser.add_argument('--emb-index', type=str, default=None, help='Path to index CSV mapping embeddings to HADM_ID')
    parser.add_argument('--out', type=str, default=None, help='Output path for features.csv (defaults to configured OUTPUT_PATH)')
    args = parser.parse_args()
    main(emb_npz=args.emb_npz, emb_index=args.emb_index, output_path=args.out)
